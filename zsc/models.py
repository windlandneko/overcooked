"""Training-side networks: partners, their batched forms, and the student's privileged critic.
The submitted policy itself lives in zsc.policy.ZSCStudentNet."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution

from zsc.policy import coord_grid, init_orthogonal


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def supports_bf16(device: torch.device) -> bool:
    return (device.type == "cuda" and torch.cuda.is_bf16_supported()) or device.type == "mps"


def setup(seed: int, device: str) -> tuple[torch.device, bool]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    Distribution.set_default_validate_args(False)
    return dev, supports_bf16(dev)


def autocast(device: torch.device, enabled: bool) -> torch.autocast:
    dev_type = device.type if device.type in ("cuda", "mps") else "cpu"
    return torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=enabled)


def gae(reward: torch.Tensor, done: torch.Tensor, value: torch.Tensor, next_value: torch.Tensor,
        gamma: float, lam: float) -> torch.Tensor:
    adv = torch.zeros_like(reward)
    last = torch.zeros_like(next_value)
    for t in reversed(range(len(reward))):
        non_term = 1.0 - done[t]
        delta = reward[t] + gamma * (next_value if t == len(reward) - 1 else value[t + 1]) * non_term - value[t]
        adv[t] = last = delta + gamma * lam * non_term * last
    return adv


def extract_privileged_obs_feats(p_obs: torch.Tensor) -> torch.Tensor:
    """10-dim summary from the partner's view: held items (onion/tomato/dish/soup), counter counts
    (onion/dish/soup), cooking/ready pots, and normalized Manhattan distance between chefs."""
    h, w = p_obs.shape[-2:]
    p_pos, e_pos = p_obs[:, 0:1], p_obs[:, 1:2]
    held_items = (p_pos * p_obs[:, (23, 24, 22, 21)]).sum(dim=(-2, -1))
    counter_items = (p_obs[:, 11:12] * p_obs[:, (23, 22, 21)]).sum(dim=(-2, -1)) * 0.25

    pots = p_obs[:, 10] > 0.5
    cooking_pots = (pots & (p_obs[:, 20] > 0.5)).float().sum(dim=(-2, -1)).unsqueeze(-1)
    ready_pots = (pots & (p_obs[:, 21] > 0.5)).float().sum(dim=(-2, -1)).unsqueeze(-1)

    xs = torch.arange(w, device=p_obs.device, dtype=p_obs.dtype).view(1, 1, w)
    ys = torch.arange(h, device=p_obs.device, dtype=p_obs.dtype).view(1, h, 1)
    l1_dist = (
        ((p_pos[:, 0] * xs).sum(dim=(-2, -1)) - (e_pos[:, 0] * xs).sum(dim=(-2, -1))).abs()
        + ((p_pos[:, 0] * ys).sum(dim=(-2, -1)) - (e_pos[:, 0] * ys).sum(dim=(-2, -1))).abs()
    ).unsqueeze(-1) / float(h + w)

    return torch.cat((held_items, counter_items, cooking_pots, ready_pots, l1_dist), dim=-1)


class PrivilegedCriticNet(nn.Module):
    """Training-only CTDE value head conditioned on partner slot id and 15 privileged scalars."""

    PRIV_DIM = 15

    def __init__(self, hidden_size: int = 256, n_slots: int = 64, slot_emb_dim: int = 24):
        super().__init__()
        self.slot_emb = nn.Embedding(n_slots, slot_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_size + slot_emb_dim + self.PRIV_DIM, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        init_orthogonal(self.net[:-1], math.sqrt(2.0))
        init_orthogonal(self.net[-1], 1.0)

    def forward(self, out: torch.Tensor, slot_id: torch.Tensor, priv_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat((out, self.slot_emb(slot_id).to(dtype=out.dtype), priv_vec.to(dtype=out.dtype)), dim=-1)
        return self.net(x).squeeze(-1)


class PartnerNet(nn.Module):
    """Small CoordConv actor-critic for population partners."""

    def __init__(self, height: int, width: int, hidden: int = 128):
        super().__init__()
        self.height, self.width = height, width
        self.register_buffer("_coord_grid", coord_grid(height, width), persistent=False)
        self.conv = nn.Sequential(
            nn.Conv2d(28, 24, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * height * width + 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.actor = nn.Linear(hidden, 6)
        self.critic = nn.Linear(hidden, 1)
        init_orthogonal(self.conv, math.sqrt(2.0))
        init_orthogonal(self.head, math.sqrt(2.0))
        init_orthogonal(self.actor, 0.01)
        init_orthogonal(self.critic, 1.0)

    def forward(self, grid: torch.Tensor, seat: torch.Tensor, time_left: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coords = self._coord_grid.to(dtype=grid.dtype, device=grid.device).expand(grid.shape[0], -1, -1, -1)
        feat = self.conv(torch.cat((grid, coords), dim=1))
        h = self.head(torch.cat((feat, torch.stack((seat, time_left), dim=-1)), dim=-1))
        return self.actor(h), self.critic(h).squeeze(-1)


class VecPartnerNet(nn.Module):
    """S independent PartnerNets trained in one pass via grouped conv and batched matmul."""

    def __init__(self, num_styles: int, height: int, width: int, hidden: int = 128):
        super().__init__()
        self.num_styles, self.height, self.width, self.hidden = num_styles, height, width, hidden
        self.flat_dim = 32 * height * width
        self.register_buffer("_coord_grid", coord_grid(height, width), persistent=False)

        self.conv1_w = nn.Parameter(torch.empty(num_styles * 24, 28, 3, 3))
        self.conv1_b = nn.Parameter(torch.zeros(num_styles * 24))
        self.conv2_w = nn.Parameter(torch.empty(num_styles * 32, 24, 3, 3))
        self.conv2_b = nn.Parameter(torch.zeros(num_styles * 32))

        self.fc1_w = nn.Parameter(torch.empty(num_styles, self.flat_dim + 2, hidden))
        self.fc1_b = nn.Parameter(torch.zeros(num_styles, 1, hidden))
        self.ln_w = nn.Parameter(torch.ones(num_styles, 1, hidden))
        self.ln_b = nn.Parameter(torch.zeros(num_styles, 1, hidden))
        self.fc2_w = nn.Parameter(torch.empty(num_styles, hidden, hidden))
        self.fc2_b = nn.Parameter(torch.zeros(num_styles, 1, hidden))

        self.act_w = nn.Parameter(torch.empty(num_styles, hidden, 6))
        self.act_b = nn.Parameter(torch.zeros(num_styles, 1, 6))
        self.cri_w = nn.Parameter(torch.empty(num_styles, hidden, 1))
        self.cri_b = nn.Parameter(torch.zeros(num_styles, 1, 1))

        def init_linear_t(param: nn.Parameter, gain: float) -> None:
            for s in range(num_styles):
                w = torch.empty(param.shape[2], param.shape[1])
                nn.init.orthogonal_(w, gain=gain)
                param.data[s].copy_(w.T)

        g_h = math.sqrt(2.0)
        for s in range(num_styles):
            nn.init.orthogonal_(self.conv1_w.data[s * 24 : (s + 1) * 24], gain=g_h)
            nn.init.orthogonal_(self.conv2_w.data[s * 32 : (s + 1) * 32], gain=g_h)
        init_linear_t(self.fc1_w, g_h)
        init_linear_t(self.fc2_w, g_h)
        init_linear_t(self.act_w, 0.01)
        init_linear_t(self.cri_w, 1.0)

    def forward(self, grid: torch.Tensor, seat: torch.Tensor, time_left: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, b = grid.shape[:2]
        coords = self._coord_grid.to(dtype=grid.dtype, device=grid.device).view(1, 1, 2, self.height, self.width).expand(s, b, -1, -1, -1)
        x = torch.cat((grid, coords), dim=2).permute(1, 0, 2, 3, 4).reshape(b, s * 28, self.height, self.width)
        x = F.silu(F.conv2d(x, self.conv1_w, self.conv1_b, padding=1, groups=s))
        x = F.silu(F.conv2d(x, self.conv2_w, self.conv2_b, padding=1, groups=s))
        feat = x.reshape(b, s, self.flat_dim).permute(1, 0, 2)

        meta = torch.stack((seat, time_left), dim=-1).to(dtype=feat.dtype)
        h = torch.baddbmm(self.fc1_b, torch.cat((feat, meta), dim=-1), self.fc1_w)
        h = F.silu(F.layer_norm(h, (self.hidden,), eps=1e-5) * self.ln_w + self.ln_b)
        h = F.silu(torch.baddbmm(self.fc2_b, h, self.fc2_w))
        return torch.baddbmm(self.act_b, h, self.act_w), torch.baddbmm(self.cri_b, h, self.cri_w).squeeze(-1)

    def clip_grad_norm_per_style(self, max_norm: float = 0.5) -> None:
        """Clip each style's gradient independently, matching single-style PPO."""
        per_style = (self.fc1_w, self.fc1_b, self.ln_w, self.ln_b,
                     self.fc2_w, self.fc2_b, self.act_w, self.act_b, self.cri_w, self.cri_b)
        for s in range(self.num_styles):
            grads = [p.grad[s] for p in per_style if p.grad is not None]
            if self.conv1_w.grad is not None:
                grads.extend((
                    self.conv1_w.grad[s * 24 : (s + 1) * 24], self.conv1_b.grad[s * 24 : (s + 1) * 24],
                    self.conv2_w.grad[s * 32 : (s + 1) * 32], self.conv2_b.grad[s * 32 : (s + 1) * 32],
                ))
            if not grads:
                continue
            clip_coef = max_norm / (sum(g.float().pow(2).sum() for g in grads).sqrt() + 1e-6)
            if clip_coef < 1.0:
                for g in grads:
                    g.mul_(clip_coef.to(dtype=g.dtype))

    def export_single_state_dict(self, s: int) -> dict[str, torch.Tensor]:
        return {
            "conv.0.weight": self.conv1_w.detach()[s * 24 : (s + 1) * 24].cpu().contiguous(),
            "conv.0.bias": self.conv1_b.detach()[s * 24 : (s + 1) * 24].cpu().contiguous(),
            "conv.2.weight": self.conv2_w.detach()[s * 32 : (s + 1) * 32].cpu().contiguous(),
            "conv.2.bias": self.conv2_b.detach()[s * 32 : (s + 1) * 32].cpu().contiguous(),
            "head.0.weight": self.fc1_w.detach()[s].T.cpu().contiguous(),
            "head.0.bias": self.fc1_b.detach()[s, 0].cpu().contiguous(),
            "head.1.weight": self.ln_w.detach()[s, 0].cpu().contiguous(),
            "head.1.bias": self.ln_b.detach()[s, 0].cpu().contiguous(),
            "head.3.weight": self.fc2_w.detach()[s].T.cpu().contiguous(),
            "head.3.bias": self.fc2_b.detach()[s, 0].cpu().contiguous(),
            "actor.weight": self.act_w.detach()[s].T.cpu().contiguous(),
            "actor.bias": self.act_b.detach()[s, 0].cpu().contiguous(),
            "critic.weight": self.cri_w.detach()[s].T.cpu().contiguous(),
            "critic.bias": self.cri_b.detach()[s, 0].cpu().contiguous(),
        }


class BatchedPartnerPool(nn.Module):
    """P frozen PartnerNet actors evaluated in one pass."""

    def __init__(self, partners: list[PartnerNet], height: int, width: int, device: torch.device):
        super().__init__()
        self.num_partners, self.height, self.width = len(partners), height, width
        self.flat_dim = 32 * height * width
        self.hidden = partners[0].actor.in_features

        self.register_buffer("_coord_grid", coord_grid(height, width).to(device), persistent=False)
        self.register_buffer("conv1_w", torch.cat([p.conv[0].weight.detach() for p in partners], dim=0).to(device))
        self.register_buffer("conv1_b", torch.cat([p.conv[0].bias.detach() for p in partners], dim=0).to(device))
        self.register_buffer("conv2_w", torch.cat([p.conv[2].weight.detach() for p in partners], dim=0).to(device))
        self.register_buffer("conv2_b", torch.cat([p.conv[2].bias.detach() for p in partners], dim=0).to(device))

        self.register_buffer("fc1_w", torch.stack([p.head[0].weight.detach().T for p in partners]).to(device))
        self.register_buffer("fc1_b", torch.stack([p.head[0].bias.detach() for p in partners]).unsqueeze(1).to(device))
        self.register_buffer("ln_w", torch.stack([p.head[1].weight.detach() for p in partners]).unsqueeze(1).to(device))
        self.register_buffer("ln_b", torch.stack([p.head[1].bias.detach() for p in partners]).unsqueeze(1).to(device))
        self.register_buffer("fc2_w", torch.stack([p.head[3].weight.detach().T for p in partners]).to(device))
        self.register_buffer("fc2_b", torch.stack([p.head[3].bias.detach() for p in partners]).unsqueeze(1).to(device))
        self.register_buffer("act_w", torch.stack([p.actor.weight.detach().T for p in partners]).to(device))
        self.register_buffer("act_b", torch.stack([p.actor.bias.detach() for p in partners]).unsqueeze(1).to(device))

    def forward(self, grid: torch.Tensor, seat: torch.Tensor, time_left: torch.Tensor) -> torch.Tensor:
        p, m = grid.shape[:2]
        dt = grid.dtype
        coords = self._coord_grid.to(dtype=dt, device=grid.device).view(1, 1, 2, self.height, self.width).expand(p, m, -1, -1, -1)
        x = torch.cat((grid, coords), dim=2).permute(1, 0, 2, 3, 4).reshape(m, p * 28, self.height, self.width)
        x = F.silu(F.conv2d(x, self.conv1_w.to(dt), self.conv1_b.to(dt), padding=1, groups=p))
        x = F.silu(F.conv2d(x, self.conv2_w.to(dt), self.conv2_b.to(dt), padding=1, groups=p))
        feat = x.reshape(m, p, self.flat_dim).permute(1, 0, 2)

        meta = torch.stack((seat, time_left), dim=-1).to(dtype=feat.dtype)
        h = torch.baddbmm(self.fc1_b.to(feat.dtype), torch.cat((feat, meta), dim=-1), self.fc1_w.to(feat.dtype))
        h = F.silu(F.layer_norm(h, (self.hidden,), eps=1e-5) * self.ln_w.to(feat.dtype) + self.ln_b.to(feat.dtype))
        h = F.silu(torch.baddbmm(self.fc2_b.to(feat.dtype), h, self.fc2_w.to(feat.dtype)))
        return torch.baddbmm(self.act_b.to(feat.dtype), h, self.act_w.to(feat.dtype))


_POOL_CACHE: dict = {}


def adaptive_avg_pool2d_any(x: torch.Tensor, out: int) -> torch.Tensor:
    """`F.adaptive_avg_pool2d(x, (out, out))` as a matmul over identical bins:
    MPS lacks adaptive pooling when input size is not a multiple of output size (e.g. 5x9 -> 3x3)."""
    h, w = x.shape[-2:]
    key = (h, w, out, x.device, x.dtype)
    if (mat := _POOL_CACHE.get(key)) is None:
        with torch.inference_mode(False):  # normal tensor so cached weights work under both inference and autograd
            mat = _POOL_CACHE[key] = _pool_matrix(h, w, out).to(device=x.device, dtype=x.dtype)
    return (x.flatten(-2) @ mat).unflatten(-1, (out, out))


def _pool_matrix(h: int, w: int, out: int) -> torch.Tensor:
    m = torch.zeros(out * out, h * w, dtype=torch.float64)
    for i in range(out):
        h0, h1 = (i * h) // out, -((-(i + 1) * h) // out)
        for j in range(out):
            w0, w1 = (j * w) // out, -((-(j + 1) * w) // out)
            cell = torch.zeros(h, w, dtype=torch.float64)
            cell[h0:h1, w0:w1] = 1.0 / ((h1 - h0) * (w1 - w0))
            m[i * out + j] = cell.reshape(-1)
    return m.T.contiguous()


class TeacherNet(nn.Module):
    """Matches the course partner packs' PartnerNetwork state-dict layout."""

    def __init__(self, height: int = 0, width: int = 0, hidden_size: int = 64, spatial_pool: int = 3):
        super().__init__()
        self.spatial_pool = spatial_pool
        self.body = nn.Sequential(
            nn.Conv2d(26, 32, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((spatial_pool, spatial_pool)),
            nn.Flatten(),
            nn.Linear(32 * spatial_pool**2, hidden_size),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_size, 6)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(grid))


class BatchedTeacherPool(nn.Module):
    """G frozen TeacherNets evaluated in fp32 to match the course FrozenPartner's logits."""

    def __init__(self, teachers: list[TeacherNet], device: torch.device):
        super().__init__()
        self.num_partners = len(teachers)
        self.spatial_pool = teachers[0].spatial_pool
        self.flat_dim = 32 * self.spatial_pool**2
        self.register_buffer("conv1_w", torch.cat([t.body[0].weight.detach().float() for t in teachers], dim=0).to(device))
        self.register_buffer("conv1_b", torch.cat([t.body[0].bias.detach().float() for t in teachers], dim=0).to(device))
        self.register_buffer("conv2_w", torch.cat([t.body[2].weight.detach().float() for t in teachers], dim=0).to(device))
        self.register_buffer("conv2_b", torch.cat([t.body[2].bias.detach().float() for t in teachers], dim=0).to(device))
        self.register_buffer("fc_w", torch.stack([t.body[6].weight.detach().float().T for t in teachers]).to(device))
        self.register_buffer("fc_b", torch.stack([t.body[6].bias.detach().float() for t in teachers]).unsqueeze(1).to(device))
        self.register_buffer("head_w", torch.stack([t.head.weight.detach().float().T for t in teachers]).to(device))
        self.register_buffer("head_b", torch.stack([t.head.bias.detach().float() for t in teachers]).unsqueeze(1).to(device))

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        g, m, c, h, w = grid.shape
        with torch.autocast(device_type=grid.device.type, enabled=False):
            x = grid.float().permute(1, 0, 2, 3, 4).reshape(m, g * c, h, w)
            x = torch.tanh(F.conv2d(x, self.conv1_w, self.conv1_b, padding=1, groups=g))
            x = torch.tanh(F.conv2d(x, self.conv2_w, self.conv2_b, padding=1, groups=g))
            x = adaptive_avg_pool2d_any(x, self.spatial_pool).reshape(m, g, self.flat_dim).permute(1, 0, 2)
            x = torch.tanh(torch.baddbmm(self.fc_b, x, self.fc_w))
            return torch.baddbmm(self.head_b, x, self.head_w)
