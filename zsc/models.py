"""Training-side networks: partners, their batched forms, the student's privileged critic. The student
itself is zsc.policy.ZSCStudentNet, the class the submission runs."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution

from zsc.policy import coord_grid, init_orthogonal


def setup(seed: int, device: str) -> tuple[torch.device, bool]:
    """Seed torch and numpy; returns the device and whether it runs bf16 autocast."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    Distribution.set_default_validate_args(False)
    return dev, dev.type == "cuda" and torch.cuda.is_bf16_supported()


def autocast(enabled: bool) -> torch.autocast:
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def gae(reward: torch.Tensor, done: torch.Tensor, value: torch.Tensor, next_value: torch.Tensor,
        gamma: float, lam: float) -> torch.Tensor:
    """Advantages of a (T, ...) rollout."""
    adv = torch.zeros_like(reward)
    last = torch.zeros_like(next_value)
    for t in reversed(range(len(reward))):
        non_term = 1.0 - done[t]
        delta = reward[t] + gamma * (next_value if t == len(reward) - 1 else value[t + 1]) * non_term - value[t]
        last = delta + gamma * lam * non_term * last
        adv[t] = last
    return adv

# schelling is symmetric under a 180-degree turn (Other-Play): flipping the grid swaps north/south and
# east/west in the facing channels (2-5 own, 6-9 partner) and in the actions; stay, interact and the start
# token map to themselves.
ROT180_CH_PERM = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8] + list(range(10, 26))
ROT180_ACT_NP = np.array([1, 0, 3, 2, 4, 5, 6], dtype=np.int32)


def rot180_obs_gpu(obs: torch.Tensor, ch_perm: torch.Tensor) -> torch.Tensor:
    return obs.index_select(-3, ch_perm).flip(-2, -1)


def extract_privileged_obs_feats(p_obs: torch.Tensor) -> torch.Tensor:
    """(N, 10) summary of the partner's view (N, 26, H, W): the partner's held onion, tomato, dish, soup;
    onions, dishes and soups on counters; cooking and ready pots; normalized distance between the chefs."""
    h, w = p_obs.shape[-2], p_obs.shape[-1]
    p_pos = p_obs[:, 0:1]  # (N, 1, H, W) partner position
    e_pos = p_obs[:, 1:2]  # (N, 1, H, W) ego position
    held_items = (p_pos * p_obs[:, (23, 24, 22, 21)]).sum(dim=(-2, -1))  # (N, 4)

    counters = p_obs[:, 11:12]
    counter_items = (counters * p_obs[:, (23, 22, 21)]).sum(dim=(-2, -1)) * 0.25  # (N, 3)

    pots = p_obs[:, 10] > 0.5
    cooking_pots = (pots & (p_obs[:, 20] > 0.5)).float().sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)
    ready_pots = (pots & (p_obs[:, 21] > 0.5)).float().sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)

    xs = torch.arange(w, device=p_obs.device, dtype=p_obs.dtype).view(1, 1, w)
    ys = torch.arange(h, device=p_obs.device, dtype=p_obs.dtype).view(1, h, 1)
    px = (p_pos[:, 0] * xs).sum(dim=(-2, -1))
    py = (p_pos[:, 0] * ys).sum(dim=(-2, -1))
    ex = (e_pos[:, 0] * xs).sum(dim=(-2, -1))
    ey = (e_pos[:, 0] * ys).sum(dim=(-2, -1))
    l1_dist = ((px - ex).abs() + (py - ey).abs()).unsqueeze(-1) / float(h + w)

    return torch.cat((held_items, counter_items, cooking_pots, ready_pots, l1_dist), dim=-1)  # (N, 10)


class PrivilegedCriticNet(nn.Module):
    """Value of the student's GRU features given what only training knows: which partner slot it plays with and
    22 privileged numbers (12 on how that partner acts, 10 from extract_privileged_obs_feats). It is never
    packaged, so it may see things the policy cannot."""

    PRIV_DIM = 22

    def __init__(self, hidden_size: int = 256, n_slots: int = 64, slot_emb_dim: int = 24):
        super().__init__()
        self.slot_emb = nn.Embedding(n_slots, slot_emb_dim)
        in_dim = hidden_size + slot_emb_dim + self.PRIV_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        init_orthogonal(self.net[:-1], math.sqrt(2.0))
        init_orthogonal(self.net[-1], 1.0)

    def forward(
        self,
        out: torch.Tensor,       # (B, T, hidden)
        slot_id: torch.Tensor,   # (B, T)
        priv_vec: torch.Tensor,  # (B, T, 22)
    ) -> torch.Tensor:
        emb = self.slot_emb(slot_id)
        x = torch.cat((out, emb.to(dtype=out.dtype), priv_vec.to(dtype=out.dtype)), dim=-1)
        return self.net(x).squeeze(-1)


class PartnerNet(nn.Module):
    """Small CoordConv actor-critic of the partner population."""

    def __init__(self, height: int, width: int, hidden: int = 128):
        super().__init__()
        self.height = height
        self.width = width
        self.register_buffer("_coord_grid", coord_grid(height, width), persistent=False)
        self.conv = nn.Sequential(
            nn.Conv2d(28, 24, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        flat_dim = 32 * height * width
        self.head = nn.Sequential(
            nn.Linear(flat_dim + 2, hidden),
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

    def forward(
        self,
        grid: torch.Tensor,
        seat: torch.Tensor,
        time_left: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = self._coord_grid.to(dtype=grid.dtype, device=grid.device).expand(grid.shape[0], -1, -1, -1)
        feat = self.conv(torch.cat((grid, coords), dim=1))
        meta = torch.stack((seat, time_left), dim=-1)
        h = self.head(torch.cat((feat, meta), dim=-1))
        return self.actor(h), self.critic(h).squeeze(-1)


class VecPartnerNet(nn.Module):
    """S independent PartnerNets trained in one pass (grouped conv + batched matmul)."""

    def __init__(self, num_styles: int, height: int, width: int, hidden: int = 128):
        super().__init__()
        self.num_styles = num_styles
        self.height = height
        self.width = width
        self.hidden = hidden
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

        g_h = math.sqrt(2.0)
        for s in range(num_styles):
            nn.init.orthogonal_(self.conv1_w.data[s * 24 : (s + 1) * 24], gain=g_h)
            nn.init.orthogonal_(self.conv2_w.data[s * 32 : (s + 1) * 32], gain=g_h)
            w1 = torch.empty(hidden, self.flat_dim + 2)
            nn.init.orthogonal_(w1, gain=g_h)
            self.fc1_w.data[s].copy_(w1.T)
            w2 = torch.empty(hidden, hidden)
            nn.init.orthogonal_(w2, gain=g_h)
            self.fc2_w.data[s].copy_(w2.T)
            wa = torch.empty(6, hidden)
            nn.init.orthogonal_(wa, gain=0.01)
            self.act_w.data[s].copy_(wa.T)
            wc = torch.empty(1, hidden)
            nn.init.orthogonal_(wc, gain=1.0)
            self.cri_w.data[s].copy_(wc.T)

    def forward(
        self,
        grid: torch.Tensor,       # (S, B, 26, H, W)
        seat: torch.Tensor,       # (S, B)
        time_left: torch.Tensor,  # (S, B)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s, b = grid.shape[0], grid.shape[1]
        coords = self._coord_grid.to(dtype=grid.dtype, device=grid.device).view(1, 1, 2, self.height, self.width).expand(s, b, -1, -1, -1)
        x = torch.cat((grid, coords), dim=2)  # (S, B, 28, H, W)
        x = x.permute(1, 0, 2, 3, 4).reshape(b, s * 28, self.height, self.width)
        x = F.silu(F.conv2d(x, self.conv1_w, self.conv1_b, padding=1, groups=s))
        x = F.silu(F.conv2d(x, self.conv2_w, self.conv2_b, padding=1, groups=s))
        feat = x.reshape(b, s, self.flat_dim).permute(1, 0, 2)  # (S, B, flat_dim)

        meta = torch.stack((seat, time_left), dim=-1).to(dtype=feat.dtype)
        h = torch.baddbmm(self.fc1_b, torch.cat((feat, meta), dim=-1), self.fc1_w)
        h = F.layer_norm(h, (self.hidden,), eps=1e-5) * self.ln_w + self.ln_b
        h = F.silu(h)
        h = F.silu(torch.baddbmm(self.fc2_b, h, self.fc2_w))
        logits = torch.baddbmm(self.act_b, h, self.act_w)
        value = torch.baddbmm(self.cri_b, h, self.cri_w).squeeze(-1)
        return logits, value

    def clip_grad_norm_per_style(self, max_norm: float = 0.5) -> None:
        """Clip each style's gradient on its own, as if it were trained alone."""
        eps = 1e-6
        for s in range(self.num_styles):
            grads = []
            if self.conv1_w.grad is not None:
                grads.append(self.conv1_w.grad[s * 24 : (s + 1) * 24])
                grads.append(self.conv1_b.grad[s * 24 : (s + 1) * 24])
                grads.append(self.conv2_w.grad[s * 32 : (s + 1) * 32])
                grads.append(self.conv2_b.grad[s * 32 : (s + 1) * 32])
            for p in (
                self.fc1_w, self.fc1_b, self.ln_w, self.ln_b,
                self.fc2_w, self.fc2_b, self.act_w, self.act_b, self.cri_w, self.cri_b,
            ):
                if p.grad is not None:
                    grads.append(p.grad[s])
            if not grads:
                continue
            total_sq = sum(g.float().pow(2).sum() for g in grads)
            norm = total_sq.sqrt()
            clip_coef = max_norm / (norm + eps)
            if clip_coef < 1.0:
                for g in grads:
                    g.mul_(clip_coef.to(dtype=g.dtype))

    def export_single_state_dict(self, s: int) -> dict[str, torch.Tensor]:
        """Style `s` as a PartnerNet state dict."""
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
    """P frozen PartnerNets' actors in one pass: grid (P, M, 26, H, W) -> logits (P, M, 6)."""

    def __init__(self, partners: list[PartnerNet], height: int, width: int, device: torch.device):
        super().__init__()
        self.num_partners = len(partners)
        self.height = height
        self.width = width
        self.flat_dim = 32 * height * width
        self.hidden = partners[0].actor.in_features

        self.register_buffer("_coord_grid", coord_grid(height, width).to(device), persistent=False)
        self.register_buffer("conv1_w", torch.cat([p.conv[0].weight.detach() for p in partners], dim=0).to(device))
        self.register_buffer("conv1_b", torch.cat([p.conv[0].bias.detach() for p in partners], dim=0).to(device))
        self.register_buffer("conv2_w", torch.cat([p.conv[2].weight.detach() for p in partners], dim=0).to(device))
        self.register_buffer("conv2_b", torch.cat([p.conv[2].bias.detach() for p in partners], dim=0).to(device))

        self.register_buffer("fc1_w", torch.stack([p.head[0].weight.detach().T for p in partners], dim=0).to(device))
        self.register_buffer("fc1_b", torch.stack([p.head[0].bias.detach() for p in partners], dim=0).unsqueeze(1).to(device))
        self.register_buffer("ln_w", torch.stack([p.head[1].weight.detach() for p in partners], dim=0).unsqueeze(1).to(device))
        self.register_buffer("ln_b", torch.stack([p.head[1].bias.detach() for p in partners], dim=0).unsqueeze(1).to(device))
        self.register_buffer("fc2_w", torch.stack([p.head[3].weight.detach().T for p in partners], dim=0).to(device))
        self.register_buffer("fc2_b", torch.stack([p.head[3].bias.detach() for p in partners], dim=0).unsqueeze(1).to(device))
        self.register_buffer("act_w", torch.stack([p.actor.weight.detach().T for p in partners], dim=0).to(device))
        self.register_buffer("act_b", torch.stack([p.actor.bias.detach() for p in partners], dim=0).unsqueeze(1).to(device))

    def forward(
        self,
        grid: torch.Tensor,       # (P, M, 26, H, W)
        seat: torch.Tensor,       # (P, M)
        time_left: torch.Tensor,  # (P, M)
    ) -> torch.Tensor:
        p, m = grid.shape[0], grid.shape[1]
        coords = self._coord_grid.to(dtype=grid.dtype, device=grid.device).view(1, 1, 2, self.height, self.width).expand(p, m, -1, -1, -1)
        x = torch.cat((grid, coords), dim=2)
        x = x.permute(1, 0, 2, 3, 4).reshape(m, p * 28, self.height, self.width)
        x = F.silu(F.conv2d(x, self.conv1_w.to(dtype=x.dtype), self.conv1_b.to(dtype=x.dtype), padding=1, groups=p))
        x = F.silu(F.conv2d(x, self.conv2_w.to(dtype=x.dtype), self.conv2_b.to(dtype=x.dtype), padding=1, groups=p))
        feat = x.reshape(m, p, self.flat_dim).permute(1, 0, 2)

        meta = torch.stack((seat, time_left), dim=-1).to(dtype=feat.dtype)
        h = torch.baddbmm(self.fc1_b.to(dtype=feat.dtype), torch.cat((feat, meta), dim=-1), self.fc1_w.to(dtype=feat.dtype))
        h = F.layer_norm(h, (self.hidden,), eps=1e-5) * self.ln_w.to(dtype=feat.dtype) + self.ln_b.to(dtype=feat.dtype)
        h = F.silu(h)
        h = F.silu(torch.baddbmm(self.fc2_b.to(dtype=feat.dtype), h, self.fc2_w.to(dtype=feat.dtype)))
        return torch.baddbmm(self.act_b.to(dtype=feat.dtype), h, self.act_w.to(dtype=feat.dtype))


class MLPPartnerNet(nn.Module):
    """The course template's network (train.py): a 2x128 tanh MLP actor-critic."""

    def __init__(self, height: int, width: int, hidden: int = 128):
        super().__init__()
        self.height = height
        self.width = width
        self.obs_dim = 26 * height * width
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, 6)
        self.critic = nn.Linear(hidden, 1)
        init_orthogonal(self.net, math.sqrt(2.0))
        init_orthogonal(self.actor, 0.01)
        init_orthogonal(self.critic, 1.0)

    def forward(
        self,
        grid: torch.Tensor,
        seat: torch.Tensor | None = None,
        time_left: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(grid)
        return self.actor(h), self.critic(h).squeeze(-1)


class VecMLPPartnerNet(nn.Module):
    """S independent MLPPartnerNets trained in one pass."""

    def __init__(self, num_styles: int, height: int, width: int, hidden: int = 128):
        super().__init__()
        self.num_styles = num_styles
        self.height = height
        self.width = width
        self.obs_dim = 26 * height * width
        self.hidden = hidden

        self.fc1_w = nn.Parameter(torch.empty(num_styles, self.obs_dim, hidden))
        self.fc1_b = nn.Parameter(torch.zeros(num_styles, 1, hidden))
        self.fc2_w = nn.Parameter(torch.empty(num_styles, hidden, hidden))
        self.fc2_b = nn.Parameter(torch.zeros(num_styles, 1, hidden))
        self.act_w = nn.Parameter(torch.empty(num_styles, hidden, 6))
        self.act_b = nn.Parameter(torch.zeros(num_styles, 1, 6))
        self.cri_w = nn.Parameter(torch.empty(num_styles, hidden, 1))
        self.cri_b = nn.Parameter(torch.zeros(num_styles, 1, 1))

        g_h = math.sqrt(2.0)
        for s in range(num_styles):
            w1 = torch.empty(hidden, self.obs_dim)
            nn.init.orthogonal_(w1, gain=g_h)
            self.fc1_w.data[s].copy_(w1.T)
            w2 = torch.empty(hidden, hidden)
            nn.init.orthogonal_(w2, gain=g_h)
            self.fc2_w.data[s].copy_(w2.T)
            wa = torch.empty(6, hidden)
            nn.init.orthogonal_(wa, gain=0.01)
            self.act_w.data[s].copy_(wa.T)
            wc = torch.empty(1, hidden)
            nn.init.orthogonal_(wc, gain=1.0)
            self.cri_w.data[s].copy_(wc.T)

    def forward(self, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # grid (S, B, 26, H, W)
        s, b = grid.shape[0], grid.shape[1]
        x = grid.reshape(s, b, self.obs_dim)
        h = torch.tanh(torch.baddbmm(self.fc1_b, x, self.fc1_w))
        h = torch.tanh(torch.baddbmm(self.fc2_b, h, self.fc2_w))
        logits = torch.baddbmm(self.act_b, h, self.act_w)
        value = torch.baddbmm(self.cri_b, h, self.cri_w).squeeze(-1)
        return logits, value

    def clip_grad_norm_per_style(self, max_norm: float = 0.5) -> None:
        eps = 1e-6
        params = (self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b, self.act_w, self.act_b, self.cri_w, self.cri_b)
        for s in range(self.num_styles):
            grads = [p.grad[s] for p in params if p.grad is not None]
            if not grads:
                continue
            norm = sum(g.float().pow(2).sum() for g in grads).sqrt()
            clip_coef = max_norm / (norm + eps)
            if clip_coef < 1.0:
                for g in grads:
                    g.mul_(clip_coef.to(dtype=g.dtype))

    def export_single_state_dict(self, s: int) -> dict[str, torch.Tensor]:
        """Style `s` as an MLPPartnerNet state dict."""
        return {
            "net.1.weight": self.fc1_w.detach()[s].T.cpu().contiguous(),
            "net.1.bias": self.fc1_b.detach()[s, 0].cpu().contiguous(),
            "net.3.weight": self.fc2_w.detach()[s].T.cpu().contiguous(),
            "net.3.bias": self.fc2_b.detach()[s, 0].cpu().contiguous(),
            "actor.weight": self.act_w.detach()[s].T.cpu().contiguous(),
            "actor.bias": self.act_b.detach()[s, 0].cpu().contiguous(),
            "critic.weight": self.cri_w.detach()[s].T.cpu().contiguous(),
            "critic.bias": self.cri_b.detach()[s, 0].cpu().contiguous(),
        }


class BatchedMLPPool(nn.Module):
    """P frozen MLPPartnerNets' actors in one pass: grid (P, M, 26, H, W) -> logits (P, M, 6)."""

    def __init__(self, partners: list[MLPPartnerNet], height: int, width: int, device: torch.device):
        super().__init__()
        self.num_partners = len(partners)
        self.obs_dim = 26 * height * width
        self.register_buffer("fc1_w", torch.stack([p.net[1].weight.detach().T for p in partners], dim=0).to(device))
        self.register_buffer("fc1_b", torch.stack([p.net[1].bias.detach() for p in partners], dim=0).unsqueeze(1).to(device))
        self.register_buffer("fc2_w", torch.stack([p.net[3].weight.detach().T for p in partners], dim=0).to(device))
        self.register_buffer("fc2_b", torch.stack([p.net[3].bias.detach() for p in partners], dim=0).unsqueeze(1).to(device))
        self.register_buffer("act_w", torch.stack([p.actor.weight.detach().T for p in partners], dim=0).to(device))
        self.register_buffer("act_b", torch.stack([p.actor.bias.detach() for p in partners], dim=0).unsqueeze(1).to(device))

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        p, m = grid.shape[0], grid.shape[1]
        x = grid.reshape(p, m, self.obs_dim)
        dt = x.dtype
        h = torch.tanh(torch.baddbmm(self.fc1_b.to(dtype=dt), x, self.fc1_w.to(dtype=dt)))
        h = torch.tanh(torch.baddbmm(self.fc2_b.to(dtype=dt), h, self.fc2_w.to(dtype=dt)))
        return torch.baddbmm(self.act_b.to(dtype=dt), h, self.act_w.to(dtype=dt))


_POOL_CACHE: dict = {}


def adaptive_avg_pool2d_any(x: torch.Tensor, out: int) -> torch.Tensor:
    """`F.adaptive_avg_pool2d(x, (out, out))` as a matmul with the same bins, [floor(i*H/out), ceil((i+1)*H/out)):
    MPS has no adaptive pooling when the input size is not a multiple of the output size, e.g. 5x9 -> 3x3."""
    h, w = x.shape[-2:]
    key = (h, w, out, x.device, x.dtype)
    mat = _POOL_CACHE.get(key)
    if mat is None:
        with torch.inference_mode(False):  # a normal tensor, usable both in inference and in autograd
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
    """The course partner packs' PartnerNetwork (same state dict keys); it sees only its own grid."""

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
    """G frozen TeacherNets in one pass, always in fp32: grid (G, M, 26, H, W) -> logits (G, M, 6)."""

    def __init__(self, teachers: list[TeacherNet], device: torch.device):
        super().__init__()
        self.num_partners = len(teachers)
        self.spatial_pool = teachers[0].spatial_pool
        self.flat_dim = 32 * self.spatial_pool**2
        self.register_buffer("conv1_w", torch.cat([t.body[0].weight.detach().float() for t in teachers], dim=0).to(device))
        self.register_buffer("conv1_b", torch.cat([t.body[0].bias.detach().float() for t in teachers], dim=0).to(device))
        self.register_buffer("conv2_w", torch.cat([t.body[2].weight.detach().float() for t in teachers], dim=0).to(device))
        self.register_buffer("conv2_b", torch.cat([t.body[2].bias.detach().float() for t in teachers], dim=0).to(device))
        self.register_buffer("fc_w", torch.stack([t.body[6].weight.detach().float().T for t in teachers], dim=0).to(device))
        self.register_buffer("fc_b", torch.stack([t.body[6].bias.detach().float() for t in teachers], dim=0).unsqueeze(1).to(device))
        self.register_buffer("head_w", torch.stack([t.head.weight.detach().float().T for t in teachers], dim=0).to(device))
        self.register_buffer("head_b", torch.stack([t.head.bias.detach().float() for t in teachers], dim=0).unsqueeze(1).to(device))

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        g, m, c, h, w = grid.shape
        with torch.autocast(device_type=grid.device.type, enabled=False):
            x = grid.float().permute(1, 0, 2, 3, 4).reshape(m, g * c, h, w)
            x = torch.tanh(F.conv2d(x, self.conv1_w, self.conv1_b, padding=1, groups=g))
            x = torch.tanh(F.conv2d(x, self.conv2_w, self.conv2_b, padding=1, groups=g))
            x = adaptive_avg_pool2d_any(x, self.spatial_pool)
            x = x.reshape(m, g, self.flat_dim).permute(1, 0, 2)
            x = torch.tanh(torch.baddbmm(self.fc_b, x, self.fc_w))
            return torch.baddbmm(self.head_b, x, self.head_w)
