"""The submitted policy: one recurrent network per map, inference only. zsc/package.py copies this file into the
package as policy.py, so it imports nothing from this repository.

The network reads the grid (CoordConv into three SE residual blocks), its own and the partner's previous action,
its seat and the time left, and keeps a two-layer GRU memory. A partner head predicts the partner's next action
under 12 partner types; a Bayesian belief over those types, updated from the partner's actual moves, mixes the
predictions and the type-specific corrections of the action logits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

MAX_H, MAX_W = 7, 9  # every map is zero-padded to this size
NUM_TYPES = 12
START = 6  # previous-action token at an episode start


def pad_grid(grid: torch.Tensor) -> torch.Tensor:
    h, w = grid.shape[-2:]
    if (h, w) == (MAX_H, MAX_W):
        return grid
    out = grid.new_zeros((*grid.shape[:-2], MAX_H, MAX_W))
    out[..., :h, :w] = grid
    return out


def coord_grid(height: int = MAX_H, width: int = MAX_W) -> torch.Tensor:
    xs = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width).expand(1, 1, height, width)
    ys = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1).expand(1, 1, height, width)
    return torch.cat((xs, ys), dim=1)


def init_orthogonal(module: nn.Module, gain: float) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=gain)
                elif "bias" in name:
                    nn.init.zeros_(param)


def update_belief(
    belief: torch.Tensor,
    prev_partner: torch.Tensor,
    prev_logits: torch.Tensor | None,
    eta: float = 1.0,
    share: float = 0.05,
) -> torch.Tensor:
    """Fixed-share Bayes step: weight each type by how well it predicted the partner's last action.

    `eta` tempers the likelihood and `share` mixes back a little uniform mass so the belief can follow a
    partner that changes its style mid-game. At an episode start there is no prediction yet: uniform.
    """
    k = belief.shape[-1]
    uniform = belief.new_full(belief.shape, 1.0 / k)
    if prev_logits is None:
        return uniform
    valid = prev_partner < START
    log_probs = torch.log_softmax(prev_logits.float(), dim=-1)  # (B, K, 6)
    log_like = log_probs.gather(-1, prev_partner.clamp(0, 5).view(-1, 1, 1).expand(-1, k, 1)).squeeze(-1)
    posterior = torch.softmax(torch.log(belief.float().clamp_min(1e-12)) + eta * log_like, dim=-1)
    return torch.where(valid.unsqueeze(-1), (1.0 - share) * posterior + share / k, uniform)


class SEResBlock(nn.Module):
    def __init__(self, channels: int = 64, groups: int = 8, se_hidden: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.se_fc1 = nn.Conv2d(channels, se_hidden, kernel_size=1)
        self.se_fc2 = nn.Conv2d(se_hidden, channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        se = torch.sigmoid(self.se_fc2(self.act(self.se_fc1(h.mean(dim=(-2, -1), keepdim=True)))))
        return self.act(x + h * se)


class ZSCStudentNet(nn.Module):
    def __init__(self, hidden: int = 256, num_archetypes: int = NUM_TYPES, belief_eta: float = 1.0,
                 belief_gamma: float = 0.05):
        super().__init__()
        self.hidden_size = hidden
        self.num_layers = 2
        self.num_archetypes = num_archetypes
        self.belief_eta = belief_eta
        self.belief_gamma = belief_gamma
        self.register_buffer("_coord_grid", coord_grid(), persistent=False)

        self.stem = nn.Sequential(nn.Conv2d(28, 64, kernel_size=3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.SiLU())
        self.res_blocks = nn.Sequential(SEResBlock(), SEResBlock(), SEResBlock())
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(64, 24, kernel_size=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(24 * MAX_H * MAX_W, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.prev_self = nn.Embedding(7, 8)
        self.prev_partner = nn.Embedding(7, 8)
        self.memory = nn.GRU(hidden + 18, hidden, num_layers=self.num_layers, batch_first=True)
        self.memory_norm = nn.LayerNorm(hidden)

        head_dim = max(64, hidden // 2)
        self.partner_head = nn.Linear(hidden, num_archetypes * 6)
        self.actor = nn.Sequential(nn.Linear(hidden + num_archetypes + 6, head_dim), nn.SiLU(), nn.Linear(head_dim, 6))
        self.actor_moe = nn.Linear(hidden, num_archetypes * 6)  # per-type corrections of the action logits

        init_orthogonal(self.stem, math.sqrt(2.0))
        init_orthogonal(self.res_blocks, math.sqrt(2.0))
        init_orthogonal(self.spatial_proj, math.sqrt(2.0))
        init_orthogonal(self.memory, 1.0)
        init_orthogonal(self.partner_head, 1.0)
        init_orthogonal(self.actor[0], math.sqrt(2.0))
        init_orthogonal(self.actor[-1], 0.01)
        nn.init.zeros_(self.actor_moe.weight)  # starts as a plain actor; the corrections are learned
        nn.init.zeros_(self.actor_moe.bias)

    def encode_spatial(self, grid: torch.Tensor) -> torch.Tensor:
        """(N, 26, H, W) -> (N, hidden)."""
        padded = pad_grid(grid)
        coords = self._coord_grid.to(dtype=padded.dtype, device=padded.device).expand(padded.shape[0], -1, -1, -1)
        return self.spatial_proj(self.res_blocks(self.stem(torch.cat((padded, coords), dim=1))))

    def forward(
        self,
        grid: torch.Tensor,          # (B, T, 26, H, W)
        prev_self: torch.Tensor,     # (B, T) int64, START at an episode start
        prev_partner: torch.Tensor,  # (B, T) int64, START at an episode start
        seat: torch.Tensor,          # (B, T)
        time_left: torch.Tensor,     # (B, T), 1 at the start and 0 at the horizon
        hidden: torch.Tensor | None = None,               # (2, B, hidden)
        belief: torch.Tensor | None = None,               # (B, K)
        prev_partner_logits: torch.Tensor | None = None,  # (B, K, 6)
        return_features: bool = False,
    ):
        b, t = grid.shape[:2]
        feat = self.encode_spatial(grid.reshape(b * t, *grid.shape[2:])).reshape(b, t, -1)
        meta = torch.stack((seat, time_left), dim=-1)
        x = torch.cat((feat, self.prev_self(prev_self), self.prev_partner(prev_partner), meta), dim=-1)

        # Training sequences can span an episode boundary: zero the memory wherever an episode starts.
        if t == 1:
            if hidden is not None:
                hidden = hidden * (prev_self[:, 0] != START).to(dtype=hidden.dtype).view(1, b, 1)
            out, hidden = self.memory(x, hidden)
        else:
            h = hidden if hidden is not None else x.new_zeros((self.num_layers, b, self.hidden_size))
            outs = []
            for step in range(t):
                h = h * (prev_self[:, step] != START).to(dtype=h.dtype).view(1, b, 1)
                out_s, h = self.memory(x[:, step : step + 1], h)
                outs.append(out_s)
            out, hidden = torch.cat(outs, dim=1), h

        out = self.memory_norm(out)
        partner_logits = self.partner_head(out).reshape(b, t, self.num_archetypes, 6)

        with torch.no_grad():
            w = belief.float() if belief is not None else out.new_full((b, self.num_archetypes), 1.0 / self.num_archetypes,
                                                                        dtype=torch.float32)
            beliefs, prev_logits = [], prev_partner_logits
            for step in range(t):
                w = update_belief(w, prev_partner[:, step], prev_logits, eta=self.belief_eta, share=self.belief_gamma)
                beliefs.append(w)
                prev_logits = partner_logits[:, step].detach()
            beliefs = torch.stack(beliefs, dim=1)

        p_probs = torch.softmax(partner_logits.float(), dim=-1)  # (B, T, K, 6)
        predicted_partner = (beliefs.unsqueeze(-1) * p_probs).sum(dim=-2).detach().to(dtype=out.dtype)
        weights = beliefs.to(dtype=out.dtype)
        base_logits = self.actor(torch.cat((out, weights, predicted_partner), dim=-1))
        corrections = self.actor_moe(out).reshape(b, t, self.num_archetypes, 6)
        logits = base_logits + (weights.unsqueeze(-1) * corrections).sum(dim=-2)

        if return_features:
            return logits, partner_logits, out, hidden, w
        return logits, partner_logits, hidden, w


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    """fp32 state dict from a package file: int8 matrices with per-row `<name>.scale`, the rest fp16."""
    raw = load_file(str(path))
    return {
        k: v.float() * raw[f"{k}.scale"].float().view(-1, *[1] * (v.ndim - 1)) if v.dtype == torch.int8 else v.float()
        for k, v in raw.items()
        if not k.endswith(".scale")
    }


class Policy:
    """The course interface. `meta.json` maps each map to its belief `eta` and sampling `temperature` (0: argmax)."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):  # the course's signature; runs on the CPU
        torch.set_num_threads(1)
        root = Path(model_dir)
        meta = json.loads((root / "meta.json").read_text())
        self.temperatures = {layout: m["temperature"] for layout, m in meta.items()}
        self.models = {}
        for layout, m in meta.items():
            net = ZSCStudentNet(belief_eta=m["eta"])
            net.load_state_dict(load_weights(root / f"{layout}.safetensors"))
            self.models[layout] = net.eval().requires_grad_(False)
        self.reset(seed=0, player_index=0, layout_id=None, scenario_spec_id=None)

    def reset(self, *, seed, player_index, layout_id, scenario_spec_id):
        self.rng = np.random.default_rng(seed + 17 * player_index)
        self.clear_memory()

    def clear_memory(self) -> None:
        self.hidden = self.belief = self.prev_partner_logits = self.last_grid = None
        self.step_count = 0
        self.last_action = START

    @staticmethod
    def partner_action_from_grids(old: np.ndarray, new: np.ndarray) -> int:
        """The partner's last action read off two consecutive grids, for platforms that do not pass
        `previous_joint_action`: a move or turn names the direction, a change of the partner's held item or of
        the tile it faces means interact, otherwise stay. Channel 1 is the partner's cell, 6-9 its facing."""
        y0, x0 = np.argwhere(old[1] > 0)[0]
        y1, x1 = np.argwhere(new[1] > 0)[0]
        moves = {(-1, 0): 0, (1, 0): 1, (0, 1): 2, (0, -1): 3}
        if (int(y1 - y0), int(x1 - x0)) in moves:
            return moves[int(y1 - y0), int(x1 - x0)]
        facing_old, facing_new = int(old[6:10, y0, x0].argmax()), int(new[6:10, y1, x1].argmax())
        if facing_new != facing_old:
            return facing_new
        fy, fx = y1 + (-1, 1, 0, 0)[facing_new], x1 + (0, 0, 1, -1)[facing_new]
        held, placed = [18, 19, 21, 22, 23, 24], [16, 17, 22, 23, 24]
        if not np.array_equal(old[held, y1, x1], new[held, y1, x1]):
            return 5
        if 0 <= fy < new.shape[1] and 0 <= fx < new.shape[2] and not np.array_equal(old[placed, fy, fx],
                                                                                     new[placed, fy, fx]):
            return 5
        return 4

    def act(self, observation: dict) -> int:
        layout, seat, timestep = observation["layout_id"], observation["player_index"], observation["timestep"]
        grid = np.array(observation["grid"], dtype=np.float32)
        # The memory is cleared only when an episode starts, never because `previous_joint_action` is missing.
        if timestep == 0 and self.step_count > 0:
            self.clear_memory()
        if self.hidden is None:
            p_self = p_other = START
        elif (previous := observation.get("previous_joint_action")) is not None:
            p_self, p_other = int(previous[seat]), int(previous[1 - seat])
        else:
            p_self, p_other = self.last_action, self.partner_action_from_grids(self.last_grid, grid)

        def scalar(value, dtype=torch.float32):
            return torch.tensor([[value]], dtype=dtype)

        with torch.inference_mode():
            logits, p_logits, self.hidden, self.belief = self.models[layout](
                torch.from_numpy(grid)[None, None], scalar(p_self, torch.long), scalar(p_other, torch.long),
                scalar(float(seat)), scalar(1.0 - timestep / observation["horizon"]),
                self.hidden, self.belief, self.prev_partner_logits,
            )
        self.prev_partner_logits = p_logits[:, -1]
        temperature = self.temperatures[layout]
        if temperature <= 0:
            action = int(logits[0, 0].argmax())
        else:
            action = int(self.rng.choice(6, p=torch.softmax(logits[0, 0] / temperature, dim=-1).numpy()))
        self.step_count = timestep + 1
        self.last_action, self.last_grid = action, grid
        return action
