"""The submitted policy: one recurrent network per map, inference only. zsc/package.py copies this file into the
package as policy.py, so it imports nothing from this repository.

The network reads the grid (CoordConv into three SE residual blocks), its own and the partner's previous action,
its seat and the time left, and keeps a two-layer GRU memory. A partner head predicts the partner's next action
and feeds its predicted distribution into the actor head.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

MAX_H, MAX_W = 7, 9  # every map is zero-padded to this size
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
        # 1D reduction and 3D multiply avoid a slow 2D-mean / 4D-broadcast path on MPS.
        pooled = h.flatten(-2).mean(dim=-1, keepdim=True).unsqueeze(-1)
        se = torch.sigmoid(self.se_fc2(self.act(self.se_fc1(pooled)))).squeeze(-1)
        return self.act(x + (h.flatten(-2) * se).view_as(h))


class ZSCStudentNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.hidden_size = hidden
        self.num_layers = 2
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
        self.partner_head = nn.Linear(hidden, 6)
        self.actor = nn.Sequential(nn.Linear(hidden + 6, head_dim), nn.SiLU(), nn.Linear(head_dim, 6))

        init_orthogonal(self.stem, math.sqrt(2.0))
        init_orthogonal(self.res_blocks, math.sqrt(2.0))
        init_orthogonal(self.spatial_proj, math.sqrt(2.0))
        init_orthogonal(self.memory, 1.0)
        init_orthogonal(self.partner_head, 1.0)
        init_orthogonal(self.actor[0], math.sqrt(2.0))
        init_orthogonal(self.actor[-1], 0.01)

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
        hidden: torch.Tensor | None = None,  # (2, B, hidden)
        return_features: bool = False,
    ):
        b, t = grid.shape[:2]
        feat = self.encode_spatial(grid.reshape(b * t, *grid.shape[2:])).reshape(b, t, -1)
        meta = torch.stack((seat, time_left), dim=-1).to(dtype=feat.dtype)
        x = torch.cat((feat, self.prev_self(prev_self).to(dtype=feat.dtype),
                       self.prev_partner(prev_partner).to(dtype=feat.dtype), meta), dim=-1)

        # Training sequences can span an episode boundary: zero the memory wherever an episode starts.
        if t == 1:
            if hidden is not None:
                hidden = hidden * (prev_self[:, 0] != START).to(dtype=hidden.dtype).view(1, b, 1)
            out, hidden = self.memory(x, hidden)
        else:
            h = hidden if hidden is not None else x.new_zeros((self.num_layers, b, self.hidden_size))
            resets = (prev_self == START).any(dim=0).nonzero(as_tuple=False).flatten().tolist()
            if not resets:
                out, hidden = self.memory(x, h)
            else:
                outs, start = [], 0
                for step in resets:
                    if step > start:
                        chunk, h = self.memory(x[:, start:step], h)
                        outs.append(chunk)
                    h = h * (prev_self[:, step] != START).to(dtype=h.dtype).view(1, b, 1)
                    start = step
                chunk, h = self.memory(x[:, start:], h)
                outs.append(chunk)
                out, hidden = torch.cat(outs, dim=1), h

        out = self.memory_norm(out)
        partner_logits = self.partner_head(out)
        predicted_partner = torch.softmax(partner_logits.float(), dim=-1).detach().to(dtype=out.dtype)
        logits = self.actor(torch.cat((out, predicted_partner), dim=-1))

        if return_features:
            return logits, partner_logits, out, hidden
        return logits, partner_logits, hidden


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    """fp32 state dict from a package file: int8 matrices with per-row `<name>.scale`, the rest fp16."""
    raw = load_file(str(path))
    return {
        k: v.float() * raw[f"{k}.scale"].float().view(-1, *[1] * (v.ndim - 1)) if v.dtype == torch.int8 else v.float()
        for k, v in raw.items()
        if not k.endswith(".scale")
    }


class Policy:
    """The course interface: loads one `<layout>.safetensors` network per map and acts greedily."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):  # the course's signature; runs on the CPU
        torch.set_num_threads(1)
        root = Path(model_dir)
        self.models = {}
        for path in sorted(root.glob("*.safetensors")):
            net = ZSCStudentNet()
            net.load_state_dict(load_weights(path))
            self.models[path.stem] = net.eval().requires_grad_(False)
        self.reset(seed=0, player_index=0, layout_id=None, scenario_spec_id=None)

    def reset(self, *, seed, player_index, layout_id, scenario_spec_id):
        self.hidden = None

    def act(self, observation: dict) -> int:
        layout, seat, timestep = observation["layout_id"], observation["player_index"], observation["timestep"]
        previous = observation["previous_joint_action"]
        if timestep == 0:
            self.hidden = None
        p_self, p_other = (START, START) if previous is None else (int(previous[seat]), int(previous[1 - seat]))

        def scalar(value, dtype=torch.float32):
            return torch.tensor([[value]], dtype=dtype)

        with torch.inference_mode():
            logits, _, self.hidden = self.models[layout](
                torch.from_numpy(np.asarray(observation["grid"], dtype=np.float32))[None, None],
                scalar(p_self, torch.long), scalar(p_other, torch.long),
                scalar(float(seat)), scalar(1.0 - timestep / observation["horizon"]), self.hidden,
            )
        return int(logits[0, 0].argmax())
