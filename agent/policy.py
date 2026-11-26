"""Shared PPO actor/critic and the platform's inference entry point."""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn


class Network(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, 6)
        self.critic = nn.Linear(128, 1)

    def forward(self, grid):
        hidden = self.body(grid)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class Policy:
    def __init__(self, model_dir, device="cpu"):
        torch.set_num_threads(1)
        weights = load_file(str(Path(model_dir) / "model.safetensors"))
        self.device = device
        self.network = Network(weights["body.1.weight"].shape[1]).to(device)
        self.network.load_state_dict(weights)
        self.network.eval()

    def reset(self, *, seed, player_index, layout_id, scenario_spec_id):
        self.rng = torch.Generator().manual_seed(seed)

    @torch.inference_mode()
    def act(self, observation):
        grid = torch.as_tensor(observation["grid"], device=self.device).unsqueeze(0)
        logits, _ = self.network(grid)
        return torch.multinomial(logits.softmax(-1).cpu(), 1, generator=self.rng).item()
