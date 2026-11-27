"""Submission policy tests: chunked GRU parity, int8 quantization fidelity, and course interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from core.actions import Action
from core.env import policy_observation, read_scenario
from core.scenario import encode_observations
from zsc import CONFIG, MAPS
from zsc.engine import VecEnv
from zsc.package import quantize
from zsc.policy import START, Policy, ZSCStudentNet, load_weights


@torch.inference_mode()
def test_chunked_gru_matches_stepwise() -> None:
    torch.manual_seed(0)
    net = ZSCStudentNet(hidden=64).eval()
    b, t, h, w = 4, 24, 5, 9
    grid = torch.randn(b, t, 26, h, w)
    prev_self = torch.randint(0, 6, (b, t))
    prev_partner = torch.randint(0, 6, (b, t))
    for row, col in ((0, 0), (1, 7), (2, 19)):
        prev_self[row, col] = prev_partner[row, col] = START
    seat = torch.randint(0, 2, (b, t)).float()
    time_left = torch.linspace(1.0, 0.0, t).expand(b, -1)
    h0 = torch.randn(net.num_layers, b, net.hidden_size)

    logits_chunk, plogits_chunk, h_chunk = net(grid, prev_self, prev_partner, seat, time_left, h0.clone())

    h_step = h0.clone()
    logits_steps, plogits_steps = [], []
    for step in range(t):
        s = slice(step, step + 1)
        step_logits, step_plogits, h_step = net(grid[:, s], prev_self[:, s], prev_partner[:, s], seat[:, s], time_left[:, s], h_step)
        logits_steps.append(step_logits)
        plogits_steps.append(step_plogits)

    assert torch.allclose(logits_chunk, torch.cat(logits_steps, dim=1), atol=1e-5)
    assert torch.allclose(plogits_chunk, torch.cat(plogits_steps, dim=1), atol=1e-5)
    assert torch.allclose(h_chunk, h_step, atol=1e-5)


@torch.inference_mode()
def test_int8_matches_fp32(tmp_path: Path, count: int = 8, steps: int = 80) -> None:
    torch.manual_seed(0)
    for layout in MAPS:
        env = VecEnv(read_scenario(CONFIG, layout_id=layout), count)
        rng = np.random.default_rng(0)
        grids, actions = [], []
        for _ in range(steps):
            grids.append(env.obs.clone())
            actions.append(torch.from_numpy(rng.integers(0, 6, size=(count, 2))))
            env.step(actions[-1].numpy())
        grid = torch.stack(grids, 1)[:, :, 0]
        joint = torch.stack(actions, 1)
        start = torch.full((count, 1), START)
        prev_self, prev_partner = torch.cat((start, joint[:, :-1, 0]), 1), torch.cat((start, joint[:, :-1, 1]), 1)
        seat, time_left = torch.zeros(count, steps), 1 - torch.arange(steps).expand(count, -1) / 400

        fp32_net = ZSCStudentNet().eval()
        q_path = tmp_path / f"{layout}.safetensors"
        save_file(quantize(fp32_net.state_dict()), str(q_path))
        int8_net = ZSCStudentNet().eval()
        int8_net.load_state_dict(load_weights(q_path))

        agree = (fp32_net(grid, prev_self, prev_partner, seat, time_left)[0].argmax(-1)
                 == int8_net(grid, prev_self, prev_partner, seat, time_left)[0].argmax(-1)).float().mean()
        assert agree > 0.96, f"{layout}: int8 picks the fp32 action in {agree:.1%} of steps"


def test_policy_interface_and_reset(tmp_path: Path) -> None:
    torch.manual_seed(0)
    layout = "counter_circuit"
    save_file(quantize(ZSCStudentNet().state_dict()), str(tmp_path / f"{layout}.safetensors"))
    policy = Policy(tmp_path)
    spec = read_scenario(CONFIG, layout_id=layout)
    mdp = spec.create_mdp()

    def run_short() -> list[int]:
        policy.reset(seed=0, player_index=0, layout_id=layout, scenario_spec_id=spec.fingerprint)
        state, prev, actions = mdp.get_standard_start_state(), None, []
        for _ in range(10):
            grids = encode_observations(mdp, state, spec.horizon)
            own = policy.act(policy_observation(spec, mdp, state, 0, grids[0], prev))
            assert 0 <= own < 6
            actions.append(own)
            prev = (own, 4)
            state, _ = mdp.get_state_transition(state, tuple(Action.ALL_ACTIONS[a] for a in prev))
        return actions

    assert run_short() == run_short()
