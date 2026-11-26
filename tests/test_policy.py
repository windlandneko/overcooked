"""The submission package: int8 weights act like the trained fp32 ones, and memory survives a missing
`previous_joint_action`."""

from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import load_file

from core.actions import Action
from core.env import policy_observation, read_scenario
from core.scenario import encode_observations
from zsc import CONFIG, MAPS, ROOT
from zsc.engine import VecEnv
from zsc.evaluate import specs, teacher, teacher_members
from zsc.policy import START, Policy, ZSCStudentNet, load_weights

AGENT = ROOT / "runs/submission/agent"
TRAINED = ROOT / "runs/students"


@torch.inference_mode()
def test_int8_matches_fp32(count: int = 16, steps: int = 200) -> None:
    for layout in MAPS:
        env = VecEnv(read_scenario(CONFIG, layout_id=layout), count)
        rng = np.random.default_rng(0)
        grids, actions = [], []
        for _ in range(steps):
            grids.append(env.obs.clone())
            actions.append(torch.from_numpy(rng.integers(0, 6, size=(count, 2))))
            env.step(actions[-1].numpy())
        grid = torch.stack(grids, 1)[:, :, 0]  # seat 0's view, (count, steps, 26, H, W)
        joint = torch.stack(actions, 1)
        start = torch.full((count, 1), START)
        prev_self, prev_partner = torch.cat((start, joint[:, :-1, 0]), 1), torch.cat((start, joint[:, :-1, 1]), 1)
        seat, time_left = torch.zeros(count, steps), 1 - torch.arange(steps).expand(count, -1) / 400
        logits = []
        for state in (load_file(str(TRAINED / layout / "best.safetensors")), load_weights(AGENT / f"{layout}.safetensors")):
            net = ZSCStudentNet().eval()
            net.load_state_dict(state)
            logits.append(net(grid, prev_self, prev_partner, seat, time_left)[0])
        agree = (logits[0].argmax(-1) == logits[1].argmax(-1)).float().mean()
        assert agree > 0.98, f"{layout}: int8 picks the fp32 action in {agree:.1%} of steps"


def test_partner_action_read_from_grid(layout: str = "counter_circuit", seed: int = 28301) -> None:
    """The partner action read off two grids reproduces the observed transition."""
    spec = specs()[layout]
    mdp = spec.create_mdp()

    def advance(state, joint):
        return mdp.get_state_transition(state, tuple(Action.ALL_ACTIONS[a] for a in joint))[0]

    same = total = 0
    for name in teacher_members(layout, "dev")[:4]:
        for seat in (0, 1):
            policy = Policy(AGENT)
            policy.reset(seed=seed, player_index=seat, layout_id=layout, scenario_spec_id=spec.fingerprint)
            partner = teacher(name, spec, seed + 1)
            state = mdp.get_standard_start_state()
            while state.timestep < spec.horizon:
                grids = encode_observations(mdp, state, spec.horizon)
                own = policy.act(policy_observation(spec, mdp, state, seat, grids[seat], None))
                joint = [own, partner(grids[1 - seat])][:: 1 if seat == 0 else -1]
                after = advance(state, joint)
                seen = encode_observations(mdp, after, spec.horizon)[seat]
                joint[1 - seat] = Policy.partner_action_from_grids(grids[seat], seen)
                same += advance(state, joint) == after
                total += 1
                state = after
    assert same / total > 0.99, f"the read action reproduces {same}/{total} transitions"
