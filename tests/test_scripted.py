"""Scripted partners: valid actions on standard and augmented states, disconnected rooms, and cooking."""

from __future__ import annotations

import numpy as np

from core.env import read_scenario
from zsc import CONFIG, MAPS
from zsc.engine import VecEnv
from zsc.scripted import INF, SCRIPTED_NAMES, get_nav, scripted_action


def test_scripted_valid_on_augmented_states() -> None:
    rng = np.random.default_rng(0)
    for layout in MAPS:
        spec = read_scenario(CONFIG, layout_id=layout)
        nav = get_nav(layout)
        env = VecEnv(spec, len(SCRIPTED_NAMES))
        env.seed(0)
        env.reset(aug_prob=1.0)
        for _ in range(40):
            obs = env.obs.numpy()
            acts = np.zeros((len(SCRIPTED_NAMES), 2), dtype=np.int32)
            for i, name in enumerate(SCRIPTED_NAMES):
                a0 = scripted_action(obs[i, 0], nav, name, rng)
                a1 = scripted_action(obs[i, 1], nav, "greedy_polite", rng)
                assert 0 <= a0 < 6 and 0 <= a1 < 6
                acts[i] = (a0, a1)
            env.step(acts, aug_prob=0.5)
        env.close()


def test_scripted_delivers_and_respects_rooms() -> None:
    nav_u = get_nav("unident_s_hard")
    # Left room (2, 1) and right room (2, 6) are disconnected components.
    assert nav_u.comp[2 * nav_u.w + 1] != nav_u.comp[2 * nav_u.w + 6]
    assert nav_u.base_dist[1 * nav_u.w + 0, 2 * nav_u.w + 6] >= INF  # right room cannot reach left onion box

    for layout in MAPS:
        spec = read_scenario(CONFIG, layout_id=layout)
        nav = get_nav(layout)
        env = VecEnv(spec, 1)
        env.reset()
        rng = np.random.default_rng(7)
        score = 0.0
        for _ in range(spec.horizon):
            obs = env.obs.numpy()[0]
            a0 = scripted_action(obs[0], nav, "onion_feeder", rng)
            a1 = scripted_action(obs[1], nav, "greedy_polite", rng)
            score += float(env.step(np.array([[a0, a1]], dtype=np.int32))[0][0])
        env.close()
        assert score >= 140.0, f"{layout}: onion_feeder + greedy_polite scored only {score}"
