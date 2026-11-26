"""The C++ engine against core/env.py, and its augmented restarts."""

from __future__ import annotations

import numpy as np
import torch

from core.env import SelfPlayEnv, read_scenario, read_scenarios
from core.scenario import ScenarioSpec
from zsc import CONFIG, MAPS, ROOT
from zsc.engine import EVENTS, VecEnv
from zsc.models import BatchedTeacherPool
from zsc.teachers import PACK, load_teacher_members

SPECS = read_scenarios(CONFIG) + [read_scenario(ROOT / "config.json", scenario_id=slot) for slot in ("slot-1", "slot-2")]
ITEM_CHANNELS = list(range(16, 25))


def assert_same(ref: SelfPlayEnv, env: VecEnv, joint: np.ndarray, where: str) -> np.ndarray:
    expected = ref.step(joint)
    ref.reset_done(expected[2])
    for name, want, got in zip(("sparse", "shaped", "done"), expected, env.step(joint)):
        np.testing.assert_array_equal(got, want, err_msg=f"{where} {name}")
    np.testing.assert_array_equal(env.obs.numpy(), ref.observe(), err_msg=f"{where} obs")
    return expected[0]


def test_random_play(steps: int = 800, count: int = 8) -> None:
    rng = np.random.default_rng(42)
    for spec in SPECS:
        ref = SelfPlayEnv(spec, count)
        ref.activate()
        env = VecEnv(spec, count)
        np.testing.assert_array_equal(env.obs.numpy(), ref.observe())
        for t in range(steps):
            assert_same(ref, env, rng.integers(0, 6, size=(count, 2), dtype=np.int32), f"{spec.layout_id} t={t}")


@torch.inference_mode()
def test_teacher_play(per_member: int = 2) -> None:
    """Random actions almost never deliver a soup; teacher pairs do, so cooking and serving are compared too."""
    for layout in MAPS:
        spec = read_scenario(CONFIG, layout_id=layout)
        members = [m for m in load_teacher_members(PACK, layout, ("train",), torch.device("cpu"))
                   if m.stage in ("best", "final")][:8]
        count = len(members) * per_member
        pool = BatchedTeacherPool([m.net for m in members], torch.device("cpu"))
        generator = torch.Generator().manual_seed(0)
        ref = SelfPlayEnv(spec, count)
        ref.activate()
        env = VecEnv(spec, count)
        score = 0.0
        for t in range(spec.horizon + 50):
            logits = pool(env.obs.reshape(len(members), per_member * 2, *env.obs.shape[2:]))
            joint = torch.multinomial(logits.reshape(-1, 6).softmax(-1), 1, generator=generator)
            score += assert_same(ref, env, joint.reshape(count, 2).int().numpy(), f"{layout} t={t}").sum()
        assert score > 20 * count, f"{layout}: teachers delivered only {score / count:.0f} per game"


def test_full_idle_pot_and_long_cook() -> None:
    """Three onions wait in the pot while a dish is fetched (no dish reward), then a 137-step cook and delivery."""
    spec = ScenarioSpec("cramped_room", cook_time=137, horizon=190)
    ref = SelfPlayEnv(spec, 1)
    ref.activate()
    env = VecEnv(spec, 1)
    seat0 = "ULIRUI" + "LLIRUI" * 2 + ".IL" + "." * 140
    seat1 = ".DLLD" + "." * 13 + "IRU" + "." * 135 + "IRDDI"
    for t, pair in enumerate(zip(seat0, seat1), 1):
        assert_same(ref, env, np.array([["UDRL.I".index(a) for a in pair]], dtype=np.int32), f"t={t}")
        if t == 19:
            assert env.shaped[0] == env.events[0, 1, EVENTS.index("dish_pickup")] == 0
        if t == 20:
            assert env.obs[0, 0, 20, 0, 2] == 136  # cook time left
    assert env.sparse[0] == spec.reward and env.events[0, 1, EVENTS.index("soup_deliver")] == 1


def neighbours(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def reachable_counters(grid: np.ndarray) -> np.ndarray:
    """Counter cells next to a floor tile either chef can reach, from a clean (26, H, W) grid."""
    floor = grid[10:16].sum(0) == 0
    seen = grid[0] + grid[1] > 0
    while (grown := seen | neighbours(seen) & floor).sum() > seen.sum():
        seen = grown
    return (grid[11] > 0) & neighbours(seen)


def test_augmentation(count: int = 2048) -> None:
    for spec in SPECS:
        env = VecEnv(spec, count)
        env.seed(3)
        reachable = reachable_counters(env.obs[0, 0].numpy())
        env.reset(aug_prob=1.0)  # the ring is empty, so every game gets a random kitchen
        clutter = env.obs[:, 0, ITEM_CHANNELS].sum(1).numpy() > 0
        counters = env.obs[0, 0, 11].numpy() > 0
        assert not clutter[:, counters & ~reachable].any(), f"{spec.layout_id}: clutter on unreachable counters"
        assert clutter[:, reachable].any()
        pot_items = env.obs[:, 0, 16:20].sum(1).numpy()  # idle and cooking onions and tomatoes
        assert (pot_items <= 3).all(), f"{spec.layout_id}: a pot with more than 3 ingredients"

        rng = np.random.default_rng(0)
        for t in range(400):
            env.step(rng.integers(0, 6, size=(count, 2), dtype=np.int32), aug_prob=0.5)
            if t % 4 == 0:
                env.record_states(stride=1)
        assert env.ring_size >= 256
        env.reset(aug_prob=1.0)  # mostly replayed states now
        chefs = env.obs[:, :, :2].sum((-2, -1)).numpy()
        assert (chefs == 1).all(), f"{spec.layout_id}: every view shows exactly one self and one partner"
        assert ((env.timestep >= 0) & (env.timestep < spec.horizon)).all()
