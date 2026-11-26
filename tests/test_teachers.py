"""The teacher pack and BatchedTeacherPool: no test members, the course's own logits, the manifest's scores."""

from __future__ import annotations

import hashlib
import importlib.util
import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from core.env import read_scenario
from zsc import CONFIG, MAPS
from zsc.engine import VecEnv
from zsc.models import BatchedTeacherPool
from zsc.teachers import OFFICIAL, PACK, load_teacher_members, split_of


def reference_cls():
    spec = importlib.util.spec_from_file_location("partner", OFFICIAL / "fcp/partner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PartnerNetwork


def collect_obs(spec, n_env: int = 48, steps: int = 60, seed: int = 7) -> torch.Tensor:
    env = VecEnv(spec, n_env)
    env.seed(seed)
    env.reset(aug_prob=0.5)
    rng = np.random.default_rng(seed)
    grids = []
    for _ in range(steps):
        grids.append(env.obs.numpy().copy())
        env.step(rng.integers(0, 6, size=(n_env, 2), dtype=np.int32), aug_prob=0.3)
    env.close()
    return torch.from_numpy(np.concatenate(grids, axis=0).reshape(-1, 26, env.height, env.width))


def test_pack_integrity() -> None:
    rows = json.loads((PACK / "manifest.json").read_text())["members"]
    assert {row["split"] for row in rows} == {"train", "dev"}
    for row in rows:
        assert split_of(row["algorithm"], row["seed"]) == row["split"]
        assert hashlib.sha256((PACK / row["file"]).read_bytes()).hexdigest() == row["sha256"], row["name"]
    with pytest.raises(AssertionError):
        load_teacher_members(PACK, "counter_circuit", ("train", "test"), torch.device("cpu"))


@torch.inference_mode()
def test_logit_parity() -> None:
    ref_cls = reference_cls()
    for layout_id in MAPS:
        spec = read_scenario(CONFIG, layout_id=layout_id)
        members = load_teacher_members(PACK, layout_id, ("train", "dev"), torch.device("cpu"))
        pool = BatchedTeacherPool([m.net for m in members], torch.device("cpu"))
        obs = collect_obs(spec)
        g = len(members)
        per = obs.shape[0] // g
        grid = obs[: g * per].reshape(g, per, *obs.shape[1:])
        batched = pool(grid)
        worst = 0.0
        refs = []
        manifest = json.loads((PACK / "manifest.json").read_text())
        rows = {r["name"]: r for r in manifest["members"] if r["layout_id"] == layout_id}
        for i, member in enumerate(members):
            row = rows[member.name]
            ref = ref_cls(**row["network"])
            src = OFFICIAL / member.algorithm
            pop = json.loads((src / "population.json").read_text())
            entry = next(p for p in pop["pools"] if p["scenario_spec_id"] == spec.fingerprint)
            pool_dir = (src / entry["manifest"]).parent
            ref.load_state_dict(load_file(str(pool_dir / f"member-{member.seed}-{member.stage}.safetensors")))
            ref.eval()
            expected = ref(grid[i])
            worst = max(worst, float((batched[i] - expected).abs().max()))
            refs.append(ref)
        assert worst < 1e-5, f"{layout_id}: max |logit diff| {worst:.2e}"
        # Each member must see its own grids: member i's logits differ from member i+1's on the same grids
        cross = min(float((batched[i] - refs[(i + 1) % g](grid[i])).abs().max()) for i in range(g))
        assert cross > 1e-3, f"{layout_id}: members indistinguishable (cross diff {cross:.2e})"


@torch.inference_mode()
def selfplay_scores(layout_id: str, members, episodes: int = 8, seed: int = 11) -> np.ndarray:
    spec = read_scenario(CONFIG, layout_id=layout_id)
    g = len(members)
    n_env = g * episodes
    env = VecEnv(spec, n_env)
    env.seed(seed)
    env.reset()
    pool = BatchedTeacherPool([m.net for m in members], torch.device("cpu"))
    gen = torch.Generator().manual_seed(seed)
    scores = np.zeros(n_env, dtype=np.float64)
    for _ in range(spec.horizon):
        obs = env.obs.clone()  # (n_env, 2, 26, H, W)
        grid = obs.reshape(g, episodes * 2, 26, env.height, env.width)
        probs = pool(grid).softmax(-1).reshape(-1, 6)
        act = torch.multinomial(probs, 1, generator=gen).reshape(n_env, 2).numpy().astype(np.int32)
        sparse, _, _, _ = env.step(act)
        scores += sparse
    env.close()
    return scores.reshape(g, episodes).mean(axis=1)


def test_selfplay_matches_manifest() -> None:
    for layout_id in MAPS:
        members = [m for m in load_teacher_members(PACK, layout_id, ("train", "dev"), torch.device("cpu"))
                   if m.stage in ("best", "final")]
        got = selfplay_scores(layout_id, members)
        want = np.array([m.self_play for m in members])
        gap = float(np.abs(got - want).mean())
        assert gap < 20.0, f"{layout_id}: mean |selfplay - manifest| = {gap:.1f}"

