"""Train one student per map against a pool of partners.

Each episode pairs the student, in a random seat, with one partner slot: an HSP population member, a course
teacher (train split), a rule-based scripted partner, or the student's EMA copy. Pool slots are drawn by regret
within each partner group's fixed share; a slot changes only when its episode ends.

Checkpoints are chosen on the teachers' dev split; test is never loaded. Each map writes best.safetensors,
selection.json, train_state.pt (for resume) and TensorBoard logs. zsc.package turns them into a submission.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from core.catalog import get_map
from core.env import read_scenario
from zsc import CONFIG, ROOT
from zsc.engine import EVENTS, VecEnv
from zsc.models import (
    BatchedPartnerPool,
    BatchedTeacherPool,
    PartnerNet,
    PrivilegedCriticNet,
    autocast,
    default_device,
    extract_privileged_obs_feats,
    gae,
    setup,
    supports_bf16,
)
from zsc.policy import START, ZSCStudentNet
from zsc.scripted import SCRIPTED_NAMES, get_nav, scripted_action
from zsc.teachers import load_teacher_members

COLLISION = EVENTS.index("collision")


class Partner(NamedTuple):
    name: str
    stage: str
    arch: str
    net: nn.Module
    ref: float


@dataclass
class RunningMeanStd:
    """EMA return statistics so the normalized critic target tracks reward scale as shaping decays."""

    decay: float = 0.99
    min_std: float = 1.0
    mean: float = 0.0
    var: float = 1.0
    initialized: bool = False

    @property
    def std(self) -> float:
        return max(self.min_std, float(np.sqrt(max(self.var, 0.0) + 1e-8)))

    def update(self, x: torch.Tensor) -> None:
        mean, var = x.float().mean().item(), x.float().var(unbiased=False).item()
        if self.initialized:
            self.mean = self.decay * self.mean + (1.0 - self.decay) * mean
            self.var = self.decay * self.var + (1.0 - self.decay) * var
        else:
            self.mean, self.var, self.initialized = mean, var, True

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "initialized": self.initialized}

    def load_state_dict(self, state: dict) -> None:
        self.mean, self.var, self.initialized = float(state["mean"]), float(state["var"]), bool(state["initialized"])

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def load_population(partners: Path, layout_id: str, height: int, width: int, device: torch.device,
                    min_partner_sp: float) -> tuple[list[Partner], list[Partner]]:
    """Train pool and final holdout styles; styles whose final self-play < `min_partner_sp` are dropped with all snapshots."""
    rows = [e for e in json.loads((partners / "manifest.json").read_text()) if e["layout_id"] == layout_id]
    final = {e["style"]: e["self_play_score"] for e in rows if e["stage"] == "final"}
    if dropped := sorted(style for style, score in final.items() if score < min_partner_sp):
        print(f"[{layout_id}] Dropped {len(dropped)} population styles with final self-play < {min_partner_sp:g}: {', '.join(dropped)}")
    rows = [e for e in rows if final[e["style"]] >= min_partner_sp]

    def build(e: dict) -> Partner:
        net = PartnerNet(height, width, hidden=128).to(device)
        net.load_state_dict(load_file(str(partners / e["weights"]), device=str(device)))
        return Partner(e["style"], e["stage"], "cnn", net.eval().requires_grad_(False), float(e["self_play_score"]))

    train = [build(e) for e in rows if not e["holdout"]]
    holdout = [build(e) for e in rows if e["holdout"] and e["stage"] == "final"]
    return train, holdout or [p for p in train if p.stage == "final"][:4] or train[:4]


def load_teachers(pack: Path, layout_id: str, device: torch.device) -> tuple[list[Partner], list[Partner]]:
    def to_partner(m) -> Partner:
        return Partner(f"{m.algorithm}-{m.seed}", m.stage, "teacher", m.net, m.self_play)

    train = [to_partner(m) for m in load_teacher_members(pack, layout_id, ("train",), device)]
    dev = [to_partner(m) for m in load_teacher_members(pack, layout_id, ("dev",), device) if m.stage in ("best", "final")]
    return train, dev


def batch_by_arch(pool: list[Partner], height: int, width: int,
                  device: torch.device) -> list[tuple[str, list[int], nn.Module]]:
    out = []
    for arch in ("cnn", "teacher"):
        if ids := [i for i, p in enumerate(pool) if p.arch == arch]:
            nets = [pool[i].net for i in ids]
            module = BatchedPartnerPool(nets, height, width, device) if arch == "cnn" else BatchedTeacherPool(nets, device)
            out.append((arch, ids, module))
    return out


def pool_logits(arch: str, module: nn.Module, obs: torch.Tensor, seat: torch.Tensor, time_left: torch.Tensor,
                device: torch.device, use_amp: bool) -> torch.Tensor:
    with autocast(device, use_amp):
        logits = module(obs, seat, time_left) if arch == "cnn" else module(obs)
    return logits.float()


def _select_score(res: dict) -> float:
    # Penalize the weaker seat because the evaluation platform pairs both seats equally.
    return 0.75 * float(res["mean"]) + 0.25 * min(float(res["seat0"]), float(res["seat1"]))


@torch.inference_mode()
def _eval_rollout(
    student: ZSCStudentNet,
    layout_id: str,
    names: list[str],
    episodes_per_partner: int,
    device: torch.device,
    partner_step: Callable[[torch.Tensor, np.ndarray, float], torch.Tensor],
) -> dict[str, float]:
    spec = read_scenario(CONFIG, layout_id=layout_id)
    n_partners, per_partner = len(names), 2 * episodes_per_partner
    n = n_partners * per_partner
    env = VecEnv(spec, n, pin_memory=device.type == "cuda")
    env.reset()

    student.eval()
    use_amp = supports_bf16(device)
    seats_np = np.tile(np.array([0, 1], dtype=np.int64), n_partners * episodes_per_partner)
    seats = torch.from_numpy(seats_np).to(device)
    seats_f = seats.float()
    hidden = torch.zeros((student.num_layers, n, student.hidden_size), device=device)
    prev_self = prev_partner = torch.full((n, 1), START, dtype=torch.long, device=device)
    scores = np.zeros(n, dtype=np.float32)
    env_idx = torch.arange(n, device=device)
    env_idx_np, partner_seats_np = np.arange(n), 1 - seats_np

    for t in range(spec.horizon):
        obs_cpu = env.obs.numpy()
        obs = env.obs.to(device, non_blocking=True)
        t_left = 1.0 - t / float(spec.horizon)
        with autocast(device, use_amp):
            logits, _, hidden = student(
                obs[env_idx, seats].unsqueeze(1), prev_self, prev_partner, seats_f.unsqueeze(1),
                torch.full((n, 1), t_left, device=device), hidden,
            )
        hidden = hidden.float()
        ego_act = logits[:, 0].float().argmax(dim=-1)
        partner_act = partner_step(obs[env_idx, 1 - seats], obs_cpu[env_idx_np, partner_seats_np], t_left)

        joint = torch.empty((n, 2), dtype=torch.int32, device=device)
        joint[env_idx, seats] = ego_act.to(torch.int32)
        joint[env_idx, 1 - seats] = partner_act.to(torch.int32)
        scores += env.step(joint.cpu().numpy())[0]
        prev_self, prev_partner = ego_act.unsqueeze(1), partner_act.unsqueeze(1)

    env.close()
    per_name = {name: float(scores[i * per_partner:(i + 1) * per_partner].mean()) for i, name in enumerate(names)}
    return per_name | {"mean": float(scores.mean()), "seat0": float(scores[seats_np == 0].mean()),
                       "seat1": float(scores[seats_np == 1].mean())}


def evaluate_against_pool(
    student: ZSCStudentNet,
    layout_id: str,
    pool: list[Partner],
    device: torch.device,
    episodes_per_partner: int = 4,
    partner_greedy_frac: float = 0.5,
    seed: int = 2026,
) -> dict[str, float]:
    grid = get_map(layout_id)["grid"]
    env_h, env_w = len(grid), len(grid[0])
    n_partners, per_partner = len(pool), 2 * episodes_per_partner
    groups = [(arch, torch.tensor(ids, device=device), module)
              for arch, ids, module in batch_by_arch(pool, env_h, env_w, device)]
    use_amp = supports_bf16(device)
    p_seats = torch.tile(torch.tensor([1.0, 0.0], device=device), (n_partners, episodes_per_partner))
    greedy = (torch.arange(per_partner, device=device) < round(per_partner * partner_greedy_frac)).expand(n_partners, -1)
    gen = torch.Generator(device=device).manual_seed(seed)

    def step(p_obs: torch.Tensor, _p_obs_cpu: np.ndarray, t_left: float) -> torch.Tensor:
        p_obs = p_obs.view(n_partners, per_partner, 26, env_h, env_w)
        logits_all = torch.empty((n_partners, per_partner, 6), device=device)
        for arch, idx, module in groups:
            t_in = torch.full((len(idx), per_partner), t_left, device=device)
            logits_all[idx] = pool_logits(arch, module, p_obs[idx], p_seats[idx], t_in, device, use_amp)
        sampled = torch.multinomial(F.softmax(logits_all, dim=-1).reshape(-1, 6), 1, generator=gen)
        return torch.where(greedy, logits_all.argmax(dim=-1), sampled.reshape(n_partners, per_partner)).reshape(-1)

    return _eval_rollout(student, layout_id, [f"{p.name}/{p.stage}" for p in pool], episodes_per_partner, device, step)


def evaluate_against_scripted(
    student: ZSCStudentNet,
    layout_id: str,
    device: torch.device,
    episodes_per_partner: int = 2,
    seed: int = 2026,
) -> dict[str, float]:
    nav = get_nav(layout_id)
    per_partner = 2 * episodes_per_partner
    names = [SCRIPTED_NAMES[i // per_partner] for i in range(len(SCRIPTED_NAMES) * per_partner)]
    rng = np.random.default_rng(seed)

    def step(_p_obs: torch.Tensor, p_obs_cpu: np.ndarray, _t_left: float) -> torch.Tensor:
        acts = np.fromiter((scripted_action(p_obs_cpu[e], nav, names[e], rng) for e in range(len(names))), dtype=np.int64)
        return torch.from_numpy(acts).to(device)

    return _eval_rollout(student, layout_id, list(SCRIPTED_NAMES), episodes_per_partner, device, step)


def _save_weights(module: nn.Module, path: Path) -> None:
    save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}, str(path))


def _cpu_state(module: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def train_map(
    layout_id: str, *, partners: Path, teachers: Path | None, output: Path, num_envs: int, rollout_steps: int,
    chunk_len: int, minibatches: int, steps_per_map: int, hidden_size: int, lr: float, clip_coef: float,
    ent_coef_start: float, ent_coef_end: float, shaping_start: float, collision_penalty: float,
    partner_aux_coef: float, state_aug_prob: float, teacher_share: float, teacher_greedy_prob: float,
    scripted_share: float, min_partner_sp: float, eval_every: int, dev_episodes: int, final_dev_episodes: int,
    num_candidates: int, seed: int, device: str, gamma: float = 0.99, gae_lambda: float = 0.95,
) -> None:
    out = output / layout_id
    out.mkdir(parents=True, exist_ok=True)
    best_path, latest_path, state_path = out / "best.safetensors", out / "latest.safetensors", out / "train_state.pt"
    if state_path.exists() and best_path.exists():
        ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
        if ckpt["done"]:
            print(f"[SKIP] {layout_id} already completed (best={ckpt['best_holdout']:.1f}).")
            return

    writer = SummaryWriter(out / "tb")
    dev, use_amp = setup(seed, device)
    rng = np.random.default_rng(seed)
    spec = read_scenario(CONFIG, layout_id=layout_id)
    env = VecEnv(spec, num_envs, pin_memory=dev.type == "cuda")
    env.seed(seed)
    h, w = env.height, env.width
    nav = get_nav(layout_id)

    pop_pool, holdout_pool = load_population(partners, layout_id, h, w, dev, min_partner_sp)
    teacher_train, dev_pool = load_teachers(teachers, layout_id, dev) if teachers else ([], [])
    net_pool = pop_pool + teacher_train
    n_pop, n_teacher, n_net = len(pop_pool), len(teacher_train), len(net_pool)
    scripted_names = SCRIPTED_NAMES if scripted_share > 0.0 else ()
    n_scripted = len(scripted_names)
    n_pool = n_net + n_scripted
    assert n_pool, f"no training partners for {layout_id}"
    select_on = "dev" if dev_pool else "holdout"

    # Slot layout: population [0, n_pop), teachers [n_pop, n_net), scripted [n_net, n_pool), SP_EMA at n_pool.
    ema_slot = n_pool
    n_slots = n_pool + 1
    slot_names = ([f"{p.arch}:{p.name}/{p.stage}" for p in net_pool]
                  + [f"scripted:{name}" for name in scripted_names]
                  + ["special:SP_EMA"])
    lazy_slot = n_net + scripted_names.index("lazy_blocker") if "lazy_blocker" in scripted_names else -1

    student = ZSCStudentNet(hidden=hidden_size).to(dev)
    critic = PrivilegedCriticNet(hidden_size=hidden_size, n_slots=n_slots).to(dev)
    params = [*student.parameters(), *critic.parameters()]
    opt = torch.optim.Adam(params, lr=lr, eps=1e-5)

    pools = []
    for arch, ids, module in batch_by_arch(net_pool, h, w, dev):
        local = np.full(n_net, -1, dtype=np.int64)
        local[ids] = np.arange(len(ids))
        pools.append((arch, len(ids), module, local))

    if dev.type == "cuda":
        # torch.compile can fail on newer GPU architectures (e.g. sm_120); fall back to eager.
        targets = [(student, "encode_spatial")] + [(module, "forward") for arch, _, module, _ in pools if arch != "teacher"]
        eager = [getattr(obj, name) for obj, name in targets]
        try:
            for (obj, name), fn in zip(targets, eager):
                setattr(obj, name, torch.compile(fn, mode="default"))
            with torch.inference_mode(), autocast(dev, use_amp):
                student.encode_spatial(torch.zeros((2, 26, h, w), device=dev))
                for arch, size, module, _ in pools:
                    if arch != "teacher":
                        pool_logits(arch, module, torch.zeros((size, 2, 26, h, w), device=dev),
                                    torch.zeros((size, 2), device=dev), torch.ones((size, 2), device=dev), dev, use_amp)
        except Exception as exc:
            for (obj, name), fn in zip(targets, eager):
                setattr(obj, name, fn)
            print(f"[WARN] torch.compile skipped ({exc})")

    # Cap lazy_blocker's regret reference near its solo ceiling so it cannot starve productive scripts.
    scripted_ref = {"unident_s_hard": 360.0, "schelling": 240.0}.get(layout_id, 180.0)
    scripted_refs = [min(scripted_ref, 120.0) if name == "lazy_blocker" else scripted_ref for name in scripted_names]
    partner_sp = np.array([max(20.0, p.ref) for p in net_pool] + scripted_refs)
    partner_ema = np.zeros(n_pool)
    partner_var = np.full(n_pool, 400.0)
    ret_rms = RunningMeanStd()

    steps_per_update = num_envs * rollout_steps
    num_updates = max(1, steps_per_map // steps_per_update)
    start_update, best_score = 1, -1.0
    candidates: list[dict] = []
    ema_net = ZSCStudentNet(hidden=hidden_size).to(dev).eval().requires_grad_(False)

    def eval_select(net: ZSCStudentNet, episodes: int) -> tuple[float, dict]:
        if not dev_pool:
            res = evaluate_against_pool(net, layout_id, holdout_pool, dev, episodes_per_partner=episodes)
            return float(res["mean"]), res
        res = evaluate_against_pool(net, layout_id, dev_pool, dev, episodes_per_partner=episodes, partner_greedy_frac=0.0)
        return _select_score(res), res

    def add_candidate(update: int, score: float) -> None:
        if num_candidates <= 0 or not dev_pool:
            return
        if len(candidates) >= num_candidates and score <= min(c["score"] for c in candidates):
            return
        name = f"cand_u{update:05d}.safetensors"
        _save_weights(student, out / name)
        candidates.append({"update": int(update), "score": float(score), "file": name})
        candidates.sort(key=lambda c: -c["score"])
        while len(candidates) > num_candidates:
            (out / candidates.pop()["file"]).unlink(missing_ok=True)

    def save_state(update: int, done: bool = False) -> None:
        state = {
            "student": _cpu_state(student), "critic": _cpu_state(critic), "sp_ema": _cpu_state(ema_net),
            "opt": opt.state_dict(), "ret_rms": ret_rms.state_dict(),
            "partner_ema": partner_ema.tolist(), "partner_var": partner_var.tolist(), "slot_names": slot_names,
            "candidates": candidates, "select_on": select_on, "update": update, "best_holdout": best_score,
            "done": done,
        }
        torch.save(state, state_path.with_suffix(".pt.tmp"))
        os.replace(state_path.with_suffix(".pt.tmp"), state_path)

    if state_path.exists():
        ckpt = torch.load(state_path, map_location=dev, weights_only=True)
        student.load_state_dict(ckpt["student"])
        critic.load_state_dict(ckpt["critic"])
        ema_net.load_state_dict(ckpt["sp_ema"])
        opt.load_state_dict(ckpt["opt"])
        ret_rms.load_state_dict(ckpt["ret_rms"])
        partner_ema, partner_var = np.array(ckpt["partner_ema"]), np.array(ckpt["partner_var"])
        start_update, best_score = ckpt["update"] + 1, ckpt["best_holdout"]
        candidates = [c for c in ckpt["candidates"] if (out / c["file"]).exists()]
        print(f"[RESUME] {layout_id} from upd={start_update}/{num_updates} "
              f"(best={best_score:.1f} on {select_on}, ret_std={ret_rms.std:.1f}).")
    else:
        ema_net.load_state_dict(student.state_dict())
        print(f"[{layout_id}] Loaded {n_pop} population + {n_teacher} teacher train + {n_scripted} scripted partners, "
              f"{len(holdout_pool)} population holdout, {len(dev_pool)} teacher dev "
              f"(envs={num_envs}, hidden={hidden_size}, aug={state_aug_prob:.2f}, bf16={use_amp}, select_on={select_on}).")

    pool_total = 0.88
    groups = [(ids, share) for ids, share in (
        (np.arange(n_pop), None),
        (np.arange(n_pop, n_net), teacher_share),
        (np.arange(n_net, n_pool), scripted_share),
    ) if len(ids)]
    fixed = [float(np.clip(share, 0.0, pool_total)) for _, share in groups if share is not None]
    if len(groups) == 1:
        shares = [(groups[0][0], pool_total)]
    elif n_pop == 0:
        shares = [(ids, pool_total * f / (sum(fixed) or 1.0)) for (ids, _), f in zip(groups, fixed)]
    else:
        rest = max(0.0, pool_total - sum(fixed))
        shares = [(ids, rest if share is None else float(np.clip(share, 0.0, pool_total))) for ids, share in groups]
    print(f"[{layout_id}] partner shares: " + ", ".join(f"{len(ids)} slots -> {share:.2f}" for ids, share in shares))

    def sample_partners(count: int) -> np.ndarray:
        regret = np.maximum(1.0, np.maximum(0.0, partner_sp - partner_ema) + 0.5 * np.sqrt(np.maximum(0.0, partner_var)))
        probs = np.zeros(n_slots)
        for ids, share in shares:
            if share > 0.0:
                r = regret[ids]
                # Keep a 50% uniform floor inside the scripted group so hard blockers do not starve the rest.
                regret_w = 0.50 if ids[0] >= n_net else 0.80
                group = regret_w * (r / r.sum()) + (1.0 - regret_w) / len(ids)
                probs[ids] = share * group / group.sum()
        probs[ema_slot] = 1.0 - pool_total
        return rng.choice(n_slots, size=count, replace=True, p=probs / probs.sum())

    def draw_greedy(pids: np.ndarray) -> np.ndarray:
        is_t = (pids >= n_pop) & (pids < n_net)
        return rng.random(len(pids)) < np.where(is_t, teacher_greedy_prob, 0.50)

    # Seat 1 (left room) has slower onion access on unident_s_hard and converges slower.
    seat1_prob = 0.60 if layout_id == "unident_s_hard" else 0.50
    env_pids = sample_partners(num_envs)
    env_pids_t = torch.from_numpy(env_pids).to(dev)
    seats_np = (rng.random(num_envs) < seat1_prob).astype(np.int64)
    seats_t = torch.from_numpy(seats_np).to(dev)
    greedy_np = draw_greedy(env_pids)
    greedy_t = torch.from_numpy(greedy_np).to(dev)

    # Stagger initial episode lengths so completions spread across updates.
    env.reset(aug_prob=0.5 if state_aug_prob > 0 else 0.0)
    from_aug = env.augmented.numpy().copy()
    clean = ~from_aug
    from_aug_t = torch.from_numpy(from_aug.astype(np.float32)).to(dev)

    def pool_envs(local: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        loc = np.full(num_envs, -1, dtype=np.int64)
        in_net = env_pids < n_net
        loc[in_net] = local[env_pids[in_net]]
        envs = np.nonzero(loc >= 0)[0]
        if len(envs) == 0:
            return None
        envs = envs[np.argsort(loc[envs], kind="stable")]
        rows = loc[envs]
        counts = np.bincount(rows)
        ranks = np.arange(len(rows)) - (np.cumsum(counts) - counts)[rows]
        return torch.from_numpy(envs).to(dev), torch.from_numpy(rows).to(dev), torch.from_numpy(ranks).to(dev), int(counts.max())

    pool_groups = [pool_envs(local) for *_, local in pools]
    scripted_envs = np.nonzero((env_pids >= n_net) & (env_pids < n_pool))[0]
    scripted_envs_t = torch.from_numpy(scripted_envs).to(dev)
    moving_mask = env_pids != lazy_slot

    env_idx = torch.arange(num_envs, device=dev)
    sp_ref = np.full(n_slots, 200.0, dtype=np.float32)
    sp_ref[:n_pool] = partner_sp
    sp_ref_t = torch.from_numpy(sp_ref / 300.0).to(dev)

    layers = student.num_layers
    hidden = torch.zeros((layers, num_envs, hidden_size), device=dev)
    ema_hidden = hidden.clone()
    prev_self = torch.full((num_envs,), START, dtype=torch.long, device=dev)
    prev_partner = prev_self.clone()
    ep_scores = np.zeros(num_envs, dtype=np.float32)

    def buffer(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty((rollout_steps, num_envs, *shape), dtype=dtype, device=dev)

    buf = {
        "obs": buffer(26, h, w), "prev_self": buffer(dtype=torch.long), "prev_partner": buffer(dtype=torch.long),
        "seat": buffer(), "time": buffer(), "act": buffer(dtype=torch.long), "partner_act": buffer(dtype=torch.long),
        "slot": buffer(dtype=torch.long), "priv": buffer(PrivilegedCriticNet.PRIV_DIM), "logp": buffer(),
    }
    val_buf, rew_buf, done_buf = buffer(), buffer(), buffer()

    assert rollout_steps % chunk_len == 0, "rollout_steps must be divisible by chunk_len"
    num_chunks = rollout_steps // chunk_len
    total_seqs = num_chunks * num_envs
    t0 = time.perf_counter()

    def observe() -> tuple[torch.Tensor, ...]:
        obs = env.obs.to(dev, non_blocking=True)
        t_left = 1.0 - env.timestep.to(dev, non_blocking=True).float() / float(spec.horizon)
        ego_obs, p_obs = obs[env_idx, seats_t], obs[env_idx, 1 - seats_t]
        p_seats, ego_seats = (1 - seats_t).float(), seats_t.float()
        meta = torch.stack((
            p_seats, from_aug_t, ((env_pids_t < n_net) & greedy_t).float(), sp_ref_t[env_pids_t], ema_ref_t[env_pids_t],
        ), dim=-1)
        return ego_obs, ego_seats, t_left, p_obs, p_seats, torch.cat((meta, extract_privileged_obs_feats(p_obs)), dim=-1)

    def act_student(ego_obs, ego_seats, t_left, priv):
        with autocast(dev, use_amp):
            logits, _, feat, h_out = student(
                ego_obs.unsqueeze(1), prev_self.unsqueeze(1), prev_partner.unsqueeze(1), ego_seats.unsqueeze(1),
                t_left.unsqueeze(1), hidden, return_features=True,
            )
            value = critic(feat, env_pids_t.unsqueeze(1), priv.unsqueeze(1))
        return logits, h_out, ret_rms.unnormalize(value[:, 0].float())

    for update in range(start_update, num_updates + 1):
        frac = (update - 1) / max(1, num_updates)
        opt.param_groups[0]["lr"] = lr * (1.0 - 0.85 * frac)
        ent_coef = ent_coef_start + (ent_coef_end - ent_coef_start) * frac
        shaping_coef = max(0.0, shaping_start * (1.0 - 1.5 * frac))
        coll_coef = collision_penalty * max(0.0, 1.0 - frac)
        cur_aug_prob = max(0.0, state_aug_prob * (1.0 - frac / 0.80))

        with torch.no_grad():
            for p_ema, p_stu in zip(ema_net.parameters(), student.parameters()):
                p_ema.lerp_(p_stu, 0.02)

        ema_ref = np.full(n_slots, float(partner_ema.mean() / 300.0) if n_pool > 0 else 0.3, dtype=np.float32)
        ema_ref[:n_pool] = partner_ema.astype(np.float32) / 300.0
        ema_ref_t = torch.from_numpy(ema_ref).to(dev)

        chunk_hidden = torch.empty((num_chunks, layers, num_envs, hidden_size), device=dev)

        student.eval()
        critic.eval()
        with torch.inference_mode():
            for t in range(rollout_steps):
                if t % chunk_len == 0:
                    chunk_hidden[t // chunk_len] = hidden

                obs_cpu = env.obs.numpy() if len(scripted_envs) else None
                ego_obs, ego_seats, t_left, p_obs, p_seats, priv = observe()
                logits, hidden, value = act_student(ego_obs, ego_seats, t_left, priv)
                hidden = hidden.float()
                logits = logits[:, 0].float()
                ego_act = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
                ego_logp = F.log_softmax(logits, dim=-1).gather(-1, ego_act.unsqueeze(-1)).squeeze(-1)

                partner_act = torch.empty(num_envs, dtype=torch.long, device=dev)
                for (arch, size, module, _), group in zip(pools, pool_groups):
                    if group is None:
                        continue
                    envs, rows, ranks, width = group
                    pad_obs = torch.zeros((size, width, 26, h, w), device=dev)
                    pad_seat = torch.zeros((size, width), device=dev)
                    pad_time = torch.zeros((size, width), device=dev)
                    pad_obs[rows, ranks], pad_seat[rows, ranks], pad_time[rows, ranks] = p_obs[envs], p_seats[envs], t_left[envs]
                    p_logits = pool_logits(arch, module, pad_obs, pad_seat, pad_time, dev, use_amp)[rows, ranks]
                    sampled = torch.multinomial(F.softmax(p_logits, dim=-1), 1).squeeze(-1)
                    partner_act[envs] = torch.where(greedy_t[envs], p_logits.argmax(dim=-1), sampled)

                if len(scripted_envs):
                    s_acts = np.fromiter(
                        (scripted_action(obs_cpu[e, 1 - seats_np[e]], nav, scripted_names[env_pids[e] - n_net], rng)
                         for e in scripted_envs),
                        dtype=np.int64,
                    )
                    partner_act[scripted_envs_t] = torch.from_numpy(s_acts).to(dev)

                if (ema_idx := torch.nonzero(env_pids_t == ema_slot).squeeze(-1)).numel() > 0:
                    with autocast(dev, use_amp):
                        c_logits, _, h_ema = ema_net(
                            p_obs[ema_idx].unsqueeze(1), prev_partner[ema_idx].unsqueeze(1), prev_self[ema_idx].unsqueeze(1),
                            p_seats[ema_idx].unsqueeze(1), t_left[ema_idx].unsqueeze(1), ema_hidden[:, ema_idx].contiguous(),
                        )
                    ema_hidden[:, ema_idx] = h_ema.float()
                    partner_act[ema_idx] = torch.multinomial(F.softmax(c_logits[:, 0].float(), dim=-1), 1).squeeze(-1)

                step_map = {
                    "obs": ego_obs, "prev_self": prev_self, "prev_partner": prev_partner, "seat": ego_seats,
                    "time": t_left, "act": ego_act, "partner_act": partner_act, "slot": env_pids_t,
                    "priv": priv, "logp": ego_logp,
                }
                for key, x in step_map.items():
                    buf[key][t] = x
                val_buf[t] = value

                joint = torch.empty((num_envs, 2), dtype=torch.int32, device=dev)
                joint[env_idx, seats_t] = ego_act.to(torch.int32)
                joint[env_idx, 1 - seats_t] = partner_act.to(torch.int32)
                sparse_np, shaped_np, done_np, events_np = env.step(joint.cpu().numpy(), aug_prob=cur_aug_prob)
                if cur_aug_prob > 0.0 and t % 10 == 0:
                    env.record_states(10, moving_mask)

                reward = sparse_np + shaping_coef * shaped_np - coll_coef * events_np[np.arange(num_envs), seats_np, COLLISION]
                rew_buf[t] = torch.from_numpy(reward.astype(np.float32)).to(dev, non_blocking=True)
                done_buf[t] = torch.from_numpy(done_np.astype(np.float32)).to(dev, non_blocking=True)
                ep_scores += sparse_np
                prev_self, prev_partner = ego_act, partner_act

                if done_np.any():
                    n_done = int(done_np.sum())
                    done_t = torch.from_numpy(done_np).to(dev)
                    # Only standard-start episodes update regret statistics; mid-game starts have shorter horizons.
                    for e in np.nonzero(done_np)[0]:
                        p = int(env_pids[e])
                        if p < n_pool and clean[e]:
                            diff = float(ep_scores[e]) - partner_ema[p]
                            partner_ema[p] += 0.1 * diff
                            partner_var[p] = 0.9 * partner_var[p] + 0.1 * (diff * diff)
                    ep_scores[done_np] = 0.0

                    hidden[:, done_t] = 0.0
                    ema_hidden[:, done_t] = 0.0
                    prev_self[done_t] = START
                    prev_partner[done_t] = START

                    env_pids[done_np] = sample_partners(n_done)
                    env_pids_t = torch.from_numpy(env_pids).to(dev)
                    seats_np[done_np] = (rng.random(n_done) < seat1_prob).astype(np.int64)
                    seats_t = torch.from_numpy(seats_np).to(dev)
                    greedy_np[done_np] = draw_greedy(env_pids[done_np])
                    greedy_t = torch.from_numpy(greedy_np).to(dev)

                    from_aug[done_np] = env.augmented.numpy()[done_np]
                    clean[done_np] = ~from_aug[done_np]
                    from_aug_t = torch.from_numpy(from_aug.astype(np.float32)).to(dev)
                    moving_mask = env_pids != lazy_slot
                    pool_groups = [pool_envs(local) for *_, local in pools]
                    scripted_envs = np.nonzero((env_pids >= n_net) & (env_pids < n_pool))[0]
                    scripted_envs_t = torch.from_numpy(scripted_envs).to(dev)

            ego_obs, ego_seats, t_left, _, _, priv = observe()
            next_value = act_student(ego_obs, ego_seats, t_left, priv)[-1]

        adv = gae(rew_buf, done_buf, val_buf, next_value, gamma, gae_lambda)
        ret = adv + val_buf
        ret_rms.update(ret)

        def seq(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(num_chunks, chunk_len, num_envs, *x.shape[2:]).transpose(1, 2).reshape(total_seqs, chunk_len, *x.shape[2:])

        s = {key: seq(x) for key, x in buf.items()}
        s_adv, s_ret = seq(adv), seq(ret_rms.normalize(ret))
        s_h0 = chunk_hidden.permute(0, 2, 1, 3).reshape(total_seqs, layers, hidden_size)
        s_adv = (s_adv - s_adv.mean()) / (s_adv.std() + 1e-8)

        student.train()
        critic.train()
        mb_seqs = max(1, total_seqs // minibatches)
        for _ in range(4):
            perm = torch.randperm(total_seqs, device=dev)
            for start in range(0, total_seqs, mb_seqs):
                i = perm[start:start + mb_seqs]
                with autocast(dev, use_amp):
                    logits, p_logits, feat, _ = student(
                        s["obs"][i], s["prev_self"][i], s["prev_partner"][i], s["seat"][i], s["time"][i],
                        s_h0[i].transpose(0, 1).contiguous(), return_features=True,
                    )
                    value = critic(feat, s["slot"][i], s["priv"][i])
                logits, p_logits, value = logits.float(), p_logits.float(), value.float()

                log_probs = F.log_softmax(logits, dim=-1)
                ratio = (log_probs.gather(-1, s["act"][i].unsqueeze(-1)).squeeze(-1) - s["logp"][i]).exp()
                pg_loss = -torch.min(ratio * s_adv[i], ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * s_adv[i]).mean()
                v_loss = F.smooth_l1_loss(value, s_ret[i], beta=2.0)
                ent = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
                aux_loss = F.cross_entropy(p_logits.reshape(-1, 6), s["partner_act"][i].reshape(-1), label_smoothing=0.03)
                loss = pg_loss + 0.5 * v_loss - ent_coef * ent + partner_aux_coef * aux_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()

        cur_eval_every = max(1, eval_every // 2) if update >= int(0.70 * num_updates) else eval_every
        if update % cur_eval_every == 0 or update == num_updates:
            _save_weights(student, latest_path)
            holdout = (evaluate_against_pool(student, layout_id, holdout_pool, dev, episodes_per_partner=4)
                       if holdout_pool else {"mean": float("nan")})
            scripted_res = evaluate_against_scripted(student, layout_id, dev, episodes_per_partner=2)
            select, dev_res = eval_select(student, dev_episodes) if dev_pool else (float(holdout["mean"]), None)
            if select >= best_score:
                best_score = select
                _save_weights(student, best_path)
            add_candidate(update, select)
            save_state(update)

            step = update * steps_per_update
            sps = (update - start_update + 1) * steps_per_update / max(1e-6, time.perf_counter() - t0)
            nan = float("nan")
            pop_ema = float(partner_ema[:n_pop].mean()) if n_pop else nan
            teacher_ema = float(partner_ema[n_pop:n_net].mean()) if n_teacher else nan
            scripted_ema = float(partner_ema[n_net:n_pool].mean()) if n_scripted else nan
            scalars = {
                "train/sps": sps, "train/lr": opt.param_groups[0]["lr"], "train/ent_coef": ent_coef,
                "train/aug_prob": cur_aug_prob, "train/ret_std": ret_rms.std,
                "partner_ema/all": partner_ema.mean(), "partner_ema/population": pop_ema,
                "partner_ema/teacher": teacher_ema, "partner_ema/scripted": scripted_ema,
                "holdout/mean": holdout["mean"], "scripted/mean": scripted_res["mean"],
                "select/score": select, "select/best": best_score,
                "loss/policy": pg_loss.item(), "loss/value": v_loss.item(), "loss/partner_aux": aux_loss.item(),
                "loss/entropy": ent.item(),
            }
            if dev_res is not None:
                scalars |= {f"dev/{k}": dev_res[k] for k in ("mean", "seat0", "seat1")}
            for tag, x in scalars.items():
                writer.add_scalar(tag, float(x), step)
            dev_msg = f"dev={dev_res['mean']:.1f} (s0={dev_res['seat0']:.1f}, s1={dev_res['seat1']:.1f}) " if dev_res else ""
            print(f"[{layout_id}] upd={update}/{num_updates} steps={step:,} aug={cur_aug_prob:.2f} ring={env.ring_size} "
                  f"ret_std={ret_rms.std:.1f} train_ema={partner_ema.mean():.1f} (pop={pop_ema:.1f}, "
                  f"teacher={teacher_ema:.1f}, scr={scripted_ema:.1f}) {dev_msg}scripted={scripted_res['mean']:.1f} "
                  f"holdout={holdout['mean']:.1f} best={best_score:.1f}[{select_on}] "
                  f"ent={ent.item():.2f} aux={aux_loss.item():.2f} ({sps:,.0f} sps)", flush=True)

    selection: dict = {"select_on": select_on}
    if dev_pool and candidates:
        rescored = []
        for cand in candidates:
            student.load_state_dict(load_file(str(out / cand["file"]), device=str(dev)))
            score, res = eval_select(student, final_dev_episodes)
            rescored.append({**cand, "rescore": score, "dev": res["mean"], "seat0": res["seat0"], "seat1": res["seat1"]})
            print(f"[{layout_id}] candidate upd={cand['update']}: dev select {cand['score']:.1f} -> {score:.1f} "
                  f"(mean={res['mean']:.1f}, seat0={res['seat0']:.1f}, seat1={res['seat1']:.1f})", flush=True)
        chosen = max(rescored, key=lambda c: c["rescore"])
        shutil.copy2(out / chosen["file"], best_path)
        best_score = float(chosen["rescore"])
        selection |= {"candidates": rescored, "chosen_update": int(chosen["update"])}

    student.load_state_dict(load_file(str(best_path), device=str(dev)))
    scr_final = evaluate_against_scripted(student, layout_id, dev, episodes_per_partner=4)
    print(f"[{layout_id}] Best {select_on} -> score={best_score:.1f} scripted={scr_final['mean']:.1f}", flush=True)
    selection |= {"best_score": best_score, "scripted": scr_final}
    (out / "selection.json").write_text(json.dumps(selection, indent=1) + "\n", encoding="utf-8")
    save_state(num_updates, done=True)
    env.close()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", nargs="+", default=["counter_circuit", "schelling", "unident_s_hard"])
    parser.add_argument("--partners", type=Path, default=ROOT / "runs/population")
    parser.add_argument("--teachers", type=Path, help="teacher pack (train and dev members)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--chunk-len", type=int, default=40)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--steps-per-map", type=int, default=None, help="override steps for all maps (default: 120M/140M/180M)")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef-start", type=float, default=0.02)
    parser.add_argument("--ent-coef-end", type=float, default=0.003)
    parser.add_argument("--shaping-start", type=float, default=0.4, help="shaped-reward weight, decays to 0 at 2/3 of training")
    parser.add_argument("--collision-penalty", type=float, default=0.05)
    parser.add_argument("--partner-aux-coef", type=float, default=0.25)
    parser.add_argument("--state-aug-prob", type=float, default=0.35, help="initial augmented restart prob, decays to 0 at 80%")
    parser.add_argument("--teacher-share", type=float, default=0.46, help="share of the 0.88 pool draws for teachers")
    parser.add_argument("--teacher-greedy-prob", type=float, default=0.15)
    parser.add_argument("--scripted-share", type=float, default=0.12, help="share of the 0.88 pool draws for scripted partners")
    parser.add_argument("--min-partner-sp", type=float, default=20.0, help="drop population styles below this self-play")
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--dev-episodes", type=int, default=8, help="per dev member and seat at each evaluation")
    parser.add_argument("--final-dev-episodes", type=int, default=24, help="per dev member and seat for the final choice")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    args = vars(parser.parse_args())
    default_steps = {"counter_circuit": 120_000_000, "unident_s_hard": 140_000_000, "schelling": 180_000_000}
    override_steps = args.pop("steps_per_map")
    for layout_id in args.pop("layouts"):
        train_map(layout_id, steps_per_map=override_steps or default_steps.get(layout_id, 120_000_000), **args)


if __name__ == "__main__":
    main()
