"""Train one student per map against a pool of partners, or fine-tune a trained one (--init-from).

Each episode pairs the student, in a random seat, with one partner slot: a population member, a course teacher
(train split), a simulated classmate (train runs), a copy of the student (an EMA and two snapshots), the EMA
with noised actions, or a chef that mostly stays put. Pool slots are drawn by regret, the gap between what a
partner reaches in self-play and what the student reaches with it; a slot changes only when its episode ends.

Checkpoints, the belief eta and the sampling temperature are chosen on the teachers' and classmates' dev splits;
test is never loaded. Each map writes best.safetensors, selection.json, train_state.pt (resume, and the critic's
warm start when fine-tuning) and TensorBoard logs. zsc.package turns them into a submission.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from core.env import read_scenario
from zsc import CONFIG, ROOT
from zsc.engine import EVENTS, VecEnv
from zsc.models import (
    ROT180_ACT_NP,
    ROT180_CH_PERM,
    BatchedMLPPool,
    BatchedPartnerPool,
    BatchedTeacherPool,
    MLPPartnerNet,
    PartnerNet,
    PrivilegedCriticNet,
    TeacherNet,
    autocast,
    extract_privileged_obs_feats,
    gae,
    rot180_obs_gpu,
    setup,
)
from zsc.policy import ZSCStudentNet
from zsc.teachers import load_teacher_members

COLLISION = EVENTS.index("collision")
SPECIAL_SLOTS = ("SP_EMA", "SP_HIST1", "SP_HIST2", "UPD_MIX", "STICKY_STAY")

# Partner types, the targets of the student's partner head: 0 balanced, 1 clockwise loop, 2 counter-clockwise
# loop, 3 onion supplier, 4 cook and plater, 5 upper-left zone, 6 lower-right zone, 7 shared pot, 8 cautious,
# 9 student copies, 10 noised student, 11 novice (early snapshots, untrained teachers, the staying chef).
# A population style's type depends on the seat it plays; styles not listed count as balanced.
STYLE_TYPES = {
    "cw_loop": (1, 1), "ccw_loop": (2, 2), "counter_relay_ab": (3, 4), "onion_vs_plater": (3, 4),
    "counter_relay_ba": (4, 3), "plater_vs_onion": (4, 3), "zone_split_tb": (5, 6), "zone_split_bt": (6, 5),
    "shared_pot_top": (7, 7), "shared_pot_bot": (7, 7), "cautious_sparse": (8, 8),
}
CONVENTIONS = list(range(9))
COPY, NOISED, NOVICE = 9, 10, 11
ZONE_SWAP = [0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11]  # the 180-degree flip swaps the two zone types

DEV_TAGS = {"mean": "dev/mean", "seat0": "dev/seat0", "seat1": "dev/seat1", "cm_mean": "classmate_dev/mean",
            "cm_seat0": "classmate_dev/seat0", "cm_seat1": "classmate_dev/seat1",
            "teacher_select": "select/teacher", "cm_select": "select/classmate"}


class Partner(NamedTuple):
    name: str
    stage: str
    arch: str  # cnn, mlp or teacher
    net: nn.Module
    ref: float  # the regret reference: what the partner reaches in self-play


class RunningMeanStd:
    """Exponential moving return statistics: the value scale has to follow the reward as shaping decays."""

    def __init__(self, decay: float = 0.99, min_std: float = 1.0):
        self.decay, self.min_std = decay, min_std
        self.mean, self.var, self.initialized = 0.0, 1.0, False

    @property
    def std(self) -> float:
        return max(self.min_std, float(np.sqrt(max(self.var, 0.0) + 1e-8)))

    def update(self, x: torch.Tensor) -> None:
        x = x.float()
        mean, var = x.mean().item(), x.var(unbiased=False).item()
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
    """The population's train pool and its final holdout styles.

    A style whose final self-play is below `min_partner_sp` never learned to deliver and is dropped with all
    its snapshots; the early snapshots of competent styles stay, as novice partners.
    """
    rows = [e for e in json.loads((partners / "manifest.json").read_text()) if e["layout_id"] == layout_id]
    final = {(e["style"], e["arch"]): e["self_play_score"] for e in rows if e["stage"] == "final"}
    dropped = sorted({f"{arch}:{style}" for (style, arch), score in final.items() if score < min_partner_sp})
    if dropped:
        print(f"[{layout_id}] Dropped {len(dropped)} population styles with final self-play < {min_partner_sp:g}: {', '.join(dropped)}")
    rows = [e for e in rows if final[e["style"], e["arch"]] >= min_partner_sp]

    def build(e: dict) -> Partner:
        net = (MLPPartnerNet if e["arch"] == "mlp" else PartnerNet)(height, width, hidden=128).to(device)
        net.load_state_dict(load_file(str(partners / e["weights"]), device=str(device)))
        return Partner(e["style"], e["stage"], e["arch"], net.eval().requires_grad_(False), float(e["self_play_score"]))

    train = [build(e) for e in rows if not e["holdout"]]
    holdout = [build(e) for e in rows if e["holdout"] and e["stage"] == "final"]
    return train, holdout or [p for p in train if p.stage == "final"][:4] or train[:4]


def load_teachers(pack: Path, layout_id: str, device: torch.device) -> tuple[list[Partner], list[Partner]]:
    """Course teachers: every train member (init/mid/best/final) and the mature dev members (best/final)."""
    train = load_teacher_members(pack, layout_id, ("train",), device)
    dev = load_teacher_members(pack, layout_id, ("dev",), device)

    def item(m) -> Partner:
        return Partner(f"{m.algorithm}-{m.seed}", m.stage, "teacher", m.net, m.self_play)

    return [item(m) for m in train], [item(m) for m in dev if m.stage in ("best", "final")]


def load_classmates(classmates: Path, layout_id: str, height: int, width: int,
                    device: torch.device) -> tuple[list[Partner], list[Partner]]:
    """Classmate snapshots of the train runs and of the dev runs; test runs are never loaded.

    The regret reference is max(self-play, score with teachers): a classmate trained with the teacher packs is
    measured against what it reaches with the partners it was trained for.
    """
    rows = [e for e in json.loads((classmates / "manifest.json").read_text()) if e["layout_id"] == layout_id]

    def build(e: dict) -> Partner:
        net = MLPPartnerNet(height, width, hidden=128) if e["arch"] == "mlp" else TeacherNet(height, width)
        net.load_state_dict(load_file(str(classmates / e["weights"]), device=str(device)))
        return Partner(f"cm:{e['style']}", e["stage"], e["arch"], net.to(device).eval().requires_grad_(False),
                       float(e["ref_score"]))

    return [build(e) for e in rows if e["split"] == "train"], [build(e) for e in rows if e["split"] == "dev"]


def batch_by_arch(pool: list[Partner], height: int, width: int,
                  device: torch.device) -> list[tuple[str, list[int], nn.Module]]:
    """One batched module per architecture: (arch, pool indices, module)."""
    out = []
    for arch in ("cnn", "mlp", "teacher"):
        ids = [i for i, p in enumerate(pool) if p.arch == arch]
        if ids:
            nets = [pool[i].net for i in ids]
            module = (BatchedPartnerPool(nets, height, width, device) if arch == "cnn"
                      else BatchedMLPPool(nets, height, width, device) if arch == "mlp"
                      else BatchedTeacherPool(nets, device))
            out.append((arch, ids, module))
    return out


def pool_logits(arch: str, module: nn.Module, obs: torch.Tensor, seat: torch.Tensor, time_left: torch.Tensor,
                use_amp: bool) -> torch.Tensor:
    """Logits (P, M, 6) of a batched pool for grids (P, M, 26, H, W); only population CNNs read seat and time.
    Teachers always run in fp32."""
    with autocast(use_amp):
        logits = module(obs, seat, time_left) if arch == "cnn" else module(obs)
    return logits.float()


def _select_score(res: dict) -> float:
    """Mostly the mean, with a penalty for a weak seat: the platform plays both seats."""
    return 0.75 * float(res["mean"]) + 0.25 * min(float(res["seat0"]), float(res["seat1"]))


@torch.inference_mode()
def evaluate_against_pool(
    student: ZSCStudentNet,
    layout_id: str,
    pool: list[Partner],
    device: torch.device,
    episodes_per_partner: int = 4,
    temperature: float = 0.0,
    belief_eta: float | None = None,
    partner_greedy_frac: float = 0.5,
    return_bucket_ll: bool = False,
    seed: int = 2026,
):
    """Scores from standard starts with every partner in `pool`, `episodes_per_partner` per seat.

    A `partner_greedy_frac` share of each partner's episodes plays its argmax action, the rest sample; use 0
    for teachers, which sample in the course's FrozenPartner. Returns per-partner scores, `holdout_<arch>`,
    `mean`, `seat0` and `seat1` (the student's seat); with `return_bucket_ll` also the mean log-likelihood of
    each partner's actions under each partner type, shaped (partners, partner seat, types).
    """
    spec = read_scenario(CONFIG, layout_id=layout_id)
    n_partners, per_partner = len(pool), 2 * episodes_per_partner
    n = n_partners * per_partner
    env = VecEnv(spec, n, pin_memory=device.type == "cuda")
    env.reset()
    groups = [(arch, torch.tensor(ids, device=device), module)
              for arch, ids, module in batch_by_arch(pool, env.height, env.width, device)]

    eta = student.belief_eta
    if belief_eta is not None:
        student.belief_eta = float(belief_eta)
    student.eval()
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    seats_np = np.tile(np.array([0, 1], dtype=np.int64), n_partners * episodes_per_partner)
    seats = torch.from_numpy(seats_np).to(device)
    seats_f = seats.float()
    p_seats = (1.0 - seats_f).view(n_partners, per_partner)
    greedy = (torch.arange(per_partner, device=device) < round(per_partner * partner_greedy_frac)).expand(n_partners, -1)

    k = student.num_archetypes
    hidden = torch.zeros((student.num_layers, n, student.hidden_size), device=device)
    belief = torch.full((n, k), 1.0 / k, device=device)
    prev_plogits = None
    prev_self = prev_partner = torch.full((n, 1), 6, dtype=torch.long, device=device)
    scores = np.zeros(n, dtype=np.float32)
    ll_sum = torch.zeros((n, k), device=device) if return_bucket_ll else None
    env_idx = torch.arange(n, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    for t in range(spec.horizon):
        obs = env.obs.to(device, non_blocking=True)
        t_left = 1.0 - t / float(spec.horizon)
        with autocast(use_amp):
            logits, p_logits, hidden, belief = student(
                obs[env_idx, seats].unsqueeze(1), prev_self, prev_partner, seats_f.unsqueeze(1),
                torch.full((n, 1), t_left, device=device), hidden, belief, prev_plogits,
            )
        hidden = hidden.float()
        prev_plogits = p_logits[:, -1].float()
        logits = logits[:, 0].float()
        if temperature <= 0.0:
            ego_act = logits.argmax(dim=-1)
        else:
            ego_act = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1, generator=gen).squeeze(-1)

        p_obs = obs[env_idx, 1 - seats].view(n_partners, per_partner, 26, env.height, env.width)
        p_logits_all = torch.empty((n_partners, per_partner, 6), device=device)
        for arch, idx, module in groups:
            time_left = torch.full((len(idx), per_partner), t_left, device=device)
            p_logits_all[idx] = pool_logits(arch, module, p_obs[idx], p_seats[idx], time_left, use_amp)
        sampled = torch.multinomial(F.softmax(p_logits_all, dim=-1).reshape(-1, 6), 1, generator=gen)
        partner_act = torch.where(greedy, p_logits_all.argmax(dim=-1), sampled.reshape(n_partners, per_partner)).reshape(n)
        if ll_sum is not None:
            log_p = F.log_softmax(prev_plogits, dim=-1)  # (E, K, 6), the prediction of this step's action
            ll_sum += log_p.gather(-1, partner_act.view(-1, 1, 1).expand(-1, k, 1)).squeeze(-1)

        joint = torch.empty((n, 2), dtype=torch.int32, device=device)
        joint[env_idx, seats] = ego_act.to(torch.int32)
        joint[env_idx, 1 - seats] = partner_act.to(torch.int32)
        scores += env.step(joint.cpu().numpy())[0]
        prev_self, prev_partner = ego_act.unsqueeze(1), partner_act.unsqueeze(1)

    student.belief_eta = eta
    env.close()
    res: dict[str, float] = {}
    by_arch: dict[str, list[float]] = {"cnn": [], "mlp": [], "teacher": []}
    for i, p in enumerate(pool):
        res[f"{p.name}/{p.stage}"] = val = float(scores[i * per_partner:(i + 1) * per_partner].mean())
        by_arch[p.arch].append(val)
    for arch, vals in by_arch.items():
        res[f"holdout_{arch}"] = float(np.mean(vals)) if vals else float("nan")
    res |= {"mean": float(scores.mean()), "seat0": float(scores[seats_np == 0].mean()),
            "seat1": float(scores[seats_np == 1].mean())}
    if ll_sum is None:
        return res
    ll = (ll_sum / float(spec.horizon)).view(n_partners, per_partner, k).cpu().numpy()
    partner_seat = (1 - seats_np).reshape(n_partners, per_partner)
    return res, np.stack([np.stack([ll[i][partner_seat[i] == s].mean(axis=0) for s in (0, 1)]) for i in range(n_partners)])


def _save_weights(module: nn.Module, path: Path) -> None:
    save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}, str(path))


def _cpu_state(module: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _warm_start_critic(critic: PrivilegedCriticNet, old_state: dict, old_names: list[str], new_names: list[str]) -> int:
    """Load a previous critic, matching slot embeddings by slot name; new slots start at the mean embedding."""
    new_sd = critic.state_dict()
    for k, v in old_state.items():
        if k != "slot_emb.weight" and k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v.to(new_sd[k].dtype)
    old_emb = old_state["slot_emb.weight"]
    index = {n: i for i, n in enumerate(old_names)}
    mean_row = old_emb[[i for i, n in enumerate(old_names) if not n.startswith("special:")]].mean(dim=0)
    emb = new_sd["slot_emb.weight"].clone()
    for j, name in enumerate(new_names):
        emb[j] = old_emb[index[name]] if name in index else mean_row
    new_sd["slot_emb.weight"] = emb
    critic.load_state_dict(new_sd)
    return sum(name in index for name in new_names)


def train_map(
    layout_id: str, *, partners: Path, teachers: Path | None, classmates: Path | None, output: Path,
    init_from: Path | None, init_which: str, num_envs: int, rollout_steps: int, chunk_len: int, minibatches: int,
    steps_per_map: int, hidden_size: int, lr: float, clip_coef: float, ent_coef_start: float, ent_coef_end: float,
    shaping_start: float, collision_penalty: float, partner_aux_coef: float, state_aug_prob: float,
    critic_warmup_updates: int, teacher_share: float, teacher_greedy_prob: float, teacher_upd_prob: float,
    classmate_share: float, classmate_weight: float, min_partner_sp: float, eval_every: int, dev_episodes: int,
    final_dev_episodes: int, num_candidates: int, seed: int, device: str, gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> None:
    """Train or fine-tune one map, resuming from train_state.pt when the output already has one.

    Without teachers, checkpoints are chosen on the population's holdout styles.
    """
    out = output / layout_id
    out.mkdir(parents=True, exist_ok=True)
    best_path, latest_path, state_path = out / "best.safetensors", out / "latest.safetensors", out / "train_state.pt"
    if state_path.exists() and best_path.exists():
        ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
        if ckpt["done"]:
            print(f"[SKIP] {layout_id} already completed (best={ckpt['best_holdout']:.1f}, "
                  f"eta={ckpt['best_eta']:.2f}, temp={ckpt['best_temp']:.2f}).")
            return

    writer = SummaryWriter(out / "tb")

    def log_dev(res: dict, step: int) -> None:
        for key, tag in DEV_TAGS.items():
            if key in res:
                writer.add_scalar(tag, float(res[key]), step)

    dev, use_amp = setup(seed, device)
    rng = np.random.default_rng(seed)
    other_play = layout_id == "schelling"
    spec = read_scenario(CONFIG, layout_id=layout_id)
    env = VecEnv(spec, num_envs, pin_memory=dev.type == "cuda")
    env.seed(seed)
    H, W = env.height, env.width
    pop_pool, holdout_pool = load_population(partners, layout_id, H, W, dev, min_partner_sp)
    teacher_train, dev_pool = load_teachers(teachers, layout_id, dev) if teachers else ([], [])
    cm_train, cm_dev_pool = load_classmates(classmates, layout_id, H, W, dev) if classmates else ([], [])
    train_pool = pop_pool + teacher_train + cm_train
    n_pop, n_teacher, n_cm, n_pool = len(pop_pool), len(teacher_train), len(cm_train), len(train_pool)
    cm_lo = n_pop + n_teacher
    assert n_pool, f"no training partners for {layout_id}"
    select_on = "dev" if dev_pool else "holdout"

    # Slots: population [0, n_pop), teachers [n_pop, cm_lo), classmates [cm_lo, n_pool), then the special slots.
    EMA, HIST1, HIST2, NOISED_EMA, STAY = range(n_pool, n_pool + len(SPECIAL_SLOTS))
    n_slots = n_pool + len(SPECIAL_SLOTS)
    slot_names = [f"{p.arch}:{p.name}/{p.stage}" for p in train_pool] + [f"special:{n}" for n in SPECIAL_SLOTS]

    student = ZSCStudentNet(hidden=hidden_size).to(dev)
    critic = PrivilegedCriticNet(hidden_size=hidden_size, n_slots=n_slots).to(dev)
    params = [*student.parameters(), *critic.parameters()]
    opt = torch.optim.Adam(params, lr=lr, eps=1e-5)
    k = student.num_archetypes

    # Pool partners run as one batched module per architecture; `local` takes a slot to its row in the module.
    pools = []
    for arch, ids, module in batch_by_arch(train_pool, H, W, dev):
        local = np.full(n_pool, -1, dtype=np.int64)
        local[ids] = np.arange(len(ids))
        pools.append((arch, len(ids), module, local))

    if dev.type == "cuda":
        # torch.compile has failed on new GPUs (sm_120): compile and run once, falling back to eager on any error.
        targets = [(student, "encode_spatial")] + [(module, "forward") for arch, _, module, _ in pools if arch != "teacher"]
        eager = [getattr(obj, name) for obj, name in targets]
        try:
            for (obj, name), fn in zip(targets, eager):
                setattr(obj, name, torch.compile(fn, mode="default"))
            with torch.inference_mode(), autocast(use_amp):
                student.encode_spatial(torch.zeros((2, 26, H, W), device=dev))
                for arch, size, module, _ in pools:
                    if arch != "teacher":
                        pool_logits(arch, module, torch.zeros((size, 2, 26, H, W), device=dev),
                                    torch.zeros((size, 2), device=dev), torch.ones((size, 2), device=dev), use_amp)
        except Exception as exc:
            for (obj, name), fn in zip(targets, eager):
                setattr(obj, name, fn)
            print(f"[WARN] torch.compile skipped ({exc})")

    # Partner type of every slot, by the seat the partner plays.
    slot_types = np.zeros((n_slots, 2), dtype=np.int64)
    for i, p in enumerate(train_pool):
        if p.arch == "teacher":  # replaced from the partner head's predictions when fine-tuning
            slot_types[i] = NOVICE if p.stage == "init" else 0
        else:
            slot_types[i] = NOVICE if p.stage == "early" else STYLE_TYPES.get(p.name, (0, 0))
    slot_types[n_pool:] = np.array([COPY, COPY, COPY, NOISED, NOVICE])[:, None]

    partner_sp = np.array([max(20.0, p.ref) for p in train_pool])
    partner_ema = np.zeros(n_pool)
    partner_var = np.full(n_pool, 400.0)
    ret_rms = RunningMeanStd()

    steps_per_update = num_envs * rollout_steps
    num_updates = max(1, steps_per_map // steps_per_update)
    start_update, best_score = 1, -1.0
    hist1_latched = hist2_latched = warm_started = False
    candidates: list[dict] = []
    init_info: dict = {}

    # Student copies as partners: an EMA of the weights and snapshots latched at 25% and 60% of training.
    copies = [ZSCStudentNet(hidden=hidden_size).to(dev).eval().requires_grad_(False) for _ in range(3)]
    ema_net, hist1_net, hist2_net = copies

    def eval_select(net: ZSCStudentNet, episodes: int, temperature: float = 0.0, eta: float = 1.0) -> tuple[float, dict]:
        """Selection score of `net`: teacher dev blended with classmate dev, or the holdout mean without teachers."""
        if not dev_pool:
            res = evaluate_against_pool(net, layout_id, holdout_pool, dev, episodes_per_partner=episodes,
                                        temperature=temperature, belief_eta=eta)
            return float(res["mean"]), res
        res = evaluate_against_pool(net, layout_id, dev_pool, dev, episodes_per_partner=episodes,
                                    temperature=temperature, belief_eta=eta, partner_greedy_frac=0.0)
        score = _select_score(res)
        if cm_dev_pool:
            cres = evaluate_against_pool(net, layout_id, cm_dev_pool, dev, episodes_per_partner=max(2, episodes // 2),
                                         temperature=temperature, belief_eta=eta, partner_greedy_frac=0.0)
            res = {**res, "teacher_select": score, "cm_mean": cres["mean"], "cm_seat0": cres["seat0"],
                   "cm_seat1": cres["seat1"], "cm_select": _select_score(cres)}
            score = (1.0 - classmate_weight) * score + classmate_weight * res["cm_select"]
        return score, res

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

    def save_state(update: int, done: bool = False, best_temp: float = 0.0, best_eta: float = 1.0) -> None:
        state = {
            "student": _cpu_state(student), "critic": _cpu_state(critic), "sp_ema": _cpu_state(ema_net),
            "sp_hist1": _cpu_state(hist1_net), "sp_hist2": _cpu_state(hist2_net), "hist1_latched": hist1_latched,
            "hist2_latched": hist2_latched, "opt": opt.state_dict(), "ret_rms": ret_rms.state_dict(),
            "partner_ema": partner_ema.tolist(), "partner_var": partner_var.tolist(), "slot_names": slot_names,
            "slot_seat_buckets": slot_types.tolist(), "candidates": candidates, "warm_started": warm_started,
            "init_info": init_info, "select_on": select_on, "update": update, "best_holdout": best_score,
            "best_temp": best_temp, "best_eta": best_eta, "done": done,
        }
        torch.save(state, state_path.with_suffix(".pt.tmp"))
        os.replace(state_path.with_suffix(".pt.tmp"), state_path)

    if state_path.exists():
        ckpt = torch.load(state_path, map_location=dev, weights_only=True)
        student.load_state_dict(ckpt["student"])
        critic.load_state_dict(ckpt["critic"])
        opt.load_state_dict(ckpt["opt"])
        ret_rms.load_state_dict(ckpt["ret_rms"])
        for net, key in zip(copies, ("sp_ema", "sp_hist1", "sp_hist2")):
            net.load_state_dict(ckpt[key])
        partner_ema, partner_var = np.array(ckpt["partner_ema"]), np.array(ckpt["partner_var"])
        slot_types = np.array(ckpt["slot_seat_buckets"], dtype=np.int64)
        start_update, best_score = ckpt["update"] + 1, ckpt["best_holdout"]
        hist1_latched, hist2_latched, warm_started = ckpt["hist1_latched"], ckpt["hist2_latched"], ckpt["warm_started"]
        candidates = [c for c in ckpt["candidates"] if (out / c["file"]).exists()]
        init_info = ckpt["init_info"]
        print(f"[RESUME] {layout_id} from upd={start_update}/{num_updates} "
              f"(best={best_score:.1f} on {select_on}, ret_std={ret_rms.std:.1f}).")
    else:
        if init_from is not None:
            src = init_from / layout_id
            which = ("latest", "best") if init_which == "auto" else (init_which,)
            scored = []
            for w in which:
                path = src / f"{w}.safetensors"
                if init_which == "auto" and not path.exists():
                    continue
                student.load_state_dict(load_file(str(path), device=str(dev)))
                score, res = eval_select(student, dev_episodes)
                scored.append((score, w, path, res))
                print(f"[INIT] {layout_id} {w}: {select_on}={score:.1f} "
                      f"(mean={res['mean']:.1f}, seat0={res['seat0']:.1f}, seat1={res['seat1']:.1f})")
            init_score, init_w, init_path, init_res = max(scored, key=lambda x: x[0])
            student.load_state_dict(load_file(str(init_path), device=str(dev)))
            init_info = {"source": str(init_path), "which": init_w, "select_on": select_on,
                         "scores": {w: float(s) for s, w, _, _ in scored},
                         "seat0": float(init_res["seat0"]), "seat1": float(init_res["seat1"])}
            if (src / "train_state.pt").exists():
                old = torch.load(src / "train_state.pt", map_location="cpu", weights_only=True)
                ret_rms.load_state_dict(old["ret_rms"])
                init_info["critic_slots_mapped"] = mapped = _warm_start_critic(critic, old["critic"], old["slot_names"], slot_names)
                print(f"[INIT] critic warm-started from {src / 'train_state.pt'} ({mapped}/{n_slots} slot embeddings mapped)")
            warm_started = True

            # A teacher's type, per seat, is the type whose head best predicts its actions; untrained `init`
            # members are novices.
            t_ids = list(range(n_pop, n_pool))
            if t_ids:
                _, ll = evaluate_against_pool(student, layout_id, [train_pool[i] for i in t_ids], dev,
                                              episodes_per_partner=2, partner_greedy_frac=0.0, return_bucket_ll=True)
                for j, i in enumerate(t_ids):
                    if train_pool[i].stage == "init":
                        slot_types[i] = NOVICE
                    else:
                        for s in (0, 1):
                            slot_types[i, s] = CONVENTIONS[int(np.argmax(ll[j, s, CONVENTIONS]))]
                counts = np.bincount(slot_types[t_ids].ravel(), minlength=k)
                init_info["teacher_bucket_counts"] = counts.tolist()
                print(f"[INIT] teachers per partner type 0..11: {counts.tolist()}")
        for net in copies:
            net.load_state_dict(student.state_dict())
        if warm_started:
            best_score = float(init_score)
            _save_weights(student, best_path)
            _save_weights(student, latest_path)
            add_candidate(0, best_score)
            writer.add_scalar("select/best", best_score, 0)
            log_dev(init_res, 0)
        print(f"[{layout_id}] Loaded {n_pop} population + {n_teacher} teacher train + {n_cm} classmate train partners "
              f"({len(cm_dev_pool)} classmate dev), {len(holdout_pool)} population holdout, {len(dev_pool)} teacher dev "
              f"(envs={num_envs}, hidden={hidden_size}, K={k}, aug={state_aug_prob:.2f}, other_play={other_play}, "
              f"bf16={use_amp}, select_on={select_on}, warm={warm_started}).")
    slot_types_t = torch.from_numpy(slot_types).to(dev)

    # 72% of the draws go to pool slots: fixed shares for teachers and classmates, the rest to the population.
    groups = [(ids, share) for ids, share in ((np.arange(n_pop), None), (np.arange(n_pop, cm_lo), teacher_share),
                                              (np.arange(cm_lo, n_pool), classmate_share)) if len(ids)]
    fixed = [float(np.clip(share, 0.0, 0.72)) for _, share in groups if share is not None]
    if len(groups) == 1:
        shares = [(groups[0][0], 0.72)]
    elif n_pop == 0:  # no population to take the rest: rescale the fixed shares
        shares = [(ids, 0.72 * f / (sum(fixed) or 1.0)) for (ids, _), f in zip(groups, fixed)]
    else:
        rest = max(0.0, 0.72 - sum(fixed))
        shares = [(ids, rest if share is None else float(np.clip(share, 0.0, 0.72))) for ids, share in groups]
    print(f"[{layout_id}] partner shares: " + ", ".join(f"{len(ids)} slots -> {share:.2f}" for ids, share in shares))

    def sample_partners(count: int) -> np.ndarray:
        regret = np.maximum(1.0, np.maximum(0.0, partner_sp - partner_ema) + 0.5 * np.sqrt(np.maximum(0.0, partner_var)))
        probs = np.zeros(n_slots)
        for ids, share in shares:
            if share > 0.0:
                r = regret[ids]
                group = 0.80 * (r / r.sum()) + 0.20 / len(ids)
                probs[ids] = share * group / group.sum()
        probs[n_pool:] = (0.08, 0.04, 0.04, 0.07, 0.05)  # the special slots
        return rng.choice(n_slots, size=count, replace=True, p=probs / probs.sum())

    def draw_partner_modes(pids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-episode partner mode: argmax, sampling with noise mixed into the policy (UPD), or plain sampling.

        Population partners: 50% argmax, 25% noised, 25% sampled. Teachers and classmates mostly sample plainly,
        as the course's FrozenPartner and the course template play.
        """
        is_t = (pids >= n_pop) & (pids < n_pool)
        greedy = rng.random(len(pids)) < np.where(is_t, teacher_greedy_prob, 0.50)
        upd = (~greedy) & (rng.random(len(pids)) < np.where(is_t, teacher_upd_prob, 0.50))
        return greedy, upd

    # Per-environment partner, seat, flip and mode, redrawn only when that environment's episode ends.
    seat1_prob = 0.60 if layout_id == "unident_s_hard" else 0.50  # asymmetric green seat learns slower
    env_pids = sample_partners(num_envs)
    env_pids_t = torch.from_numpy(env_pids).to(dev)
    seats_np = (rng.random(num_envs) < seat1_prob).astype(np.int64)
    seats_t = torch.from_numpy(seats_np).to(dev)
    flip = (torch.randint(0, 2, (num_envs,), dtype=torch.bool, device=dev) if other_play
            else torch.zeros(num_envs, dtype=torch.bool, device=dev))
    greedy_np, noised_np = draw_partner_modes(env_pids)
    greedy_t, noised_t = torch.from_numpy(greedy_np).to(dev), torch.from_numpy(noised_np).to(dev)

    upd_m = torch.full((num_envs, 6), 1.0 / 6.0, device=dev)
    upd_eps = torch.full((num_envs,), 0.20, device=dev)

    def resample_noise(mask: np.ndarray) -> None:
        idxs = np.nonzero(mask)[0]
        if len(idxs) == 0:
            return
        in_pool = env_pids[idxs] < n_pool
        # Noise is a Dirichlet action distribution mixed in with weight eps: mild for pool partners
        # (alpha 0.3, eps 0.05-0.3), anything up to fully random for the noised student (alpha 0.1-1, eps 0-1).
        alphas = np.where(in_pool, 0.3, rng.uniform(0.1, 1.0, size=len(idxs)))
        ms = np.stack([rng.dirichlet(np.full(6, a)) for a in alphas]).astype(np.float32)
        eps = np.where(in_pool, rng.uniform(0.05, 0.30, size=len(idxs)), rng.uniform(0.0, 1.0, size=len(idxs)))
        idx_t = torch.from_numpy(idxs).to(dev)
        upd_m[idx_t] = torch.from_numpy(ms).to(dev)
        upd_eps[idx_t] = torch.from_numpy(eps.astype(np.float32)).to(dev)

    resample_noise(np.ones(num_envs, dtype=bool))

    # Half the first episodes start mid-game, so episode ends stay spread out from then on.
    env.reset(aug_prob=0.5 if state_aug_prob > 0 else 0.0)
    from_aug = env.augmented.numpy().copy()
    clean = ~from_aug & ~(noised_np & (env_pids < n_pool))
    from_aug_t = torch.from_numpy(from_aug.astype(np.float32)).to(dev)

    def pool_envs(local: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        """Environments whose partner is in one batched module, sorted by its row: (environment, row, rank
        within the row, environments of the fullest row)."""
        loc = np.full(num_envs, -1, dtype=np.int64)
        in_pool = env_pids < n_pool
        loc[in_pool] = local[env_pids[in_pool]]
        envs = np.nonzero(loc >= 0)[0]
        if len(envs) == 0:
            return None
        envs = envs[np.argsort(loc[envs], kind="stable")]
        rows = loc[envs]
        counts = np.bincount(rows)
        ranks = np.arange(len(rows)) - (np.cumsum(counts) - counts)[rows]
        return torch.from_numpy(envs).to(dev), torch.from_numpy(rows).to(dev), torch.from_numpy(ranks).to(dev), int(counts.max())

    pool_groups = [pool_envs(local) for *_, local in pools]
    moving_mask = env_pids != STAY  # states worth recording for augmented restarts

    ch_perm = torch.tensor(ROT180_CH_PERM, device=dev)
    rot_act = torch.tensor(ROT180_ACT_NP, dtype=torch.long, device=dev)
    zone_swap = torch.tensor(ZONE_SWAP, device=dev)
    env_idx = torch.arange(num_envs, device=dev)
    sp_ref = np.full(n_slots, 200.0, dtype=np.float32)
    sp_ref[:n_pool] = partner_sp
    sp_ref_t = torch.from_numpy(sp_ref / 300.0).to(dev)

    L = student.num_layers
    hidden = torch.zeros((L, num_envs, hidden_size), device=dev)
    belief = torch.full((num_envs, k), 1.0 / k, device=dev)
    prev_plogits = torch.zeros((num_envs, k, 6), device=dev)
    copy_hidden, copy_belief, copy_plogits = hidden.clone(), belief.clone(), prev_plogits.clone()
    # The student's previous actions as it saw them (rotated under the flip), and as the environment did.
    prev_self = torch.full((num_envs,), 6, dtype=torch.long, device=dev)
    prev_partner, prev_self_env, prev_partner_env = prev_self.clone(), prev_self.clone(), prev_self.clone()
    ep_scores = np.zeros(num_envs, dtype=np.float32)

    def buffer(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty((rollout_steps, num_envs, *shape), dtype=dtype, device=dev)

    long = torch.long
    keys = ("obs", "prev_self", "prev_partner", "seat", "time", "act", "partner_act", "type", "slot", "priv", "logp")
    buf = dict(zip(keys, (buffer(26, H, W), buffer(dtype=long), buffer(dtype=long), buffer(), buffer(), buffer(dtype=long),
                          buffer(dtype=long), buffer(dtype=long), buffer(dtype=long), buffer(PrivilegedCriticNet.PRIV_DIM), buffer())))
    val_buf, rew_buf, done_buf = buffer(), buffer(), buffer()

    state_aug_end = min(0.10, state_aug_prob)
    assert rollout_steps % chunk_len == 0, "rollout_steps must be divisible by chunk_len"
    num_chunks = rollout_steps // chunk_len
    total_seqs = num_chunks * num_envs
    t0 = time.perf_counter()

    def observe() -> tuple[torch.Tensor, ...]:
        """The student's inputs (rotated under the flip), the partner's view and seat, and the critic's features."""
        obs = env.obs.to(dev, non_blocking=True)
        t_left = 1.0 - env.timestep.to(dev, non_blocking=True).float() / float(spec.horizon)
        ego_obs, p_obs = obs[env_idx, seats_t], obs[env_idx, 1 - seats_t]
        p_seats, ego_seats = (1 - seats_t).float(), seats_t.float()
        if other_play:
            ego_obs = torch.where(flip.view(num_envs, 1, 1, 1), rot180_obs_gpu(ego_obs, ch_perm), ego_obs)
            ego_seats = torch.where(flip, p_seats, ego_seats)

        # 12 numbers on how the partner acts, then 10 from its view
        in_pool = env_pids_t < n_pool
        noised = (env_pids_t == NOISED_EMA) | (in_pool & noised_t)
        meta = torch.cat((p_seats.unsqueeze(-1), from_aug_t.unsqueeze(-1), (in_pool & greedy_t).float().unsqueeze(-1),
                          sp_ref_t[env_pids_t].unsqueeze(-1), ema_ref_t[env_pids_t].unsqueeze(-1),
                          torch.where(noised, upd_eps, torch.zeros_like(upd_eps)).unsqueeze(-1),
                          torch.where(noised.unsqueeze(-1), upd_m, torch.zeros_like(upd_m))), dim=-1)
        priv = torch.cat((meta, extract_privileged_obs_feats(p_obs)), dim=-1)
        return ego_obs, ego_seats, t_left, p_obs, p_seats, priv

    def act_student(ego_obs, ego_seats, t_left, priv):
        with autocast(use_amp):
            logits, p_logits, feat, h, b = student(
                ego_obs.unsqueeze(1), prev_self.unsqueeze(1), prev_partner.unsqueeze(1), ego_seats.unsqueeze(1),
                t_left.unsqueeze(1), hidden, belief, prev_plogits, return_features=True,
            )
            value = critic(feat, env_pids_t.unsqueeze(1), priv.unsqueeze(1))
        return logits, p_logits, h, b, ret_rms.unnormalize(value[:, 0].float())

    def pool_partner_actions(logits: torch.Tensor, envs: torch.Tensor) -> torch.Tensor:
        """Greedy / UPD-noised / sampled actions for pool partners from their logits."""
        probs = F.softmax(logits, dim=-1)
        eps = upd_eps[envs].unsqueeze(-1)
        mixed = (1.0 - eps) * probs + eps * upd_m[envs]
        sampled = torch.multinomial(torch.where(noised_t[envs].unsqueeze(-1), mixed, probs), 1).squeeze(-1)
        return torch.where(greedy_t[envs], logits.argmax(dim=-1), sampled)

    for update in range(start_update, num_updates + 1):
        frac = (update - 1) / max(1, num_updates)
        opt.param_groups[0]["lr"] = lr * (1.0 - 0.85 * frac)
        ent_coef = ent_coef_start + (ent_coef_end - ent_coef_start) * frac
        shaping_coef = max(0.0, shaping_start * (1.0 - 1.5 * frac))
        coll_coef = collision_penalty * max(0.0, 1.0 - frac)
        cur_aug_prob = state_aug_prob + (state_aug_end - state_aug_prob) * frac
        critic_only = warm_started and update <= critic_warmup_updates

        with torch.no_grad():
            for p_ema, p_stu in zip(ema_net.parameters(), student.parameters()):
                p_ema.lerp_(p_stu, 0.02)
            if not hist1_latched and frac >= 0.25:
                hist1_net.load_state_dict(student.state_dict())
                hist1_latched = True
            if not hist2_latched and frac >= 0.60:
                hist2_net.load_state_dict(student.state_dict())
                hist2_latched = True

        ema_ref = np.full(n_slots, float(partner_ema.mean() / 300.0) if n_pool > 0 else 0.3, dtype=np.float32)
        ema_ref[:n_pool] = partner_ema.astype(np.float32) / 300.0
        ema_ref_t = torch.from_numpy(ema_ref).to(dev)

        chunk_hidden = torch.empty((num_chunks, L, num_envs, hidden_size), device=dev)
        chunk_belief = torch.empty((num_chunks, num_envs, k), device=dev)
        chunk_plogits = torch.empty((num_chunks, num_envs, k, 6), device=dev)

        student.eval()
        critic.eval()
        with torch.inference_mode():
            for t in range(rollout_steps):
                if t % chunk_len == 0:
                    c = t // chunk_len
                    chunk_hidden[c], chunk_belief[c], chunk_plogits[c] = hidden, belief, prev_plogits

                ego_obs, ego_seats, t_left, p_obs, p_seats, priv = observe()
                logits, p_logits, hidden, belief, value = act_student(ego_obs, ego_seats, t_left, priv)
                hidden = hidden.float()
                prev_plogits = p_logits[:, -1].float()
                logits = logits[:, 0].float()
                ego_act = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
                ego_logp = F.log_softmax(logits, dim=-1).gather(-1, ego_act.unsqueeze(-1)).squeeze(-1)

                partner_act = torch.empty(num_envs, dtype=torch.long, device=dev)
                for (arch, size, module, _), group in zip(pools, pool_groups):
                    if group is None:
                        continue
                    envs, rows, ranks, width = group
                    pad_obs = torch.zeros((size, width, 26, H, W), device=dev)
                    pad_seat, pad_time = torch.zeros((size, width), device=dev), torch.zeros((size, width), device=dev)
                    pad_obs[rows, ranks], pad_seat[rows, ranks], pad_time[rows, ranks] = p_obs[envs], p_seats[envs], t_left[envs]
                    logits_pad = pool_logits(arch, module, pad_obs, pad_seat, pad_time, use_amp)
                    partner_act[envs] = pool_partner_actions(logits_pad[rows, ranks], envs)

                # Student copies play from the partner's seat: the EMA (plain or noised) and the two snapshots
                for net, mask, noisy in ((ema_net, (env_pids_t == EMA) | (env_pids_t == NOISED_EMA), True),
                                         (hist1_net, env_pids_t == HIST1, False), (hist2_net, env_pids_t == HIST2, False)):
                    idx = torch.nonzero(mask).squeeze(-1)
                    if idx.numel() == 0:
                        continue
                    with autocast(use_amp):
                        c_logits, c_plogits, h, b = net(
                            p_obs[idx].unsqueeze(1), prev_partner_env[idx].unsqueeze(1), prev_self_env[idx].unsqueeze(1),
                            p_seats[idx].unsqueeze(1), t_left[idx].unsqueeze(1), copy_hidden[:, idx].contiguous(),
                            copy_belief[idx], copy_plogits[idx],
                        )
                    copy_hidden[:, idx], copy_belief[idx], copy_plogits[idx] = h.float(), b.float(), c_plogits[:, -1].float()
                    probs = F.softmax(c_logits[:, 0].float(), dim=-1)
                    if noisy:
                        eps = upd_eps[idx].unsqueeze(-1)
                        probs = torch.where((env_pids_t[idx] == NOISED_EMA).unsqueeze(-1), (1.0 - eps) * probs + eps * upd_m[idx], probs)
                    partner_act[idx] = torch.multinomial(probs, 1).squeeze(-1)

                # The staying chef: stays 85% of the time, otherwise a random action
                stay = torch.nonzero(env_pids_t == STAY).squeeze(-1)
                if stay.numel() > 0:
                    moves = torch.rand(stay.numel(), device=dev) < 0.15
                    partner_act[stay] = torch.where(moves, torch.randint(0, 6, (stay.numel(),), device=dev), 4)

                # Under the flip the student acts in the rotated kitchen; map its actions back.
                ego_act_env, partner_act_seen = ego_act, partner_act
                types = slot_types_t[env_pids_t, 1 - seats_t]
                if other_play:
                    ego_act_env = torch.where(flip, rot_act[ego_act], ego_act)
                    partner_act_seen = torch.where(flip, rot_act[partner_act], partner_act)
                    types = torch.where(flip, zone_swap[types], types)

                step = (ego_obs, prev_self, prev_partner, ego_seats, t_left, ego_act, partner_act_seen, types, env_pids_t, priv, ego_logp)
                for key, x in zip(keys, step):
                    buf[key][t] = x
                val_buf[t] = value

                joint = torch.empty((num_envs, 2), dtype=torch.int32, device=dev)
                joint[env_idx, seats_t] = ego_act_env.to(torch.int32)
                joint[env_idx, 1 - seats_t] = partner_act.to(torch.int32)
                sparse_np, shaped_np, done_np, events_np = env.step(joint.cpu().numpy(), aug_prob=cur_aug_prob)
                if t % 10 == 0:
                    env.record_states(10, moving_mask)

                reward = sparse_np + shaping_coef * shaped_np - coll_coef * events_np[np.arange(num_envs), seats_np, COLLISION]
                rew_buf[t] = torch.from_numpy(reward.astype(np.float32)).to(dev, non_blocking=True)
                done_buf[t] = torch.from_numpy(done_np.astype(np.float32)).to(dev, non_blocking=True)
                ep_scores += sparse_np
                prev_self, prev_partner, prev_self_env, prev_partner_env = ego_act, partner_act_seen, ego_act_env, partner_act

                if done_np.any():
                    n_done = int(done_np.sum())
                    done_t = torch.from_numpy(done_np).to(dev)
                    # Regret statistics only from clean episodes (a standard start, no added noise): a
                    # mid-game start or a noised partner scores differently from the reference.
                    for e in np.nonzero(done_np)[0]:
                        p = int(env_pids[e])
                        if p < n_pool and clean[e]:
                            diff = float(ep_scores[e]) - partner_ema[p]
                            partner_ema[p] += 0.1 * diff
                            partner_var[p] = 0.9 * partner_var[p] + 0.1 * (diff * diff)
                    ep_scores[done_np] = 0.0

                    for x in (hidden, copy_hidden):
                        x[:, done_t] = 0.0
                    for x in (belief, copy_belief):
                        x[done_t] = 1.0 / k
                    for x in (prev_plogits, copy_plogits):
                        x[done_t] = 0.0
                    for x in (prev_self, prev_partner, prev_self_env, prev_partner_env):
                        x[done_t] = 6

                    env_pids[done_np] = sample_partners(n_done)
                    env_pids_t = torch.from_numpy(env_pids).to(dev)
                    seats_np[done_np] = (rng.random(n_done) < seat1_prob).astype(np.int64)
                    seats_t = torch.from_numpy(seats_np).to(dev)
                    if other_play:
                        flip[done_t] = torch.randint(0, 2, (n_done,), dtype=torch.bool, device=dev)
                    greedy_np[done_np], noised_np[done_np] = draw_partner_modes(env_pids[done_np])
                    greedy_t, noised_t = torch.from_numpy(greedy_np).to(dev), torch.from_numpy(noised_np).to(dev)
                    resample_noise(done_np)

                    from_aug[done_np] = env.augmented.numpy()[done_np]
                    clean[done_np] = ~from_aug[done_np] & ~(noised_np[done_np] & (env_pids[done_np] < n_pool))
                    from_aug_t = torch.from_numpy(from_aug.astype(np.float32)).to(dev)
                    moving_mask = env_pids != STAY
                    pool_groups = [pool_envs(local) for *_, local in pools]

            ego_obs, ego_seats, t_left, _, _, priv = observe()
            next_value = act_student(ego_obs, ego_seats, t_left, priv)[-1]

        # GAE in the true reward scale
        adv = gae(rew_buf, done_buf, val_buf, next_value, gamma, gae_lambda)
        ret = adv + val_buf
        ret_rms.update(ret)

        def seq(x: torch.Tensor) -> torch.Tensor:  # (T, N, ...) -> (chunks * N, chunk_len, ...): recurrent PPO
            return x.reshape(num_chunks, chunk_len, num_envs, *x.shape[2:]).transpose(1, 2).reshape(total_seqs, chunk_len, *x.shape[2:])

        s = {key: seq(x) for key, x in buf.items()}
        s_adv, s_ret = seq(adv), seq(ret_rms.normalize(ret))
        s_h0 = chunk_hidden.permute(0, 2, 1, 3).reshape(total_seqs, L, hidden_size)
        s_b0, s_pl0 = chunk_belief.reshape(total_seqs, k), chunk_plogits.reshape(total_seqs, k, 6)
        s_adv = (s_adv - s_adv.mean()) / (s_adv.std() + 1e-8)

        student.train()
        critic.train()
        mb_seqs = max(1, total_seqs // minibatches)
        for _ in range(4):
            perm = torch.randperm(total_seqs, device=dev)
            for start in range(0, total_seqs, mb_seqs):
                i = perm[start:start + mb_seqs]
                with autocast(use_amp):
                    logits, p_logits, feat, _, _ = student(
                        s["obs"][i], s["prev_self"][i], s["prev_partner"][i], s["seat"][i], s["time"][i],
                        s_h0[i].transpose(0, 1).contiguous(), s_b0[i], s_pl0[i], return_features=True,
                    )
                    # Right after a warm start the new critic is fitted first, without moving the student.
                    value = critic(feat.detach() if critic_only else feat, s["slot"][i], s["priv"][i])
                logits = logits.float()
                p_logits = p_logits.float()
                value = value.float()

                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                ratio = (log_probs.gather(-1, s["act"][i].unsqueeze(-1)).squeeze(-1) - s["logp"][i]).exp()
                surr1 = ratio * s_adv[i]
                surr2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * s_adv[i]
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.smooth_l1_loss(value, s_ret[i], beta=2.0)
                ent = -(probs * log_probs).sum(dim=-1).mean()
                # Partner-type head: predict the partner's next action under the slot's partner type.
                typed = p_logits.gather(-2, s["type"][i][..., None, None].expand(-1, -1, 1, 6)).squeeze(-2)
                aux_loss = F.cross_entropy(typed.reshape(-1, 6), s["partner_act"][i].reshape(-1), label_smoothing=0.03)
                loss = 0.5 * v_loss if critic_only else pg_loss + 0.5 * v_loss - ent_coef * ent + partner_aux_coef * aux_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()

        cur_eval_every = max(1, eval_every // 2) if update >= int(0.70 * num_updates) else eval_every
        if update % cur_eval_every == 0 or update == num_updates:
            _save_weights(student, latest_path)
            if holdout_pool:
                holdout = evaluate_against_pool(student, layout_id, holdout_pool, dev, episodes_per_partner=4,
                                                temperature=0.0, belief_eta=1.0)
            else:
                holdout = {key: float("nan") for key in ("mean", "holdout_mlp", "holdout_cnn")}
            dev_res = None
            if dev_pool:
                select, dev_res = eval_select(student, dev_episodes)
            else:
                select = float(holdout["mean"])
            if select >= best_score:
                best_score = select
                _save_weights(student, best_path)
            if not critic_only:
                add_candidate(update, select)
            save_state(update)

            step = update * steps_per_update
            sps = (update - start_update + 1) * steps_per_update / max(1e-6, time.perf_counter() - t0)
            nan = float("nan")
            pop_ema = float(partner_ema[:n_pop].mean()) if n_pop else nan
            teacher_ema = float(partner_ema[n_pop:cm_lo].mean()) if n_teacher else nan
            cm_ema = float(partner_ema[cm_lo:].mean()) if n_cm else nan
            scalars = {
                "train/sps": sps, "train/lr": opt.param_groups[0]["lr"], "train/ent_coef": ent_coef,
                "train/aug_prob": cur_aug_prob, "train/ret_std": ret_rms.std,
                "partner_ema/all": partner_ema.mean(), "partner_ema/population": pop_ema,
                "partner_ema/teacher": teacher_ema, "partner_ema/classmate": cm_ema,
                "holdout/mean": holdout["mean"], "holdout/mlp": holdout["holdout_mlp"],
                "holdout/cnn": holdout["holdout_cnn"], "select/score": select, "select/best": best_score,
                "loss/policy": pg_loss.item(), "loss/value": v_loss.item(), "loss/partner_aux": aux_loss.item(),
                "loss/entropy": ent.item(),
            }
            for tag, x in scalars.items():
                writer.add_scalar(tag, float(x), step)
            msg = ""
            if dev_res is not None:
                log_dev(dev_res, step)
                msg = f"dev={dev_res['mean']:.1f} (s0={dev_res['seat0']:.1f}, s1={dev_res['seat1']:.1f}) "
                if "cm_mean" in dev_res:
                    msg += f"cm_dev={dev_res['cm_mean']:.1f} (s0={dev_res['cm_seat0']:.1f}, s1={dev_res['cm_seat1']:.1f}) "
            print(f"[{layout_id}] upd={update}/{num_updates} steps={step:,} aug={cur_aug_prob:.2f} ring={env.ring_size} "
                  f"ret_std={ret_rms.std:.1f} train_ema={partner_ema.mean():.1f} (pop={pop_ema:.1f}, "
                  f"teacher={teacher_ema:.1f}, cm={cm_ema:.1f}) {msg}holdout={holdout['mean']:.1f} "
                  f"(mlp={holdout['holdout_mlp']:.1f}, cnn={holdout['holdout_cnn']:.1f}) best={best_score:.1f}[{select_on}] "
                  f"ent={ent.item():.2f} aux={aux_loss.item():.2f} {'[critic-warmup] ' if critic_only else ''}({sps:,.0f} sps)")

    selection: dict = {"select_on": select_on, "init": init_info}
    # Rescore the top candidates on more dev episodes: the best score of many noisy evaluations is optimistic.
    if dev_pool and candidates:
        rescored = []
        for cand in candidates:
            student.load_state_dict(load_file(str(out / cand["file"]), device=str(dev)))
            score, res = eval_select(student, final_dev_episodes)
            rescored.append({**cand, "rescore": score, "dev": res["mean"], "seat0": res["seat0"], "seat1": res["seat1"],
                             **{key: res[key] for key in ("cm_mean", "cm_seat0", "cm_seat1", "teacher_select", "cm_select") if key in res}})
            print(f"[{layout_id}] candidate upd={cand['update']}: dev select {cand['score']:.1f} -> {score:.1f} "
                  f"(mean={res['mean']:.1f}, seat0={res['seat0']:.1f}, seat1={res['seat1']:.1f})")
        chosen = max(rescored, key=lambda c: c["rescore"])
        shutil.copy2(out / chosen["file"], best_path)
        best_score = float(chosen["rescore"])
        selection |= {"candidates": rescored, "chosen_update": int(chosen["update"])}

    # Choose the belief eta and the sampling temperature for the chosen checkpoint.
    student.load_state_dict(load_file(str(best_path), device=str(dev)))
    best_eta, best_temp, best_grid_score = 1.0, 0.0, -1.0
    grid = []
    for eta in (0.6, 1.0, 1.5):
        for temp in (0.0, 0.08, 0.15):
            score, res = eval_select(student, final_dev_episodes if dev_pool else 8, temperature=temp, eta=eta)
            grid.append({"eta": eta, "temp": temp, "score": score, "mean": res["mean"], "seat0": res["seat0"],
                         "seat1": res["seat1"], **{key: res[key] for key in ("cm_mean", "cm_seat0", "cm_seat1") if key in res}})
            if score > best_grid_score:
                best_grid_score, best_eta, best_temp = score, eta, temp
    print(f"[{layout_id}] Best {select_on} (eta={best_eta:.2f}, temp={best_temp:.2f}) -> score={best_grid_score:.1f}")
    selection |= {"grid": grid, "best_eta": best_eta, "best_temp": best_temp, "best_score": best_grid_score}
    (out / "selection.json").write_text(json.dumps(selection, indent=1) + "\n", encoding="utf-8")
    save_state(num_updates, done=True, best_temp=best_temp, best_eta=best_eta)
    env.close()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", nargs="+", default=["counter_circuit", "schelling", "unident_s_hard"])
    parser.add_argument("--partners", type=Path, default=ROOT / "runs/population")
    parser.add_argument("--teachers", type=Path, help="teacher pack (train and dev members)")
    parser.add_argument("--classmates", type=Path, help="classmate pool (train and dev runs)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, help="fine-tune from <dir>/<layout>/{latest,best}.safetensors")
    parser.add_argument("--init-which", choices=("auto", "latest", "best"), default="auto")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--chunk-len", type=int, default=40)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--steps-per-map", type=int, default=None, help="override steps for all maps (default: schelling=160M, others=120M)")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef-start", type=float, default=0.02)
    parser.add_argument("--ent-coef-end", type=float, default=0.003)
    parser.add_argument("--shaping-start", type=float, default=0.4, help="shaped-reward weight, decays to 0 at 2/3 of training")
    parser.add_argument("--collision-penalty", type=float, default=0.05)
    parser.add_argument("--partner-aux-coef", type=float, default=0.25)
    parser.add_argument("--state-aug-prob", type=float, default=0.35)
    parser.add_argument("--critic-warmup-updates", type=int, default=0)
    parser.add_argument("--teacher-share", type=float, default=0.40, help="share of the 0.72 pool draws for teachers")
    parser.add_argument("--teacher-greedy-prob", type=float, default=0.15)
    parser.add_argument("--teacher-upd-prob", type=float, default=0.15)
    parser.add_argument("--classmate-share", type=float, default=0.20, help="share of the 0.72 pool draws for classmates")
    parser.add_argument("--classmate-weight", type=float, default=0.5, help="weight of classmate dev in the selection")
    parser.add_argument("--min-partner-sp", type=float, default=20.0, help="drop population styles below this self-play")
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--dev-episodes", type=int, default=8, help="per dev member and seat at each evaluation")
    parser.add_argument("--final-dev-episodes", type=int, default=24, help="per dev member and seat for the final choice")
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = vars(parser.parse_args())
    default_steps = {"counter_circuit": 120_000_000, "schelling": 160_000_000, "unident_s_hard": 120_000_000}
    override_steps = args.pop("steps_per_map")
    for layout_id in args.pop("layouts"):
        steps = override_steps if override_steps is not None else default_steps.get(layout_id, 120_000_000)
        train_map(layout_id, steps_per_map=steps, **args)


if __name__ == "__main__":
    main()
