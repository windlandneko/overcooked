"""Simulated classmates: partners trained the way other groups plausibly trained theirs, mostly by running the
course material with small changes:
  * the course template (train.py: 2x128 tanh MLP, parameter-sharing self-play, shaped reward), often longer
    or with tweaked hyper-parameters;
  * the same template trained with the course's FCP/MEP partner packs instead of itself;
  * the teaching kit's `train.py --algorithm {fcp,mep,sp} --stage 2`, whose learner is the packs' own CNN;
  * a few with dense event rewards and mid-game starts, which is what it takes for self-play to deliver on
    counter_circuit at all.

Each run is one classmate, and snapshots at several budgets stand in for classmates who stopped earlier. Whole
runs are split train / dev / test, so a test classmate shares no run with anything the student trains or is
selected on; the packs are restricted to their train members, so teacher dev and test stay unseen too.

    python -m zsc.classmates --layouts counter_circuit --group mlp
    python -m zsc.classmates --layouts counter_circuit --group kit
    python -m zsc.classmates --layouts counter_circuit --evaluate

Output: runs/classmates/<map>/<run>/<snapshot>.safetensors and manifest.json, written by --evaluate with each
snapshot's self-play score and its score with the teachers.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

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
    BatchedMLPPool,
    BatchedTeacherPool,
    MLPPartnerNet,
    TeacherNet,
    adaptive_avg_pool2d_any,
)
from zsc.teachers import PACK, load_teacher_members

DEFAULT_OUT = ROOT / "runs/classmates"
M = 1_000_000


def ev(**kw: float) -> tuple[float, ...]:
    return tuple(float(kw.get(k, 0.0)) for k in EVENTS)


DENSE_A = ev(pot_onion=3.0, start_cook=2.0, dish_pickup=2.0, soup_pickup=4.0, collision=-0.05)
DENSE_B = ev(pot_onion=2.0, start_cook=3.0, dish_pickup=1.0, soup_pickup=5.0, soup_deliver=2.0)
DENSE_C = ev(pot_onion=4.0, start_cook=1.0, dish_pickup=3.0, soup_pickup=3.0, collision=-0.2)


@dataclass(frozen=True)
class Classmate:
    name: str
    family: str
    split: str  # train / dev / test (whole run)
    arch: str  # "mlp" (template Network) or "teacher" (kit PartnerNetwork)
    partners: str  # sp | fcp | mep | fcp+mep | sp+fcp
    seed: int
    snapshots: tuple[int, ...]
    lr: float = 3e-4
    ent: float = 0.02
    shaping: float = 1.0
    anneal: float = 0.0  # fraction of the final budget over which shaping decays to 0 (0 = constant)
    events: tuple[float, ...] | None = None  # dense per-seat event reward
    aug: bool = False  # mid-game starts

    @property
    def budget(self) -> int:
        return max(self.snapshots)

    @property
    def sp_frac(self) -> float:
        return {"sp": 1.0, "sp+fcp": 0.5}.get(self.partners, 0.0)


def roster() -> list[Classmate]:
    """8 families x 4 runs; per family 2 train, 1 dev, 1 test runs."""
    out: list[Classmate] = []

    def fam(family: str, arch: str, base_seed: int, variants: list[dict]) -> None:
        for i, (split, kw) in enumerate(zip(("train", "train", "dev", "test"), variants)):
            out.append(Classmate(name=f"{family}_{i + 1}", family=family, split=split, arch=arch,
                                 seed=base_seed + i, **kw))

    # Template self-play, close to defaults (budget raised to a few million)
    fam("tpl_sp", "mlp", 7100, [
        dict(partners="sp", snapshots=(2 * M, 6 * M)),
        dict(partners="sp", snapshots=(3 * M, 10 * M)),
        dict(partners="sp", snapshots=(2 * M, 8 * M)),
        dict(partners="sp", snapshots=(4 * M, 12 * M)),
    ])
    # Template self-play with the hyper-parameter changes a coding assistant would suggest
    fam("tpl_tuned", "mlp", 7200, [
        dict(partners="sp", snapshots=(5 * M, 15 * M), anneal=0.6),
        dict(partners="sp", snapshots=(5 * M, 15 * M), ent=0.01, lr=5e-4),
        dict(partners="sp", snapshots=(5 * M, 15 * M), anneal=0.5, ent=0.05),
        dict(partners="sp", snapshots=(5 * M, 15 * M), lr=1e-4, shaping=0.5),
    ])
    # Template self-play with dense event rewards and mid-game starts (the ones that get competent)
    fam("tpl_dense", "mlp", 7300, [
        dict(partners="sp", snapshots=(6 * M, 15 * M), events=DENSE_A, aug=True, anneal=0.7),
        dict(partners="sp", snapshots=(6 * M, 15 * M), events=DENSE_B, aug=True, anneal=0.7, ent=0.01),
        dict(partners="sp", snapshots=(6 * M, 15 * M), events=DENSE_C, aug=True, anneal=0.7),
        dict(partners="sp", snapshots=(6 * M, 15 * M), events=DENSE_A, aug=True, anneal=0.7, lr=5e-4),
    ])
    # Template learner trained against the FCP16 pack (and variants)
    fam("tpl_fcp", "mlp", 7400, [
        dict(partners="fcp", snapshots=(2 * M, 8 * M)),
        dict(partners="sp+fcp", snapshots=(3 * M, 12 * M), anneal=0.6),
        dict(partners="fcp", snapshots=(1 * M, 6 * M), ent=0.01),
        dict(partners="fcp", snapshots=(2 * M, 10 * M), anneal=0.5),
    ])
    # Template learner trained against MEP16 (or both packs)
    fam("tpl_mep", "mlp", 7500, [
        dict(partners="mep", snapshots=(2 * M, 8 * M)),
        dict(partners="fcp+mep", snapshots=(3 * M, 12 * M), anneal=0.6),
        dict(partners="mep", snapshots=(1 * M, 6 * M)),
        dict(partners="fcp+mep", snapshots=(2 * M, 10 * M)),
    ])
    # Teaching kit: --algorithm fcp --stage 2 (default 200k steps; the course env is slow, so few go past 3M)
    fam("kit_fcp", "teacher", 7600, [
        dict(partners="fcp", snapshots=(200_000, 1 * M, 3 * M)),
        dict(partners="fcp", snapshots=(200_000, 1 * M, 3 * M), anneal=0.5),
        dict(partners="fcp", snapshots=(200_000, 1 * M, 3 * M)),
        dict(partners="fcp", snapshots=(200_000, 1 * M, 3 * M), ent=0.01),
    ])
    # Teaching kit: --algorithm mep --stage 2
    fam("kit_mep", "teacher", 7700, [
        dict(partners="mep", snapshots=(200_000, 1 * M, 3 * M)),
        dict(partners="mep", snapshots=(200_000, 1 * M, 3 * M), anneal=0.5),
        dict(partners="mep", snapshots=(200_000, 1 * M, 3 * M)),
        dict(partners="fcp+mep", snapshots=(200_000, 1 * M, 3 * M)),
    ])
    # Teaching kit self-play (no pack used)
    fam("kit_sp", "teacher", 7800, [
        dict(partners="sp", snapshots=(1 * M, 3 * M)),
        dict(partners="sp", snapshots=(1 * M, 3 * M), anneal=0.5),
        dict(partners="sp", snapshots=(1 * M, 3 * M)),
        dict(partners="sp", snapshots=(1 * M, 3 * M), ent=0.01),
    ])
    return out


def tag_of(steps: int) -> str:
    return f"s{steps / M:g}M"


# --------------------------------------------------------------------------------------------------
# Vectorized learners (S independent networks trained in one pass) and a per-run Adam
# --------------------------------------------------------------------------------------------------


class VecTemplateNet(nn.Module):
    """S template `Network`s: Flatten -> 128 Tanh -> 128 Tanh -> actor / critic (orthogonal init)."""

    def __init__(self, s: int, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.s, self.obs_dim = s, obs_dim

        def stack(rows: int, cols: int, gain: float) -> torch.Tensor:
            out = torch.empty(s, rows, cols)
            for i in range(s):
                w = torch.empty(cols, rows)
                nn.init.orthogonal_(w, gain)
                out[i] = w.T
            return out

        self.w1 = nn.Parameter(stack(obs_dim, hidden, math.sqrt(2)))
        self.b1 = nn.Parameter(torch.zeros(s, 1, hidden))
        self.w2 = nn.Parameter(stack(hidden, hidden, math.sqrt(2)))
        self.b2 = nn.Parameter(torch.zeros(s, 1, hidden))
        self.wa = nn.Parameter(stack(hidden, 6, 0.01))
        self.ba = nn.Parameter(torch.zeros(s, 1, 6))
        self.wc = nn.Parameter(stack(hidden, 1, 1.0))
        self.bc = nn.Parameter(torch.zeros(s, 1, 1))

    def forward(self, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = grid.reshape(self.s, grid.shape[1], self.obs_dim)
        h = torch.tanh(torch.baddbmm(self.b1, x, self.w1))
        h = torch.tanh(torch.baddbmm(self.b2, h, self.w2))
        return torch.baddbmm(self.ba, h, self.wa), torch.baddbmm(self.bc, h, self.wc).squeeze(-1)

    def export(self, i: int) -> dict[str, torch.Tensor]:
        """MLPPartnerNet keys (net.1 / net.3 / actor / critic)."""
        d = lambda t: t.detach()[i].cpu().contiguous()  # noqa: E731
        return {
            "net.1.weight": d(self.w1).T.contiguous(), "net.1.bias": d(self.b1)[0],
            "net.3.weight": d(self.w2).T.contiguous(), "net.3.bias": d(self.b2)[0],
            "actor.weight": d(self.wa).T.contiguous(), "actor.bias": d(self.ba)[0],
            "critic.weight": d(self.wc).T.contiguous(), "critic.bias": d(self.bc)[0],
        }


class VecKitNet(nn.Module):
    """S teaching-kit learners: the pack's PartnerNetwork (Conv32-Conv32-Pool3-64) plus a critic head."""

    def __init__(self, s: int, hidden: int = 64, pool: int = 3):
        super().__init__()
        self.s, self.pool, self.flat = s, pool, 32 * pool * pool
        g = math.sqrt(2)

        def conv(cin: int) -> torch.Tensor:
            w = torch.empty(s * 32, cin, 3, 3)
            for i in range(s):
                nn.init.orthogonal_(w[i * 32:(i + 1) * 32].view(32, -1), g)
            return w

        def lin(rows: int, cols: int, gain: float) -> torch.Tensor:
            out = torch.empty(s, rows, cols)
            for i in range(s):
                w = torch.empty(cols, rows)
                nn.init.orthogonal_(w, gain)
                out[i] = w.T
            return out

        self.c1 = nn.Parameter(conv(26))
        self.c1b = nn.Parameter(torch.zeros(s * 32))
        self.c2 = nn.Parameter(conv(32))
        self.c2b = nn.Parameter(torch.zeros(s * 32))
        self.fw = nn.Parameter(lin(self.flat, hidden, g))
        self.fb = nn.Parameter(torch.zeros(s, 1, hidden))
        self.hw = nn.Parameter(lin(hidden, 6, 0.01))
        self.hb = nn.Parameter(torch.zeros(s, 1, 6))
        self.vw = nn.Parameter(lin(hidden, 1, 1.0))
        self.vb = nn.Parameter(torch.zeros(s, 1, 1))

    def forward(self, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, b, c, h, w = grid.shape
        x = grid.permute(1, 0, 2, 3, 4).reshape(b, s * c, h, w)
        x = torch.tanh(F.conv2d(x, self.c1, self.c1b, padding=1, groups=s))
        x = torch.tanh(F.conv2d(x, self.c2, self.c2b, padding=1, groups=s))
        x = adaptive_avg_pool2d_any(x, self.pool).reshape(b, s, self.flat).permute(1, 0, 2)
        z = torch.tanh(torch.baddbmm(self.fb, x, self.fw))
        return torch.baddbmm(self.hb, z, self.hw), torch.baddbmm(self.vb, z, self.vw).squeeze(-1)

    def export(self, i: int) -> dict[str, torch.Tensor]:
        """TeacherNet / PartnerNetwork keys (body.0, body.2, body.6, head)."""
        d = lambda t: t.detach().cpu()  # noqa: E731
        sl = slice(i * 32, (i + 1) * 32)
        return {
            "body.0.weight": d(self.c1)[sl].contiguous(), "body.0.bias": d(self.c1b)[sl].contiguous(),
            "body.2.weight": d(self.c2)[sl].contiguous(), "body.2.bias": d(self.c2b)[sl].contiguous(),
            "body.6.weight": d(self.fw)[i].T.contiguous(), "body.6.bias": d(self.fb)[i, 0].contiguous(),
            "head.weight": d(self.hw)[i].T.contiguous(), "head.bias": d(self.hb)[i, 0].contiguous(),
        }


class RunAdam:
    """Adam with one learning rate per run (leading dim of every parameter splits into S equal blocks)."""

    def __init__(self, params: list[nn.Parameter], lrs: torch.Tensor, betas=(0.9, 0.999), eps: float = 1e-5):
        self.params, self.lrs, self.b1, self.b2, self.eps = params, lrs, betas[0], betas[1], eps
        self.s = len(lrs)
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    @torch.no_grad()
    def step(self, active: torch.Tensor) -> None:
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        lr = (self.lrs * active).view(self.s, 1)
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad.view(self.s, -1)
            mv, vv = m.view(self.s, -1), v.view(self.s, -1)
            mv.mul_(self.b1).add_(g, alpha=1 - self.b1)
            vv.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.view(self.s, -1).sub_(lr * (mv / c1) / ((vv / c2).sqrt() + self.eps))

    def clip_per_run(self, max_norm: float = 0.5) -> None:
        norms = torch.zeros(self.s, device=self.lrs.device)
        for p in self.params:
            norms += p.grad.view(self.s, -1).pow(2).sum(1)
        scale = (max_norm / (norms.sqrt() + 1e-6)).clamp(max=1.0).view(self.s, 1)
        for p in self.params:
            p.grad.view(self.s, -1).mul_(scale)


# --------------------------------------------------------------------------------------------------
# Course teacher packs (train split only), played like the course's FrozenPartner
# --------------------------------------------------------------------------------------------------


class PackPartners:
    """Train-split FCP/MEP members; one member per partner-env, resampled every episode.

    `fcp` samples uniformly over all FCP members incl. init/mid snapshots (what FrozenPartner does);
    `mep` over MEP finals; `fcp+mep` picks a pack 50/50 first. Actions are sampled from the softmax.
    """

    def __init__(self, layout_id: str, device: torch.device):
        members = load_teacher_members(PACK, layout_id, ("train",), device)
        self.names = [f"{m.algorithm}-{m.seed}/{m.stage}" for m in members]
        self.fcp = np.array([i for i, m in enumerate(members) if m.algorithm == "fcp"])
        self.mep = np.array([i for i, m in enumerate(members) if m.algorithm == "mep"])
        self.pool = BatchedTeacherPool([m.net for m in members], device)
        self.g = len(members)
        self.device = device

    def draw(self, kinds: list[str], rng: np.random.Generator) -> np.ndarray:
        out = np.empty(len(kinds), dtype=np.int64)
        for i, k in enumerate(kinds):
            src = self.fcp if k == "fcp" else self.mep if k == "mep" else (self.fcp if rng.random() < 0.5 else self.mep)
            out[i] = src[rng.integers(len(src))]
        return out

    @torch.inference_mode()
    def act(self, grids: torch.Tensor, member: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """grids (N, 26, H, W) for N partner-envs, member (N,) -> sampled actions (N,)."""
        n = grids.shape[0]
        counts = torch.bincount(member, minlength=self.g)
        width = int(counts.max().item())
        order = torch.argsort(member, stable=True)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(n, device=grids.device) - starts[member[order]]
        slot = member[order] * width + rank
        buf = torch.zeros((self.g * width, *grids.shape[1:]), device=grids.device)
        buf[slot] = grids[order]
        logits = self.pool(buf.view(self.g, width, *grids.shape[1:])).reshape(self.g * width, 6)[slot]
        acts = torch.multinomial(F.softmax(logits, -1), 1, generator=gen).squeeze(-1)
        out = torch.empty_like(acts)
        out[order] = acts
        return out


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------


def train_group(layout_id: str, runs: list[Classmate], out_dir: Path, *, envs_per_run: int, rollout: int,
                epochs: int, minibatch_size: int, device: str, aug_prob: float = 0.3) -> list[dict]:
    dev = torch.device(device)
    torch.manual_seed(runs[0].seed)
    rng = np.random.default_rng(runs[0].seed)
    gen = torch.Generator(device=dev).manual_seed(runs[0].seed)
    spec = read_scenario(CONFIG, layout_id=layout_id)
    s, e = len(runs), envs_per_run
    n = s * e
    env = VecEnv(spec, n)
    env.seed(runs[0].seed)
    env.reset()
    h, w = env.height, env.width
    arch = runs[0].arch
    net = (VecTemplateNet(s, 26 * h * w) if arch == "mlp" else VecKitNet(s)).to(dev)
    opt = RunAdam(list(net.parameters()), torch.tensor([r.lr for r in runs], device=dev))
    ent = torch.tensor([r.ent for r in runs], device=dev).view(s, 1)
    budget = np.array([r.budget for r in runs])

    # Env roles: SP envs (both seats learner) vs pack envs (learner in one seat, frozen member in the other)
    n_sp = np.array([int(round(r.sp_frac * e)) for r in runs])
    is_sp = np.zeros((s, e), dtype=bool)
    for i in range(s):
        is_sp[i, : n_sp[i]] = True
    is_sp = is_sp.reshape(n)
    pack_idx = np.nonzero(~is_sp)[0]
    pack = PackPartners(layout_id, dev) if len(pack_idx) else None
    pack_kind = [runs[i // e].partners.replace("sp+", "") for i in pack_idx]
    member = torch.from_numpy(pack.draw(pack_kind, rng)).to(dev) if pack else None
    lseat = rng.integers(0, 2, size=n)
    lseat[is_sp] = 0
    pack_idx_t = torch.from_numpy(pack_idx).to(dev)
    aug_env = np.repeat(np.array([r.aug for r in runs]), e)
    ev_w = torch.tensor([r.events if r.events else (0.0,) * len(EVENTS) for r in runs], device=dev)  # (S, 12)
    shaping = np.array([r.shaping for r in runs], dtype=np.float32)
    anneal = np.array([r.anneal for r in runs], dtype=np.float32)

    steps_per_update = e * rollout
    num_updates = int(math.ceil(budget.max() / steps_per_update))
    snap_left = {i: sorted(r.snapshots) for i, r in enumerate(runs)}
    b2 = 2 * e
    obs_buf = torch.empty((rollout, s, b2, 26, h, w), device=dev)
    act_buf = torch.empty((rollout, s, b2), dtype=torch.long, device=dev)
    logp_buf = torch.empty((rollout, s, b2), device=dev)
    val_buf = torch.empty((rollout, s, b2), device=dev)
    rew_buf = torch.empty((rollout, s, b2), device=dev)
    done_buf = torch.empty((rollout, s, b2), device=dev)
    mask_buf = torch.empty((rollout, s, b2), device=dev)
    ep_score = np.zeros(n, dtype=np.float32)
    recent: list[list[float]] = [[] for _ in range(s)]
    saved: list[dict] = []
    t0 = time.perf_counter()
    for r in runs:
        (out_dir / layout_id / r.name).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / layout_id / "tb" / arch)
    print(f"[{layout_id}/{arch}] {s} runs x {e} envs, {num_updates} updates, pack envs={len(pack_idx)}", flush=True)

    for update in range(1, num_updates + 1):
        done_steps = (update - 1) * steps_per_update
        active = torch.from_numpy((done_steps < budget).astype(np.float32)).to(dev)
        frac = done_steps / np.maximum(1, budget)
        sh_scale = np.where(anneal > 0, np.clip(1.0 - frac / np.maximum(anneal, 1e-6), 0.0, 1.0), 1.0) * shaping
        sh_t = torch.from_numpy(sh_scale.astype(np.float32)).to(dev).view(s, 1)
        ev_scale = torch.from_numpy(np.where(anneal > 0, np.clip(1.0 - frac / np.maximum(anneal, 1e-6), 0, 1), 1.0)
                                    .astype(np.float32)).to(dev).view(s, 1)
        net.eval()
        with torch.inference_mode():
            for t in range(rollout):
                obs = env.obs.to(dev).view(s, e, 2, 26, h, w)
                x = torch.cat((obs[:, :, 0], obs[:, :, 1]), dim=1)  # (S, 2E): seat-0 block, seat-1 block
                logits, value = net(x)
                act = torch.multinomial(F.softmax(logits.reshape(-1, 6), -1), 1, generator=gen).view(s, b2)
                logp = F.log_softmax(logits, -1).gather(-1, act.unsqueeze(-1)).squeeze(-1)
                lseat_t = torch.from_numpy(lseat).to(dev).view(s, e)
                sp_t = torch.from_numpy(is_sp).to(dev).view(s, e)
                valid = torch.cat((sp_t | (lseat_t == 0), sp_t | (lseat_t == 1)), dim=1).float()
                joint = torch.stack((act[:, :e], act[:, e:]), dim=-1).reshape(n, 2)
                if pack is not None:
                    pseat = torch.from_numpy(1 - lseat[pack_idx]).to(dev)
                    flat_obs = env.obs.to(dev)
                    p_act = pack.act(flat_obs[pack_idx_t, pseat], member, gen)
                    joint[pack_idx_t, pseat] = p_act
                obs_buf[t], act_buf[t], logp_buf[t], val_buf[t], mask_buf[t] = x, act, logp, value, valid
                sparse, shaped, done, events = env.step(joint.to(torch.int32).cpu().numpy(),
                                                        aug_prob=aug_prob if aug_env.any() else 0.0)
                if done.any() and aug_env.any():
                    fix = done & ~aug_env
                    if fix.any():
                        env.reset(fix)
                sp_t2 = torch.from_numpy(sparse).to(dev).view(s, e)
                sh_t2 = torch.from_numpy(shaped).to(dev).view(s, e)
                ev_t = torch.from_numpy(events).to(dev).view(s, e, 2, -1)
                team = sp_t2 + sh_t * sh_t2
                dense = (ev_t * ev_w.view(s, 1, 1, -1)).sum(-1) * ev_scale.view(s, 1, 1)  # (S, E, 2)
                rew_buf[t] = torch.cat((team + dense[:, :, 0], team + dense[:, :, 1]), dim=1)
                d_t = torch.from_numpy(done.astype(np.float32)).to(dev).view(s, e)
                done_buf[t] = torch.cat((d_t, d_t), dim=1)
                ep_score += sparse
                if done.any():
                    idx = np.nonzero(done)[0]
                    for i in idx:
                        if not aug_env[i]:
                            recent[i // e].append(float(ep_score[i]))
                    ep_score[idx] = 0.0
                    # new episode: resample learner seat and pack member (FrozenPartner: once per episode)
                    if pack is not None:
                        pd = done[pack_idx]
                        if pd.any():
                            sel = np.nonzero(pd)[0]
                            lseat[pack_idx[sel]] = rng.integers(0, 2, size=len(sel))
                            member[torch.from_numpy(sel).to(dev)] = torch.from_numpy(
                                pack.draw([pack_kind[j] for j in sel], rng)).to(dev)
            obs = env.obs.to(dev).view(s, e, 2, 26, h, w)
            _, next_val = net(torch.cat((obs[:, :, 0], obs[:, :, 1]), dim=1))

        adv = torch.zeros_like(rew_buf)
        gae = torch.zeros((s, b2), device=dev)
        for t in reversed(range(rollout)):
            nt = 1.0 - done_buf[t]
            nv = next_val if t == rollout - 1 else val_buf[t + 1]
            delta = rew_buf[t] + 0.99 * nv * nt - val_buf[t]
            gae = delta + 0.99 * 0.95 * nt * gae
            adv[t] = gae
        ret = adv + val_buf
        msum = mask_buf.sum(dim=(0, 2), keepdim=True).clamp(min=1)
        mean = (adv * mask_buf).sum(dim=(0, 2), keepdim=True) / msum
        std = (((adv - mean) ** 2 * mask_buf).sum(dim=(0, 2), keepdim=True) / msum).sqrt() + 1e-8
        adv = (adv - mean) / std

        bs = rollout * b2
        flat = lambda b: b.permute(1, 0, *range(2, b.dim())).reshape(s, bs, *b.shape[3:])  # noqa: E731
        f_obs, f_act, f_logp, f_adv, f_ret, f_mask = map(flat, (obs_buf, act_buf, logp_buf, adv, ret, mask_buf))
        net.train()
        minibatches = max(1, bs // minibatch_size)  # per-run minibatch = the template's 2048 samples
        mb = bs // minibatches
        for _ in range(epochs):
            perm = torch.randperm(bs, device=dev)
            for k in range(minibatches):
                ids = perm[k * mb:(k + 1) * mb]
                lg, v = net(f_obs[:, ids])
                lp = F.log_softmax(lg, -1)
                new = lp.gather(-1, f_act[:, ids].unsqueeze(-1)).squeeze(-1)
                ratio = (new - f_logp[:, ids]).exp()
                a = f_adv[:, ids]
                m = f_mask[:, ids]
                den = m.sum(1, keepdim=True).clamp(min=1)
                pg = -(torch.min(ratio * a, ratio.clamp(0.8, 1.2) * a) * m).sum(1, keepdim=True) / den
                vl = (((v - f_ret[:, ids]) ** 2) * m).sum(1, keepdim=True) / den
                en = (-(lp.exp() * lp).sum(-1) * m).sum(1, keepdim=True) / den
                loss = (pg + 0.5 * vl - ent * en).sum()
                for p in net.parameters():
                    p.grad = None
                loss.backward()
                opt.clip_per_run(0.5)
                opt.step(active)

        steps_now = update * steps_per_update
        for i, r in enumerate(runs):
            while snap_left[i] and steps_now >= snap_left[i][0]:
                target = snap_left[i].pop(0)
                tag = tag_of(target)
                path = out_dir / layout_id / r.name / f"{tag}.safetensors"
                save_file(net.export(i), str(path))
                sp_est = float(np.mean(recent[i][-64:])) if recent[i] else 0.0
                saved.append({
                    "layout_id": layout_id, "style": r.name, "stage": tag, "steps": int(target),
                    "arch": r.arch, "family": r.family, "split": r.split, "kind": r.partners,
                    "holdout": r.split != "train", "train_score_ema": round(sp_est, 2),
                    "weights": str(path.relative_to(out_dir)), "config": asdict(r),
                })
        if update % 10 == 0 or update == num_updates:
            el = time.perf_counter() - t0
            ema = [np.mean(x[-64:]) if x else 0.0 for x in recent]
            writer.add_scalar("sps", update * steps_per_update * s / el, steps_now)
            for r, v in zip(runs, ema):
                writer.add_scalar(f"score/{r.name}", v, steps_now)
            print(f"[{layout_id}/{arch}] upd {update}/{num_updates} steps/run {steps_now / M:.1f}M "
                  f"sps {update * steps_per_update * s / el / 1e3:.0f}k el {el / 60:.1f}m "
                  "score " + " ".join(f"{r.name}:{v:.0f}" for r, v in zip(runs, ema)),
                  flush=True)
    env.close()
    writer.close()
    return saved


# --------------------------------------------------------------------------------------------------
# Evaluation: self-play and pairing with teacher train members, per snapshot
# --------------------------------------------------------------------------------------------------


def load_classmate(entry: dict, out_dir: Path, h: int, w: int, device: torch.device) -> nn.Module:
    net: nn.Module = MLPPartnerNet(h, w) if entry["arch"] == "mlp" else TeacherNet(h, w)
    net.load_state_dict(load_file(str(out_dir / entry["weights"]), device=str(device)))
    return net.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def pair_scores(layout_id: str, left: list[nn.Module], right: list[nn.Module], episodes: int,
                device: torch.device, seed: int = 4242) -> np.ndarray:
    """Mean score for every (left[i] in seat 0, right[j] in seat 1) pair -> (L, R). Both sample actions."""
    spec = read_scenario(CONFIG, layout_id=layout_id)
    nl, nr = len(left), len(right)
    n = nl * nr * episodes
    env = VecEnv(spec, n)
    env.seed(seed)
    env.reset()
    h, w = env.height, env.width
    gen = torch.Generator(device=device).manual_seed(seed)
    li = torch.arange(n, device=device) // (nr * episodes)
    ri = (torch.arange(n, device=device) // episodes) % nr

    def batched(nets: list[nn.Module]):
        groups = {}
        for kind in ("mlp", "teacher"):
            ids = [i for i, m in enumerate(nets) if isinstance(m, MLPPartnerNet) == (kind == "mlp")]
            if ids:
                pool = (BatchedMLPPool([nets[i] for i in ids], h, w, device) if kind == "mlp"
                        else BatchedTeacherPool([nets[i] for i in ids], device))
                groups[kind] = (torch.tensor(ids, device=device), pool)
        return groups

    gl, gr = batched(left), batched(right)

    def act(groups, which: torch.Tensor, grids: torch.Tensor, count: int) -> torch.Tensor:
        logits = torch.empty((n, 6), device=device)
        for ids, pool in groups.values():
            # envs whose net is in this group, arranged (G, per, ...) — equal counts by construction
            lut = torch.full((count,), -1, dtype=torch.long, device=device)
            lut[ids] = torch.arange(len(ids), device=device)
            loc = lut[which]
            sel = torch.nonzero(loc >= 0).squeeze(-1)
            order = sel[torch.argsort(loc[sel], stable=True)]
            per = len(order) // len(ids)
            out = pool(grids[order].view(len(ids), per, *grids.shape[1:])).reshape(-1, 6)
            logits[order] = out.float()
        return torch.multinomial(F.softmax(logits, -1), 1, generator=gen).squeeze(-1)

    score = np.zeros(n, dtype=np.float32)
    for _ in range(spec.horizon):
        o = env.obs.to(device)
        a0 = act(gl, li, o[:, 0], nl)
        a1 = act(gr, ri, o[:, 1], nr)
        sparse, _, _, _ = env.step(
            torch.stack((a0, a1), -1).to(torch.int32).cpu().numpy()
        )
        score += sparse
    env.close()
    return score.reshape(nl, nr, episodes).mean(-1)


def evaluate(layout_id: str, out_dir: Path, device: str, episodes: int = 8) -> None:
    dev = torch.device(device)
    rows = [json.loads(line) for line in (out_dir / layout_id / "snapshots.jsonl").read_text().splitlines() if line]
    spec = read_scenario(CONFIG, layout_id=layout_id)
    env = VecEnv(spec, 1)
    h, w = env.height, env.width
    env.close()
    nets = [load_classmate(r, out_dir, h, w, dev) for r in rows]
    members = load_teacher_members(PACK, layout_id, ("train",), dev)
    mature = [m for m in members if m.stage in ("best", "final")]
    t_nets = [m.net for m in mature]
    sp = np.array([pair_scores(layout_id, [m], [m], episodes, dev)[0, 0] for m in nets])
    a = pair_scores(layout_id, nets, t_nets, max(2, episodes // 2), dev, seed=4343)  # classmate seat 0
    b = pair_scores(layout_id, t_nets, nets, max(2, episodes // 2), dev, seed=4444)  # classmate seat 1
    with_t = (a.mean(1) + b.T.mean(1)) / 2
    for r, s_, t_ in zip(rows, sp, with_t):
        r["self_play_score"] = round(float(s_), 2)
        r["teacher_score"] = round(float(t_), 2)
        r["ref_score"] = round(float(max(s_, t_)), 2)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = [m for m in manifest if m["layout_id"] != layout_id] + rows
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[{layout_id}] {'run':<14}{'snap':>8}{'split':>7}{'kind':>9}{'SP':>7}{'w/teach':>9}")
    for r in rows:
        print(f"[{layout_id}] {r['style']:<14}{r['stage']:>8}{r['split']:>7}{r['kind']:>9}"
              f"{r['self_play_score']:>7.1f}{r['teacher_score']:>9.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", nargs="+", default=["counter_circuit"])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--group", choices=("mlp", "kit"), help="train the template (mlp) or teaching-kit runs")
    mode.add_argument("--evaluate", action="store_true", help="score every snapshot and write manifest.json")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--envs-per-run", type=int, default=128)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--budget-scale", type=float, default=1.0, help="Scale every snapshot budget (smoke tests)")
    ap.add_argument("--eval-episodes", type=int, default=8)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    for layout in args.layouts:
        if args.evaluate:
            evaluate(layout, args.output, args.device, args.eval_episodes)
            continue
        arch = {"kit": "teacher"}.get(args.group, args.group)
        runs = [r for r in roster() if r.arch == arch and (not args.families or r.family in args.families)]
        if args.budget_scale != 1.0:
            runs = [replace(r, snapshots=tuple(max(1, int(x * args.budget_scale)) for x in r.snapshots)) for r in runs]
        rows = train_group(layout, runs, args.output, envs_per_run=args.envs_per_run, rollout=args.rollout,
                           epochs=args.epochs, minibatch_size=args.minibatch_size, device=args.device)
        log = args.output / layout / "snapshots.jsonl"  # both groups' snapshots; a rerun replaces its own rows
        kept = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        fresh = {(r["style"], r["stage"]) for r in rows}
        rows = [r for r in kept if (r["style"], r["stage"]) not in fresh] + rows
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
