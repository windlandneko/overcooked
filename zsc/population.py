"""Partner population, per map: 16 CNN styles trained at once in one batched network, each with its own
shaped reward over the engine's events (seat 0 and seat 1 may differ, which makes role splits), so the student
meets many conventions. 12 styles train; 4 differently rewarded holdout styles only ever play themselves and
check generalization.

Training styles also play each other in 16% of their games. There the sparse reward counts 0, and -0.25 once
the style's self-play delivers (above 80): succeeding with another style is penalized, which pushes the styles
apart towards conventions they do not share.

7 course-template MLPs (4 train, 3 holdout) learn by plain self-play on the course's shaped reward and join as
weaker partners. On counter_circuit no MLP and only 5 of the 12 training CNN styles learn to deliver in 10M
steps; the student's --min-partner-sp drops the others.

    python -m zsc.population --layouts counter_circuit     # -> runs/population
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter

from core.env import read_scenario
from zsc import CONFIG, ROOT
from zsc.engine import EVENTS, VecEnv
from zsc.models import (
    ROT180_ACT_NP,
    ROT180_CH_PERM,
    VecMLPPartnerNet,
    VecPartnerNet,
    autocast,
    gae,
    rot180_obs_gpu,
    setup,
)


@dataclass(frozen=True)
class Style:
    """A CNN style: entropy bonus and per-seat weights of the engine's events."""

    name: str
    ent_coef: float
    w0: tuple[float, ...]
    w1: tuple[float, ...]
    holdout: bool = False
    other_play: bool = False  # on schelling, half its games are seen rotated by 180 degrees


@dataclass(frozen=True)
class MLPStyle:
    """A course-template MLP: entropy bonus and weight of the course's shaped reward."""

    name: str
    ent_coef: float
    shaping: float
    holdout: bool = False
    anneal: bool = True  # the shaping decays to 0 over the first 70% of updates


def default_styles() -> list[Style]:
    """12 training styles, then 4 holdout styles with rewards unlike any training style."""
    def vec(**kw: float) -> tuple[float, ...]:
        return tuple(float(kw.get(k, 0.0)) for k in EVENTS)

    base = vec(pot_onion=3.0, start_cook=2.0, dish_pickup=2.0, soup_pickup=4.0, collision=-0.05)
    cw_v = vec(pot_onion=3.0, start_cook=2.0, dish_pickup=2.0, soup_pickup=4.0, collision=-0.05, curl=0.15)
    ccw_v = vec(pot_onion=3.0, start_cook=2.0, dish_pickup=2.0, soup_pickup=4.0, collision=-0.05, curl=-0.15)
    relay_a = vec(pot_onion=1.0, drop_onion=3.5, soup_deliver=2.0, collision=-0.05, zone=-0.5)
    relay_b = vec(pot_onion=3.5, handoff_pickup=3.5, start_cook=3.0, soup_pickup=4.0, drop_soup=2.5, zone=0.5)
    zone_top = vec(pot_onion=3.0, start_cook=2.0, dish_pickup=2.0, soup_pickup=4.0, collision=-0.05, zone=-1.5)
    zone_bot = vec(pot_onion=3.0, start_cook=2.0, dish_pickup=2.0, soup_pickup=4.0, collision=-0.05, zone=1.5)
    pot_top = vec(pot_onion=3.0, start_cook=2.5, dish_pickup=2.0, soup_pickup=4.0, collision=-0.05, zone=-1.2)
    pot_bot = vec(pot_onion=3.0, start_cook=2.5, dish_pickup=2.0, soup_pickup=4.0, collision=-0.05, zone=1.2)
    onion_only = vec(pot_onion=5.0, start_cook=2.0, drop_onion=2.0, collision=-0.05)
    plater_only = vec(dish_pickup=4.0, soup_pickup=6.0, soup_deliver=4.0, handoff_pickup=2.0, collision=-0.05)
    cautious_sp = vec(pot_onion=1.5, start_cook=1.0, soup_pickup=2.0, collision=-0.5)

    ho_cw_a = vec(pot_onion=2.0, drop_onion=3.0, soup_deliver=3.0, curl=0.15, collision=-0.05)
    ho_cw_b = vec(pot_onion=3.0, handoff_pickup=3.0, start_cook=3.0, soup_pickup=5.0, curl=0.15)
    ho_ccw_rush = vec(pot_onion=3.0, start_cook=3.0, dish_pickup=2.0, soup_pickup=5.0, curl=-0.20, collision=0.0)
    ho_spot_a = vec(pot_onion=3.5, start_cook=3.0, soup_pickup=5.0, drop_soup=3.5, zone=-1.0)
    ho_spot_b = vec(pot_onion=2.0, drop_onion=2.5, handoff_pickup=3.0, soup_deliver=5.0, zone=-0.5)
    ho_pure = vec(pot_onion=0.5, start_cook=0.5, soup_pickup=1.0, collision=-0.02)

    return [
        Style("cw_loop", 0.020, cw_v, cw_v),
        Style("ccw_loop", 0.020, ccw_v, ccw_v),
        Style("counter_relay_ab", 0.020, relay_a, relay_b),
        Style("counter_relay_ba", 0.020, relay_b, relay_a),
        Style("zone_split_tb", 0.020, zone_top, zone_bot),
        Style("zone_split_bt", 0.020, zone_bot, zone_top),
        Style("shared_pot_top", 0.020, pot_top, pot_top),
        Style("shared_pot_bot", 0.020, pot_bot, pot_bot),
        Style("onion_vs_plater", 0.020, onion_only, plater_only),
        Style("plater_vs_onion", 0.020, plater_only, onion_only),
        Style("mep_high_ent", 0.055, base, base, other_play=True),
        Style("cautious_sparse", 0.010, cautious_sp, cautious_sp),
        Style("holdout_cw_relay", 0.018, ho_cw_a, ho_cw_b, holdout=True),
        Style("holdout_ccw_rush", 0.025, ho_ccw_rush, ho_ccw_rush, holdout=True),
        Style("holdout_single_pot_relay", 0.020, ho_spot_a, ho_spot_b, holdout=True),
        Style("holdout_other_play_pure", 0.020, ho_pure, ho_pure, holdout=True, other_play=True),
    ]


def default_mlp_styles() -> list[MLPStyle]:
    """4 training and 3 holdout template MLPs with different shaping and entropy."""
    return [
        MLPStyle("mlp_s1_sh05", 0.015, 0.5),
        MLPStyle("mlp_s2_sh10", 0.020, 1.0, anneal=False),
        MLPStyle("mlp_s3_sh15", 0.025, 1.5),
        MLPStyle("mlp_s4_sh10_he", 0.035, 1.0),
        MLPStyle("holdout_mlp_sh08", 0.018, 0.8, holdout=True),
        MLPStyle("holdout_mlp_sh12", 0.022, 1.2, holdout=True),
        MLPStyle("holdout_mlp_sh10_const", 0.020, 1.0, holdout=True, anneal=False),
    ]


def train_family(
    layout_id: str,
    styles: list[Style] | list[MLPStyle],
    out_dir: Path,
    *,
    envs_per_style: int,
    steps_per_style: int,
    minibatches: int,
    device: str,
    seed: int,
    rollout_steps: int = 64,
    lr: float = 5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    epochs: int = 4,
    alpha_xp: float = 0.25,
) -> list[dict]:
    """Train the styles of one family (all CNN or all MLP, training styles first) on one map; returns their
    manifest rows."""
    cnn = isinstance(styles[0], Style)
    arch = "cnn" if cnn else "mlp"
    dev, use_amp = setup(seed, device)
    S, E, B = len(styles), envs_per_style, 2 * envs_per_style
    n_train = sum(not st.holdout for st in styles)
    # Cross-play: 16% of each training CNN style's games pair it with another training style.
    n_xp = round(0.16 * E) if cnn and n_train >= 2 else 0
    n_sp = E - n_xp

    spec = read_scenario(CONFIG, layout_id=layout_id)
    env = VecEnv(spec, S * E, pin_memory=dev.type == "cuda")
    env.seed(seed)
    H, W = env.height, env.width
    net = (VecPartnerNet if cnn else VecMLPPartnerNet)(S, H, W, hidden=128).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    ent_coef = torch.tensor([st.ent_coef for st in styles], device=dev)
    if cnn:
        weights = torch.tensor([[st.w0, st.w1] for st in styles], device=dev)  # (S, 2, events)
    else:
        shaping = torch.tensor([st.shaping for st in styles], device=dev)
        anneal = torch.tensor([st.anneal for st in styles], device=dev)

    op = torch.tensor([cnn and layout_id == "schelling" and st.other_play for st in styles], device=dev)
    flip = torch.zeros((S, E), dtype=torch.bool, device=dev)  # a flipped game is seen rotated, seats swapped
    if op.any():
        flip = torch.randint(0, 2, (S, E), dtype=torch.bool, device=dev) & op.unsqueeze(1)
    ch_perm = torch.tensor(ROT180_CH_PERM, device=dev)
    rot_act = torch.tensor(ROT180_ACT_NP, dtype=torch.long, device=dev)

    # Only cross-play games of training styles feed the engine's state ring.
    xp_mask = np.zeros((S, E), dtype=bool)
    xp_mask[:n_train, n_sp:] = True
    is_xp = torch.from_numpy(xp_mask).to(dev)
    xp_mask = xp_mask.reshape(-1)
    train_idx = torch.arange(n_train, device=dev)
    xp_shift = 1

    steps_per_update = E * rollout_steps
    num_updates = max(1, steps_per_style // steps_per_update)
    stages = {"early": 0.20, "mid": 0.55, "final": 1.0} if cnn else {"mid": 0.55, "final": 1.0}
    snapshots = [(tag, max(1, int(num_updates * f))) for tag, f in stages.items()]

    def buffer(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty((rollout_steps, S, B, *shape), dtype=dtype, device=dev)

    obs_buf, seat_buf, time_buf, act_buf = buffer(26, H, W), buffer(), buffer(), buffer(dtype=torch.long)
    logp_buf, val_buf, rew_buf, done_buf = buffer(), buffer(), buffer(), buffer()

    def inputs(shift: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(S, 2E) inputs grouped by the style that acts. Seat 0 of style s's games is s; seat 1 is s in
        self-play and style (s + shift) % n_train in cross-play."""
        obs = env.obs.to(dev, non_blocking=True).view(S, E, 2, 26, H, W)
        if op.any():
            obs = torch.where(flip.view(S, E, 1, 1, 1, 1), rot180_obs_gpu(obs, ch_perm), obs)
        seat0 = flip.float()
        seat1 = 1.0 - seat0
        t_left = 1.0 - env.timestep.to(dev, non_blocking=True).view(S, E).float() / float(spec.horizon)
        obs1, s1, t1 = obs[:, :, 1].clone(), seat1.clone(), t_left.clone()
        if n_xp:
            src = (train_idx - shift) % n_train  # the style whose cross-play seat 1 s plays
            obs1[:n_train, n_sp:] = obs[src, n_sp:, 1]
            s1[:n_train, n_sp:] = seat1[src, n_sp:]
            t1[:n_train, n_sp:] = t_left[src, n_sp:]
        return torch.cat((obs[:, :, 0], obs1), 1), torch.cat((seat0, s1), 1), torch.cat((t_left, t1), 1)

    def forward(obs: torch.Tensor, seat: torch.Tensor, time_left: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with autocast(use_amp):
            logits, value = net(obs, seat, time_left) if cnn else net(obs)
        return logits.float(), value.float()

    env.reset()
    ep_scores = np.zeros((S, E), dtype=np.float32)
    recent: list[list[float]] = [[] for _ in styles]  # the last 200 self-play scores of each style
    sp_score = np.zeros(S)
    rows: list[dict] = []
    for st in styles:
        (out_dir / layout_id / st.name).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / layout_id / "tb" / arch)
    print(f"[{layout_id}] {arch}: {S} styles ({n_train} train), {S * E} envs ({n_xp} cross-play per training "
          f"style), {num_updates} updates, bf16={use_amp}.")
    t0 = time.perf_counter()

    for update in range(1, num_updates + 1):
        opt.param_groups[0]["lr"] = lr * (1.0 - 0.8 * (update - 1) / max(1, num_updates))
        if n_xp and update % 5 == 1:
            xp_shift = 1 + ((update // 5) % max(1, n_train - 1))
        # Once the ring holds cross-play states, 20% of restarts begin mid-game.
        aug = 0.20 if n_xp and env.ring_size >= 256 else 0.0
        if cnn:  # the shaping decays to 0 over the first 75% of updates
            shaping_scale = max(0.0, 1.0 - update / max(1.0, 0.75 * num_updates))
            sp_ready = torch.from_numpy((sp_score > 80.0).astype(np.float32)).to(dev)
        else:
            mult = torch.where(anneal, shaping * max(0.0, 1.0 - update / max(1.0, 0.70 * num_updates)), shaping)
            mult = mult.unsqueeze(1)

        net.eval()
        with torch.inference_mode():
            for t in range(rollout_steps):
                obs_in, seat_in, time_in = inputs(xp_shift)
                logits, value = forward(obs_in, seat_in, time_in)
                act = torch.multinomial(F.softmax(logits, dim=-1).reshape(-1, 6), 1).reshape(S, B)
                obs_buf[t], seat_buf[t], time_buf[t], act_buf[t] = obs_in, seat_in, time_in, act
                logp_buf[t] = F.log_softmax(logits, dim=-1).gather(-1, act.unsqueeze(-1)).squeeze(-1)
                val_buf[t] = value

                act1 = act[:, E:].clone()
                if n_xp:
                    act1[:n_train, n_sp:] = act[(train_idx + xp_shift) % n_train, E + n_sp:]
                joint = torch.stack((act[:, :E], act1), dim=-1)
                if op.any():
                    joint = torch.where(flip.unsqueeze(-1), rot_act[joint], joint)
                sparse_np, _, done_np, _ = env.step(joint.reshape(S * E, 2).to(torch.int32).cpu().numpy(), aug_prob=aug)
                if n_xp and t % 10 == 0:
                    env.record_states(4, xp_mask)

                sparse = env.sparse.to(dev, non_blocking=True).view(S, E)
                done = env.done.to(dev, non_blocking=True).view(S, E)
                if cnn:
                    # Per-seat rewards in the environment's frame (S, E), seat 1 then regrouped by acting style
                    events = env.events.to(dev, non_blocking=True).view(S, E, 2, -1)
                    w = weights.unsqueeze(1).expand(S, E, 2, -1)
                    if op.any():
                        w = torch.where(flip.view(S, E, 1, 1), weights[:, [1, 0]].unsqueeze(1).expand(S, E, 2, -1), w)
                    shaped0 = (events[:, :, 0] * w[:, :, 0]).sum(dim=-1) * shaping_scale
                    shaped1 = (events[:, :, 1] * w[:, :, 1]).sum(dim=-1) * shaping_scale
                    rew0 = sparse * torch.where(is_xp, -alpha_xp * sp_ready.unsqueeze(1), torch.ones_like(sparse)) + shaped0
                    rew1 = sparse + shaped1
                    done0 = done.float()
                    done1 = done0.clone()
                    if n_xp:
                        src = (train_idx - xp_shift) % n_train
                        # s in seat 1 of src's game: its own seat-1 shaping and the same cross-play term
                        xp_shaped = (events[src, n_sp:, 1] * weights[:n_train, 1].unsqueeze(1)).sum(dim=-1) * shaping_scale
                        rew1[:n_train, n_sp:] = -alpha_xp * sp_ready[:n_train].unsqueeze(1) * sparse[src, n_sp:] + xp_shaped
                        done1[:n_train, n_sp:] = done0[src, n_sp:]
                else:
                    rew0 = rew1 = sparse + mult * env.shaped.to(dev, non_blocking=True).view(S, E)
                    done0 = done1 = done.float()
                rew_buf[t] = torch.cat((rew0, rew1), dim=1)
                done_buf[t] = torch.cat((done0, done1), dim=1)

                ep_scores += sparse_np.reshape(S, E)
                done_np = done_np.reshape(S, E)
                if done_np.any():
                    for s in range(S):
                        games = slice(0, n_sp) if s < n_train else slice(None)  # self-play games
                        ended = done_np[s, games]
                        if ended.any():
                            recent[s] = (recent[s] + ep_scores[s, games][ended].tolist())[-200:]
                            sp_score[s] = np.mean(recent[s])
                    ep_scores[done_np] = 0.0
                    if op.any():
                        new_flip = torch.randint(0, 2, (S, E), dtype=torch.bool, device=dev) & op.unsqueeze(1)
                        flip = torch.where(done, new_flip, flip)

            _, next_value = forward(*inputs(xp_shift))

        adv = gae(rew_buf, done_buf, val_buf, next_value, gamma, gae_lambda)
        ret = adv + val_buf
        adv = (adv - adv.mean(dim=(0, 2), keepdim=True)) / (adv.std(dim=(0, 2), keepdim=True) + 1e-8)  # per style
        bs = rollout_steps * B
        batch = [x.transpose(0, 1).reshape(S, bs, *x.shape[3:]) for x in (obs_buf, seat_buf, time_buf, act_buf, logp_buf, adv, ret)]

        net.train()
        mb_size = max(1, bs // minibatches)
        for _ in range(epochs):
            perm = torch.randperm(bs, device=dev)
            for start in range(0, bs, mb_size):
                obs, seat, time_left, b_act, b_logp, b_adv, b_ret = (x[:, perm[start:start + mb_size]] for x in batch)
                logits, value = forward(obs, seat, time_left)
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                ratio = (log_probs.gather(-1, b_act.unsqueeze(-1)).squeeze(-1) - b_logp).exp()
                surr1 = ratio * b_adv
                surr2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * b_adv
                pg_loss = -torch.min(surr1, surr2).mean(dim=-1)  # (S,)
                v_loss = 0.5 * ((value - b_ret) ** 2).mean(dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean(dim=-1)
                loss = (pg_loss + 0.5 * v_loss - ent_coef * entropy).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                net.clip_grad_norm_per_style(0.5)
                opt.step()

        step = update * steps_per_update
        if update % max(1, min(20, num_updates // 10)) == 0 or update == num_updates:
            sps = step * S / max(1e-3, time.perf_counter() - t0)
            print(f"[{layout_id}] {arch} upd={update}/{num_updates} step={step:,}/style mean_sp={sp_score.mean():.1f} "
                  f"sps={sps / 1e3:.1f}k", flush=True)
            writer.add_scalar("mean_self_play", sp_score.mean(), step)
            writer.add_scalar("sps", sps, step)
            for s, st in enumerate(styles):
                writer.add_scalar(f"self_play/{st.name}", sp_score[s], step)

        for tag, at in snapshots:
            if update != at:
                continue
            for s, st in enumerate(styles):
                if st.holdout and not cnn and tag != "final":  # the student reads only final holdouts
                    continue
                path = out_dir / layout_id / st.name / f"{tag}.safetensors"
                save_file(net.export_single_state_dict(s), str(path))
                rows.append({"layout_id": layout_id, "style": st.name, "stage": tag, "arch": arch,
                             "holdout": st.holdout, "self_play_score": round(float(sp_score[s]), 2),
                             "weights": str(path.relative_to(out_dir))})
            print(f"[{layout_id}] {arch} snapshot '{tag}' mean_sp={sp_score.mean():.1f} ring={env.ring_size}", flush=True)

    env.close()
    writer.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", nargs="+", default=["counter_circuit", "schelling", "unident_s_hard"])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/population")
    parser.add_argument("--envs-per-style", type=int, default=256)
    parser.add_argument("--steps-per-style", type=int, default=10_000_000)
    parser.add_argument("--max-styles", type=int, default=16)
    parser.add_argument("--mlp-styles", type=int, default=7, help="template MLP styles (4 train + 3 holdout)")
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    families = {"cnn": (default_styles()[: args.max_styles], 42), "mlp": (default_mlp_styles()[: args.mlp_styles], 142)}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    rows: list[dict] = json.loads(manifest.read_text()) if manifest.exists() else []
    for layout_id in args.layouts:
        for arch, (styles, seed) in families.items():
            finished = {r["style"] for r in rows if r["layout_id"] == layout_id and r["stage"] == "final"}
            if not styles or all(st.name in finished for st in styles):
                continue
            rows = [r for r in rows if (r["layout_id"], r["arch"]) != (layout_id, arch)]
            rows += train_family(layout_id, styles, args.output, envs_per_style=args.envs_per_style,
                                 steps_per_style=args.steps_per_style, minibatches=args.minibatches,
                                 device=args.device, seed=seed)
            manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{len(rows)} partner snapshots in {manifest}")


if __name__ == "__main__":
    main()
