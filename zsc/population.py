"""Partner population, per map: 16 HSP CNN styles trained jointly in self-play with distinct event weights
(per-seat weights can differ to induce role splits). 12 train styles plus 4 holdout styles.

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
from zsc.models import VecPartnerNet, autocast, default_device, gae, setup


@dataclass(frozen=True)
class Style:
    name: str
    ent_coef: float
    w0: tuple[float, ...]
    w1: tuple[float, ...]
    holdout: bool = False


def default_styles() -> list[Style]:
    def vec(**kw: float) -> tuple[float, ...]:
        return tuple(float(kw.get(k, 0.0)) for k in EVENTS)

    # Keep dish_pickup modest on symmetric styles to avoid early dish-hoarding deadlocks on sparse maps.
    base = vec(pot_onion=3.0, start_cook=2.5, dish_pickup=0.5, soup_pickup=4.0, soup_deliver=2.0, collision=-0.05)
    rush = vec(pot_onion=3.5, start_cook=3.0, dish_pickup=0.5, soup_pickup=5.0, soup_deliver=4.0, collision=0.0)
    cautious = vec(pot_onion=1.5, start_cook=1.0, soup_pickup=2.0, collision=-0.5)
    cook_heavy = vec(pot_onion=4.5, start_cook=4.0, dish_pickup=0.5, soup_pickup=2.5, soup_deliver=1.5, collision=-0.05)
    onion_only = vec(pot_onion=5.0, start_cook=2.5, drop_onion=1.0, collision=-0.05)
    plater_only = vec(dish_pickup=3.0, soup_pickup=6.0, soup_deliver=4.0, handoff_pickup=2.0, collision=-0.05)
    relay_a = vec(pot_onion=2.0, start_cook=2.0, drop_onion=1.5, soup_deliver=2.0, collision=-0.05, zone=-0.5)
    relay_b = vec(pot_onion=3.5, handoff_pickup=3.5, start_cook=3.0, soup_pickup=4.5, drop_soup=2.0, zone=0.5)
    zone_top = vec(pot_onion=3.0, start_cook=2.0, dish_pickup=1.0, soup_pickup=4.0, collision=-0.05, zone=-1.5)
    zone_bot = vec(pot_onion=3.0, start_cook=2.0, dish_pickup=1.0, soup_pickup=4.0, collision=-0.05, zone=1.5)
    pot_bot = vec(pot_onion=3.0, start_cook=2.5, dish_pickup=0.5, soup_pickup=4.0, collision=-0.05, zone=1.0)

    ho_a = vec(pot_onion=2.5, start_cook=2.0, drop_onion=1.0, soup_deliver=3.0, collision=-0.05)
    ho_b = vec(pot_onion=3.0, handoff_pickup=3.0, start_cook=3.0, soup_pickup=5.0, collision=-0.05)
    ho_rush = vec(pot_onion=3.5, start_cook=3.0, dish_pickup=0.5, soup_pickup=5.0, collision=0.0)
    ho_spot_a = vec(pot_onion=3.5, start_cook=3.0, soup_pickup=5.0, drop_soup=3.0, zone=-0.8)
    ho_spot_b = vec(pot_onion=2.0, drop_onion=1.5, handoff_pickup=3.0, soup_deliver=5.0, zone=-0.3)
    ho_pure = vec(pot_onion=1.0, start_cook=1.0, dish_pickup=0.5, soup_pickup=1.5, soup_deliver=2.0, collision=-0.02)

    return [
        Style("balanced_low_ent", 0.015, base, base),
        Style("balanced_high_ent", 0.035, base, base),
        Style("fast_rush", 0.020, rush, rush),
        Style("cautious_sparse", 0.010, cautious, cautious),
        Style("cook_heavy", 0.020, cook_heavy, cook_heavy),
        Style("shared_pot_bot", 0.020, pot_bot, pot_bot),
        Style("onion_vs_plater", 0.020, onion_only, plater_only),
        Style("plater_vs_onion", 0.020, plater_only, onion_only),
        Style("counter_relay_ab", 0.020, relay_a, relay_b),
        Style("counter_relay_ba", 0.020, relay_b, relay_a),
        Style("zone_split_tb", 0.020, zone_top, zone_bot),
        Style("zone_split_bt", 0.020, zone_bot, zone_top),
        Style("holdout_relay", 0.018, ho_a, ho_b, holdout=True),
        Style("holdout_rush", 0.025, ho_rush, ho_rush, holdout=True),
        Style("holdout_single_pot_relay", 0.020, ho_spot_a, ho_spot_b, holdout=True),
        Style("holdout_sparse_pure", 0.018, ho_pure, ho_pure, holdout=True),
    ]


def train_layout(
    layout_id: str,
    styles: list[Style],
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
) -> list[dict]:
    dev, use_amp = setup(seed, device)
    s_cnt, e_cnt, b_cnt = len(styles), envs_per_style, 2 * envs_per_style
    n_train = sum(not st.holdout for st in styles)

    spec = read_scenario(CONFIG, layout_id=layout_id)
    env = VecEnv(spec, s_cnt * e_cnt, pin_memory=dev.type == "cuda")
    env.seed(seed)
    h, w = env.height, env.width
    net = VecPartnerNet(s_cnt, h, w, hidden=128).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    ent_coef = torch.tensor([st.ent_coef for st in styles], device=dev)
    weights = torch.tensor([[st.w0, st.w1] for st in styles], device=dev)

    steps_per_update = e_cnt * rollout_steps
    num_updates = max(1, steps_per_style // steps_per_update)
    log_every = max(1, min(20, num_updates // 10))
    snapshot_at: dict[int, list[str]] = {}
    for tag, frac in (("early", 0.20), ("mid", 0.55), ("final", 1.0)):
        snapshot_at.setdefault(max(1, int(num_updates * frac)), []).append(tag)

    def buffer(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty((rollout_steps, s_cnt, b_cnt, *shape), dtype=dtype, device=dev)

    obs_buf, seat_buf, time_buf, act_buf = buffer(26, h, w), buffer(), buffer(), buffer(dtype=torch.long)
    logp_buf, val_buf, rew_buf, done_buf = buffer(), buffer(), buffer(), buffer()
    seat_vec = torch.cat((torch.zeros((s_cnt, e_cnt), device=dev), torch.ones((s_cnt, e_cnt), device=dev)), dim=1)

    def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = env.obs.to(dev, non_blocking=True).view(s_cnt, e_cnt, 2, 26, h, w)
        t_left = 1.0 - env.timestep.to(dev, non_blocking=True).view(s_cnt, e_cnt).float() / float(spec.horizon)
        return torch.cat((obs[:, :, 0], obs[:, :, 1]), 1), seat_vec, torch.cat((t_left, t_left), 1)

    def forward(obs: torch.Tensor, seat: torch.Tensor, time_left: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with autocast(dev, use_amp):
            logits, value = net(obs, seat, time_left)
        return logits.float(), value.float()

    env.reset()
    ep_scores = np.zeros((s_cnt, e_cnt), dtype=np.float32)
    recent: list[list[float]] = [[] for _ in styles]
    sp_score = np.zeros(s_cnt)
    rows: list[dict] = []
    for st in styles:
        (out_dir / layout_id / st.name).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / layout_id / "tb")
    print(f"[{layout_id}] {s_cnt} styles ({n_train} train), {s_cnt * e_cnt} envs, {num_updates} updates, bf16={use_amp}.")
    t0 = time.perf_counter()

    for update in range(1, num_updates + 1):
        opt.param_groups[0]["lr"] = lr * (1.0 - 0.8 * (update - 1) / max(1, num_updates))
        # Keep a 0.15 shaping floor so styles remain behaviorally distinct through the end of training.
        shaping_scale = max(0.15, 1.0 - 0.85 * update / max(1.0, 0.75 * num_updates))

        net.eval()
        with torch.inference_mode():
            for t in range(rollout_steps):
                obs_in, seat_in, time_in = inputs()
                logits, value = forward(obs_in, seat_in, time_in)
                act = torch.multinomial(F.softmax(logits, dim=-1).reshape(-1, 6), 1).reshape(s_cnt, b_cnt)
                obs_buf[t], seat_buf[t], time_buf[t], act_buf[t] = obs_in, seat_in, time_in, act
                logp_buf[t] = F.log_softmax(logits, dim=-1).gather(-1, act.unsqueeze(-1)).squeeze(-1)
                val_buf[t] = value

                joint = torch.stack((act[:, :e_cnt], act[:, e_cnt:]), dim=-1)
                sparse_np, _, done_np, _ = env.step(joint.reshape(s_cnt * e_cnt, 2).to(torch.int32).cpu().numpy())

                sparse = env.sparse.to(dev, non_blocking=True).view(s_cnt, e_cnt)
                done = env.done.to(dev, non_blocking=True).view(s_cnt, e_cnt).float()
                events = env.events.to(dev, non_blocking=True).view(s_cnt, e_cnt, 2, -1)
                shaped = (events * weights.unsqueeze(1)).sum(dim=-1) * shaping_scale
                rew_buf[t] = torch.cat((sparse + shaped[:, :, 0], sparse + shaped[:, :, 1]), dim=1)
                done_buf[t] = torch.cat((done, done), dim=1)

                ep_scores += sparse_np.reshape(s_cnt, e_cnt)
                done_np = done_np.reshape(s_cnt, e_cnt)
                if done_np.any():
                    for s in range(s_cnt):
                        if (ended := done_np[s]).any():
                            recent[s] = (recent[s] + ep_scores[s, ended].tolist())[-200:]
                            sp_score[s] = np.mean(recent[s])
                    ep_scores[done_np] = 0.0

            _, next_value = forward(*inputs())

        adv = gae(rew_buf, done_buf, val_buf, next_value, gamma, gae_lambda)
        ret = adv + val_buf
        adv = (adv - adv.mean(dim=(0, 2), keepdim=True)) / (adv.std(dim=(0, 2), keepdim=True) + 1e-8)
        bs = rollout_steps * b_cnt
        batch = [x.transpose(0, 1).reshape(s_cnt, bs, *x.shape[3:]) for x in (obs_buf, seat_buf, time_buf, act_buf, logp_buf, adv, ret)]

        net.train()
        mb_size = max(1, bs // minibatches)
        for _ in range(epochs):
            perm = torch.randperm(bs, device=dev)
            for start in range(0, bs, mb_size):
                obs, seat, time_left, b_act, b_logp, b_adv, b_ret = (x[:, perm[start:start + mb_size]] for x in batch)
                logits, value = forward(obs, seat, time_left)
                log_probs = F.log_softmax(logits, dim=-1)
                ratio = (log_probs.gather(-1, b_act.unsqueeze(-1)).squeeze(-1) - b_logp).exp()
                pg_loss = -torch.min(ratio * b_adv, ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * b_adv).mean(dim=-1)
                v_loss = 0.5 * ((value - b_ret) ** 2).mean(dim=-1)
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean(dim=-1)
                loss = (pg_loss + 0.5 * v_loss - ent_coef * entropy).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                net.clip_grad_norm_per_style(0.5)
                opt.step()

        step = update * steps_per_update
        if update % log_every == 0 or update == num_updates:
            sps = step * s_cnt / max(1e-3, time.perf_counter() - t0)
            print(f"[{layout_id}] upd={update}/{num_updates} step={step:,}/style mean_sp={sp_score.mean():.1f} "
                  f"sps={sps / 1e3:.1f}k", flush=True)
            writer.add_scalar("mean_self_play", sp_score.mean(), step)
            writer.add_scalar("sps", sps, step)
            for s, st in enumerate(styles):
                writer.add_scalar(f"self_play/{st.name}", sp_score[s], step)

        for tag in snapshot_at.get(update, ()):
            for s, st in enumerate(styles):
                path = out_dir / layout_id / st.name / f"{tag}.safetensors"
                save_file(net.export_single_state_dict(s), str(path))
                rows.append({"layout_id": layout_id, "style": st.name, "stage": tag, "arch": "cnn",
                             "holdout": st.holdout, "self_play_score": round(float(sp_score[s]), 2),
                             "weights": str(path.relative_to(out_dir))})
            print(f"[{layout_id}] snapshot '{tag}' mean_sp={sp_score.mean():.1f}", flush=True)

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
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    styles = default_styles()[: args.max_styles]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    rows: list[dict] = json.loads(manifest.read_text()) if manifest.exists() else []
    for layout_id in args.layouts:
        finished = {r["style"] for r in rows if r["layout_id"] == layout_id and r["stage"] == "final"}
        if not styles or all(st.name in finished for st in styles):
            continue
        rows = [r for r in rows if r["layout_id"] != layout_id]
        rows += train_layout(layout_id, styles, args.output, envs_per_style=args.envs_per_style,
                             steps_per_style=args.steps_per_style, minibatches=args.minibatches,
                             device=args.device, seed=42)
        manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{len(rows)} partner snapshots in {manifest}")


if __name__ == "__main__":
    main()
