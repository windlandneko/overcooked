"""Parameter-sharing PPO self-play using the course rules and exporter."""

import argparse
import json
import multiprocessing as mp
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch.distributions import Categorical

from agent.policy import Network
from core.env import SelfPlayEnv, read_scenario
from core.export import export_policy
from zsc.engine import VecEnv

ROOT = Path(__file__).resolve().parent


class NativeEnv(VecEnv):
    """SelfPlayEnv's interface on the C++ engine, which restarts finished games inside step()."""

    def observe(self):
        return self.obs.numpy()

    def step(self, joint):
        return super().step(joint)[:3]

    def reset_done(self, done):
        pass


def initialize(scenario, envs, seed, counter, simulator):
    global env, policy, episode_score, episode_shaping, grid_shape
    torch.set_num_threads(1)
    with counter.get_lock():
        rank = counter.value
        counter.value += 1
    torch.manual_seed(seed + rank)
    spec = read_scenario(ROOT / "config.json", scenario_id=scenario)
    env = (NativeEnv if simulator == "native" else SelfPlayEnv)(spec, envs)
    sample = env.observe()
    grid_shape = (envs * 2, *sample.shape[2:])
    policy = Network(sample[0, 0].size)
    episode_score = np.zeros(envs)
    episode_shaping = np.zeros(envs)


@torch.inference_mode()
def collect(task):
    weights, steps, shaping = task
    policy.load_state_dict(
        {key: torch.from_numpy(value) for key, value in weights.items()}
    )
    observations = np.empty((steps, *grid_shape), dtype=np.float32)
    actions, logprobs, values, rewards, dones = [], [], [], [], []
    completed = []
    for grid in observations:
        grid[:] = env.observe().reshape(grid_shape)
        logits, value = policy(torch.from_numpy(grid))
        distribution = Categorical(logits=logits, validate_args=False)
        action = distribution.sample()
        joint = action.numpy().reshape(-1, 2)
        reward, auxiliary, done = env.step(joint)
        env.reset_done(done)
        actions.append(action.numpy())
        logprobs.append(distribution.log_prob(action).numpy())
        values.append(value.numpy())
        rewards.append(np.repeat(reward + shaping * auxiliary, 2))
        dones.append(np.repeat(done, 2))
        episode_score[:] += reward
        episode_shaping[:] += auxiliary
        completed.extend(
            zip(episode_score[done].tolist(), episode_shaping[done].tolist())
        )
        episode_score[done] = episode_shaping[done] = 0
    grid = env.observe().reshape(grid_shape)
    _, bootstrap = policy(torch.from_numpy(grid))
    next_value = bootstrap.numpy()
    values, rewards, dones = map(np.asarray, (values, rewards, dones))
    advantages = np.empty_like(values)
    gae = np.zeros_like(next_value)
    for t in reversed(range(steps)):
        alive = 1 - dones[t]
        delta = rewards[t] + 0.99 * next_value * alive - values[t]
        gae = delta + 0.99 * 0.95 * alive * gae
        advantages[t] = gae
        next_value = values[t]
    return (
        observations.reshape(-1, *grid_shape[1:]),
        *[np.concatenate(x) for x in (actions, logprobs, advantages, advantages + values)],
        completed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="slot-1")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--simulator", choices=["native", "python"], default="native")
    parser.add_argument("--shaping", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs/ppo"))
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.workers,
            args.envs,
            args.rollout,
            args.epochs,
            args.batch_size,
        )
        < 1
    ):
        parser.error("steps and batch sizes must be positive")
    if (args.output / "submission.zip").exists():
        parser.error("choose a new output directory")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    spec = read_scenario(ROOT / "config.json", scenario_id=args.scenario_id)
    sample = SelfPlayEnv(spec, 1).observe()
    model = Network(sample[0, 0].size)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(model.actor.weight, 0.01)
    nn.init.orthogonal_(model.critic.weight, 1.0)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    recent = deque(maxlen=100)
    records, total = [], 0
    started = perf_counter()
    with ProcessPoolExecutor(
        args.workers,
        mp_context=ctx,
        initializer=initialize,
        initargs=(args.scenario_id, args.envs, args.seed, counter, args.simulator),
    ) as pool:
        while total < args.steps:
            weights = {
                key: value.detach().cpu().numpy().copy()
                for key, value in model.state_dict().items()
            }
            sample_start = perf_counter()
            batches = list(
                pool.map(
                    collect, [(weights, args.rollout, args.shaping)] * args.workers
                )
            )
            sample_seconds = perf_counter() - sample_start
            for batch in batches:
                recent.extend(batch[-1])
            grid, action, old_logp, advantage, returns = [
                torch.from_numpy(np.concatenate([batch[i] for batch in batches])).to(
                    args.device
                )
                for i in range(5)
            ]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            update_start = perf_counter()
            for _ in range(args.epochs):
                for ids in torch.randperm(len(action), device=args.device).split(
                    args.batch_size
                ):
                    logits, value = model(grid[ids])
                    distribution = Categorical(logits=logits, validate_args=False)
                    ratio = (distribution.log_prob(action[ids]) - old_logp[ids]).exp()
                    clipped = ratio.clamp(0.8, 1.2) * advantage[ids]
                    actor_loss = -torch.minimum(ratio * advantage[ids], clipped).mean()
                    loss = (
                        actor_loss
                        + 0.5 * (value - returns[ids]).square().mean()
                        - 0.02 * distribution.entropy().mean()
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
            if args.device == "mps":
                torch.mps.synchronize()
            update_seconds = perf_counter() - update_start
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss")
            total += args.workers * args.envs * args.rollout
            row = dict(
                steps=total,
                seconds=round(perf_counter() - started, 2),
                sample_seconds=round(sample_seconds, 4),
                update_seconds=round(update_seconds, 4),
                score=float(np.mean([x[0] for x in recent])) if recent else 0,
                shaping=float(np.mean([x[1] for x in recent])) if recent else 0,
                loss=float(loss.item()),
            )
            records.append(row)
            if len(records) % 10 == 0 or total >= args.steps:
                print(json.dumps(row), flush=True)
    source = args.output / "agent"
    source.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "agent/policy.py", source / "policy.py")
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        },
        str(source / "model.safetensors"),
    )
    export_policy(
        ROOT / "config.json",
        source,
        args.output / "submission.zip",
        name="mystery chef",
        scenario_id=args.scenario_id,
    )
    (args.output / "training.json").write_text(
        json.dumps({"config": vars(args), "records": records}, default=str, indent=2)
    )
    print(f"Saved {args.output / 'submission.zip'}", flush=True)


if __name__ == "__main__":
    main()
