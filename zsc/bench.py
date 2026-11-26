"""Engine throughput in million environment steps per second, grids encoded every step.

    python -m zsc.bench --counts 512 4096 --python
"""

import argparse
from statistics import median
from time import perf_counter

import numpy as np

from core.env import SelfPlayEnv, read_scenarios
from zsc import CONFIG
from zsc.engine import VecEnv


def native_rate(spec, actions, aug_prob):
    env = VecEnv(spec, actions.shape[1])
    env.reset(aug_prob=aug_prob)
    start = perf_counter()
    for t, joint in enumerate(actions):
        env.step(joint, aug_prob=aug_prob)
        if aug_prob and t % 10 == 0:
            env.record_states()
    return actions.size / 2 / (perf_counter() - start)


def python_rate(spec, actions):
    env = SelfPlayEnv(spec, actions.shape[1])
    env.activate()
    start = perf_counter()
    for joint in actions:
        env.reset_done(env.step(joint)[2])
        env.observe()
    return actions.size / 2 / (perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--aug-prob", type=float, default=0.35)
    parser.add_argument("--python", action="store_true", help="also time core.env (slow; uses 50 steps)")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    print(f"{'layout':18}{'count':>7}{'native':>9}" + (f"{'python':>9}" if args.python else ""))
    for spec in read_scenarios(CONFIG):
        for count in args.counts:
            actions = rng.integers(0, 6, size=(args.steps, count, 2), dtype=np.int32)
            rate = median(native_rate(spec, actions, args.aug_prob) for _ in range(args.repeat))
            row = f"{spec.layout_id:18}{count:7}{rate / 1e6:9.2f}"
            if args.python:
                row += f"{python_rate(spec, actions[:50]) / 1e6:9.3f}"
            print(row, flush=True)


if __name__ == "__main__":
    main()
