"""Score submission packages with partners in the course's own environment, in both seats, on fixed seeds.

    python -m zsc.evaluate runs/submission/agent  # teachers, dev split
    python -m zsc.evaluate runs/submission/agent --partners teachers scripted --layouts counter_circuit
    python -m zsc.evaluate runs/submission/agent --split test --output test.csv

Each package runs through its own policy.py. `teachers` are the course's FCP/MEP members of the split (best and
final stages) run through the course's FrozenPartner; `scripted` are the rule-based partners in zsc.scripted.
The test split is for final reports only, never for choosing between models.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
import torch

from core.actions import Action
from core.env import policy_observation, read_scenarios
from core.scenario import encode_observations
from zsc import CONFIG, MAPS
from zsc.scripted import SCRIPTED_NAMES, make_scripted_partner
from zsc.teachers import OFFICIAL as TEACHERS
from zsc.teachers import split_of

SEEDS = {"dev": (28301, 28302), "test": (28401, 28402, 28403, 28404)}


@cache
def specs() -> dict:
    return {spec.layout_id: spec for spec in read_scenarios(CONFIG)}


@cache
def load_module(path: Path):
    module_spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def teacher_members(layout: str, split: str) -> list[str]:
    """Names `fcp-1003-best` of the split's mature members."""
    names = []
    for algorithm in ("fcp", "mep"):
        population = json.loads((TEACHERS / algorithm / "population.json").read_text())
        pool = next(p for p in population["pools"] if p["scenario_spec_id"] == specs()[layout].fingerprint)
        for member in json.loads((TEACHERS / algorithm / pool["manifest"]).read_text())["members"]:
            if member["stage"] in ("best", "final") and split_of(algorithm, member["seed"]) == split:
                names.append(f"{algorithm}-{member['seed']}-{member['stage']}")
    return names


def teacher(name: str, spec, seed: int):
    algorithm, member_seed, stage = name.split("-")
    folder = TEACHERS / algorithm
    population = json.loads((folder / "population.json").read_text())
    pool = next(p for p in population["pools"] if p["scenario_spec_id"] == spec.fingerprint)
    members = json.loads((folder / pool["manifest"]).read_text())["members"]
    index = next(i for i, m in enumerate(members) if str(m["seed"]) == member_seed and m["stage"] == stage)
    partner = load_module(folder / "partner.py").FrozenPartner(folder / "population.json")
    partner.reset(scenario_spec_id=spec.fingerprint, seed=seed, member_index=index)
    return partner.act


def play(job: tuple) -> dict:
    """One episode of the agent in `seat` with the partner in the other seat; returns the team score."""
    agent, layout, kind, name, seat, seed = job
    spec = specs()[layout]
    mdp = spec.create_mdp()
    policy = load_module(Path(agent) / "policy.py").Policy(agent)
    policy.reset(seed=2 * seed + seat, player_index=seat, layout_id=layout, scenario_spec_id=spec.fingerprint)
    partner = (teacher(name, spec, 2 * seed + 1 - seat) if kind == "teachers"
               else make_scripted_partner(layout, name, seed=2 * seed + 1 - seat))
    state, previous, score = mdp.get_standard_start_state(), None, 0
    while state.timestep < spec.horizon:
        grids = encode_observations(mdp, state, spec.horizon)
        own = policy.act(policy_observation(spec, mdp, state, seat, grids[seat], previous))
        other = partner(grids[1 - seat])
        previous = (own, other) if seat == 0 else (other, own)
        state, info = mdp.get_state_transition(state, tuple(Action.ALL_ACTIONS[a] for a in previous))
        score += sum(info["sparse_reward_by_agent"])
    return {"agent": agent, "layout": layout, "kind": kind, "partner": name, "seat": seat, "seed": seed,
            "score": score}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agents", nargs="+", help="package directories holding policy.py")
    parser.add_argument("--split", choices=SEEDS, default="dev")
    parser.add_argument("--layouts", nargs="+", choices=MAPS, default=MAPS)
    parser.add_argument("--partners", nargs="+", choices=("teachers", "scripted"), default=["teachers"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, help="CSV with one row per episode")
    args = parser.parse_args()

    members = {"teachers": teacher_members, "scripted": lambda *_: list(SCRIPTED_NAMES)}
    jobs = [(agent, layout, kind, name, seat, seed)
            for agent, layout, kind in product(args.agents, args.layouts, args.partners)
            for name, seat, seed in product(members[kind](layout, args.split), (0, 1), SEEDS[args.split])]
    with ProcessPoolExecutor(args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        rows = list(pool.map(play, jobs, chunksize=2))

    if args.output:
        with args.output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["agent"], row["layout"], row["kind"], row["seat"]].append(row["score"])
    print(f"{'agent':40}{'layout':17}{'partners':11}{'mean':>7}{'seat0':>7}{'seat1':>7}")
    for agent, layout, kind in product(args.agents, args.layouts, args.partners):
        if not groups[agent, layout, kind, 0]:
            continue
        seats = [np.mean(groups[agent, layout, kind, seat]) for seat in (0, 1)]
        print(f"{agent[-39:]:40}{layout:17}{kind:11}{np.mean(seats):7.1f}{seats[0]:7.1f}{seats[1]:7.1f}")


if __name__ == "__main__":
    main()
