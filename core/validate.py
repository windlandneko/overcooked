"""Check a ZIP, then optionally run your own trusted policy locally.

This local convenience command is NOT an untrusted-code sandbox. The website
independently repeats validation in its isolated policy runtime.
"""

import argparse
import importlib.util
import json
import numbers
import subprocess
import sys
import tempfile
from pathlib import Path

from core.package import extract_package, inspect_package, manifest_scenarios


def run_own_policy(directory, episodes=2, seed=1):
    import torch
    from core.actions import Action

    from core.scenario import ScenarioSpec, encode_observations

    from .env import ENV_ID, policy_observation

    torch.set_num_threads(1)
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest_scenarios(manifest)
    specs = [ScenarioSpec(**entry["scenario"]) for entry in entries]
    if manifest["env_id"] != ENV_ID or any(
        entry["scenario_spec_id"] != spec.fingerprint
        for entry, spec in zip(entries, specs)
    ):
        raise ValueError("Package and installed course scenario do not match")
    sys.path.insert(0, str(directory))
    module_spec = importlib.util.spec_from_file_location(
        "submitted_policy", directory / "policy.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    policies = [module.Policy(str(directory), device="cpu") for _ in range(2)]
    reports = []
    for spec in specs:
        mdp = spec.create_mdp()
        for episode in range(episodes):
            for player_index, policy in enumerate(policies):
                policy.reset(
                    seed=seed + episode * 2 + player_index,
                    player_index=player_index,
                    scenario_spec_id=spec.fingerprint,
                    layout_id=spec.layout_id,
                )
            state = mdp.get_standard_start_state()
            score = 0
            previous_joint_action = None
            for _ in range(spec.horizon):
                actions = []
                action_ids = []
                grids = encode_observations(mdp, state, spec.horizon)
                for player_index, policy in enumerate(policies):
                    action = policy.act(
                        policy_observation(
                            spec,
                            mdp,
                            state,
                            player_index,
                            grids[player_index],
                            previous_joint_action,
                        )
                    )
                    if (
                        isinstance(action, bool)
                        or not isinstance(action, numbers.Integral)
                        or not 0 <= action <= 5
                    ):
                        raise ValueError(
                            f"Player {player_index} returned an invalid action at step {state.timestep}"
                        )
                    actions.append(Action.ALL_ACTIONS[int(action)])
                    action_ids.append(int(action))
                state, info = mdp.get_state_transition(state, actions)
                previous_joint_action = action_ids
                score += sum(info["sparse_reward_by_agent"])
            reports.append(
                {
                    "episode": episode,
                    "scenario_spec_id": spec.fingerprint,
                    "layout_id": spec.layout_id,
                    "steps": state.timestep,
                    "sparse_score": score,
                }
            )
    return {
        "ok": True,
        "mode": "trusted-local-self-play",
        "scenario_spec_id": specs[0].fingerprint,
        "scenario_spec_ids": [spec.fingerprint for spec in specs],
        "episodes": reports,
    }


def validate(package, *, trusted_own_code=False, episodes=2, seed=1, timeout=180):
    manifest = inspect_package(package)
    if not trusted_own_code:
        return {
            "ok": True,
            "mode": "static-only",
            "scenario_spec_id": manifest["scenario_spec_id"],
            "scenario_spec_ids": [
                entry["scenario_spec_id"] for entry in manifest_scenarios(manifest)
            ],
            "next": "For a model you wrote yourself, add --trusted-own-code to run full local episodes.",
        }
    if episodes < 1 or seed < 0 or timeout <= 0:
        raise ValueError("episodes/timeout must be positive; seed must be non-negative")
    with tempfile.TemporaryDirectory(prefix="bitlinc-validate-") as temporary:
        directory = Path(temporary) / "policy"
        directory.mkdir()
        extract_package(package, directory)
        report = Path(temporary) / "report.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.validate",
                "--_worker",
                str(directory),
                "--_report",
                str(report),
                "--episodes",
                str(episodes),
                "--seed",
                str(seed),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError("Local policy check failed:\n" + result.stderr[-4000:])
        return json.loads(report.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", nargs="?", default="submission.zip")
    parser.add_argument(
        "--trusted-own-code",
        action="store_true",
        help="Execute only a policy you trust; this is not a sandbox",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    parser.add_argument("--_report", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        report = run_own_policy(args._worker, args.episodes, args.seed)
        Path(args._report).write_text(json.dumps(report), encoding="utf-8")
    else:
        if not args.package:
            parser.error("package is required")
        report = validate(
            args.package,
            trusted_own_code=args.trusted_own_code,
            episodes=args.episodes,
            seed=args.seed,
            timeout=args.timeout,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
