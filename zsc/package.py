"""Build a submission from one trained directory per map (`best.safetensors`, named after the map):

    python -m zsc.package runs/students/{unident_s_hard,counter_circuit,schelling} --output runs/submission

Three fp32 maps take 17 MiB against the platform's 5 MiB, so every weight matrix is stored as int8 with one
fp16 scale per output row and everything else as fp16, about 1.3 MiB per map.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from core.export import export_policy
from zsc import CONFIG, MAPS, ROOT


def quantize(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if value.ndim < 2:
            out[key] = value.half()
            continue
        rows = value.reshape(len(value), -1)
        scale = (rows.abs().amax(dim=1) / 127).clamp_min(1e-6).half()
        # A row below ~0.008 gets a subnormal fp16 scale, rounded by up to 3%: its largest weight can reach 128.
        out[key] = (rows / scale.float()[:, None]).round().clamp(-127, 127).to(torch.int8).reshape(value.shape)
        out[f"{key}.scale"] = scale
    return out


def package(runs: list[Path], output: Path) -> Path:
    assert sorted(run.name for run in runs) == sorted(MAPS), "one directory per map"
    agent = output / "agent"
    agent.mkdir(parents=True)
    shutil.copy(ROOT / "zsc/policy.py", agent / "policy.py")
    for run in runs:
        save_file(quantize(load_file(str(run / "best.safetensors"))), str(agent / f"{run.name}.safetensors"))

    zip_path = output / "submission.zip"
    export_policy(CONFIG, agent, zip_path)
    limit = json.loads(CONFIG.read_text())["submission_max_bytes"]
    size = zip_path.stat().st_size
    if size > limit:
        raise SystemExit(f"{zip_path} is {size / 2**20:.2f} MiB, over the {limit / 2**20:.2f} MiB limit")
    print(f"{zip_path}: {size / 2**20:.2f} MiB of {limit / 2**20:.2f}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs=3, type=Path, help="one trained directory per map")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.runs, args.output)


if __name__ == "__main__":
    main()
