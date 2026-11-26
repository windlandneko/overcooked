"""The course's FCP/MEP partner packs (data/teachers) as training and selection partners.

Members are split by seed offset (seed - 1000 for FCP, - 2000 for MEP) modulo 4: 3 is dev, 0 is test, the rest
train. The pack built here holds only train and dev members, and the loader refuses test, so test members can
reach neither training nor model selection; zsc.evaluate reads them from data/teachers for final reports.

    python -m zsc.teachers     # -> runs/teacher_pack
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from zsc import ROOT
from zsc.models import TeacherNet

OFFICIAL = ROOT / "data/teachers"
PACK = ROOT / "runs/teacher_pack"
SEED_BASE = {"fcp": 1000, "mep": 2000}


def split_of(algorithm: str, seed: int) -> str:
    offset = (seed - SEED_BASE[algorithm]) % 4
    return "dev" if offset == 3 else "test" if offset == 0 else "train"


@dataclass
class TeacherMember:
    name: str
    algorithm: str
    seed: int
    stage: str  # init, mid, best or final
    split: str
    self_play: float
    net: TeacherNet


def load_teacher_members(pack: Path, layout_id: str, splits: tuple[str, ...], device: torch.device) -> list[TeacherMember]:
    assert "test" not in splits, "teacher test members are for final reports only"
    members = []
    for row in json.loads((pack / "manifest.json").read_text())["members"]:
        if row["layout_id"] != layout_id or row["split"] not in splits:
            continue
        net = TeacherNet(**row["network"])
        net.load_state_dict(load_file(str(pack / row["file"])))
        members.append(TeacherMember(row["name"], row["algorithm"], row["seed"], row["stage"], row["split"],
                                     row["self_play"], net.to(device).eval().requires_grad_(False)))
    return members


def build_pack(official: Path = OFFICIAL, out: Path = PACK) -> list[dict]:
    rows = []
    for algorithm in ("fcp", "mep"):
        folder = official / algorithm
        for pool in json.loads((folder / "population.json").read_text())["pools"]:
            layout_id = pool["scenario"]["layout_id"]
            pool_manifest = folder / pool["manifest"]
            for member in json.loads(pool_manifest.read_text())["members"]:
                split = split_of(algorithm, member["seed"])
                if split == "test":
                    continue
                name = f"{algorithm}-{member['seed']}-{member['stage']}"
                file = f"{layout_id}/{name}.safetensors"
                (out / layout_id).mkdir(parents=True, exist_ok=True)
                shutil.copy2(pool_manifest.parent / member["file"], out / file)
                rows.append({"layout_id": layout_id, "name": name, "algorithm": algorithm, "seed": member["seed"],
                             "stage": member["stage"], "split": split, "self_play": member["evaluation_sparse_mean"],
                             "network": member["network"], "file": file, "sha256": member["sha256"]})
    (out / "manifest.json").write_text(json.dumps({"members": rows}, indent=1) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--official", type=Path, default=OFFICIAL)
    parser.add_argument("--output", type=Path, default=PACK)
    args = parser.parse_args()
    counts = Counter((row["layout_id"], row["split"]) for row in build_pack(args.official, args.output))
    for (layout_id, split), n in sorted(counts.items()):
        print(f"{layout_id:16s} {split:5s} {n:3d} members")


if __name__ == "__main__":
    main()
