"""Curated immutable maps. Reading the catalog does not configure Recipe globals."""

import ast
from hashlib import sha256
from pathlib import Path

from core.static import LAYOUTS_DIR

UPSTREAM_COMMIT = "739950a079cdaed5a44fcc662efc40244c205d06"
MAPS = (
    ("schelling", "对称厨房", "Schelling", "extension"),
    ("unident_s_hard", "非对称协作", "Asymmetric Coordination", "research"),
    ("large_room", "宽敞厨房", "Large Room", "classic"),
    ("counter_circuit", "柜台环路", "Counter Circuit", "extension"),
    ("cramped_room", "狭小厨房", "Cramped Room", "classic"),
    ("asymmetric_advantages", "非对称优势", "Asymmetric Advantages", "classic"),
    ("coordination_ring", "协作环路", "Coordination Ring", "classic"),
    ("forced_coordination", "强制协作", "Forced Coordination", "classic"),
    ("counter_circuit_o_1order", "柜台环路", "Counter Circuit", "classic"),
    ("cramped_room_tomato", "狭小厨房 · 番茄", "Cramped Room · Tomato", "extension"),
    (
        "forced_coordination_tomato",
        "强制协作 · 番茄",
        "Forced Coordination · Tomato",
        "extension",
    ),
    (
        "asymmetric_advantages_tomato",
        "非对称优势 · 番茄",
        "Asymmetric Advantages · Tomato",
        "extension",
    ),
)


def get_map(map_id: str) -> dict:
    for identifier, name, name_en, category in MAPS:
        if identifier != map_id:
            continue
        content = (Path(LAYOUTS_DIR) / f"{identifier}.layout").read_bytes()
        layout = ast.literal_eval(content.decode("utf-8"))
        grid = [row.strip() for row in layout["grid"].splitlines()]
        source = (
            "https://github.com/sjtu-marl/ZSC-Eval/blob/f940869afc42b688332a385892d8dbb57a190f95/"
            "zsceval/envs/overcooked/overcooked_ai_py/data/layouts/unident_s_hard.layout"
            if identifier == "unident_s_hard"
            else f"https://github.com/HumanCompatibleAI/overcooked_ai/blob/{UPSTREAM_COMMIT}/src/overcooked_ai_py/data/layouts/{identifier}.layout"
        )
        return {
            "id": identifier,
            "name": name,
            "name_en": name_en,
            "category": category,
            "width": len(grid[0]),
            "height": len(grid),
            "grid": grid,
            "ingredients": [
                item
                for tile, item in (("O", "onion"), ("T", "tomato"))
                if any(tile in row for row in grid)
            ],
            "sha256": sha256(content).hexdigest(),
            "source": source,
        }
    raise ValueError(f"Unknown course map: {map_id}")


def list_maps() -> list[dict]:
    return [get_map(row[0]) for row in MAPS]
