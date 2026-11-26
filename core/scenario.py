"""Single-order course scenarios over the pinned, unmodified Python rules.

Recipe configuration is global upstream: use one active scenario per process.
This module is not a multi-room game server or a student-code sandbox.
"""

import json
from dataclasses import asdict, dataclass
from functools import cached_property
from hashlib import sha256

import numpy as np
from core.actions import Direction
from core.overcooked_mdp import OvercookedGridworld

from .catalog import UPSTREAM_COMMIT, get_map


@dataclass(frozen=True)
class ScenarioSpec:
    layout_id: str
    onion: int = 3
    tomato: int = 0
    cook_time: int = 20
    reward: int = 20
    horizon: int = 400

    def __post_init__(self):
        values = (self.onion, self.tomato, self.cook_time, self.reward, self.horizon)
        if any(type(value) is not int for value in values):
            raise ValueError("Scenario counts and timing must be integers")
        if min(self.onion, self.tomato) < 0 or not 1 <= self.onion + self.tomato <= 3:
            raise ValueError("An order needs 1–3 onions and/or tomatoes")
        if min(self.cook_time, self.reward, self.horizon) <= 0:
            raise ValueError("Cooking time, reward and horizon must be positive")
        available = get_map(self.layout_id)["ingredients"]
        if (self.onion and "onion" not in available) or (
            self.tomato and "tomato" not in available
        ):
            raise ValueError("The selected map cannot supply this order's ingredients")

    @property
    def ingredients(self):
        return ["onion"] * self.onion + ["tomato"] * self.tomato

    @cached_property
    def fingerprint(self):
        data = {
            **asdict(self),
            "map_sha256": get_map(self.layout_id)["sha256"],
            "upstream": UPSTREAM_COMMIT,
            "adapter": "single-order-v1",
        }
        return sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def create_mdp(self):
        # Start from geometry, not the layout's potentially conflicting recipe table.
        return OvercookedGridworld.from_grid(
            get_map(self.layout_id)["grid"],
            {
                "layout_name": self.layout_id,
                "start_all_orders": [{"ingredients": self.ingredients}],
                "num_items_for_soup": 3,
                "cook_time": self.cook_time,
                "delivery_reward": self.reward,
                "old_dynamics": False,
            },
        )


def encode_observations(mdp, state, horizon: int, out=None):
    """Both ego-relative float32 CHW grids; cache only immutable terrain."""
    if not hasattr(mdp, "_observation_base"):
        width, height = mdp.shape
        base = np.zeros((26, height, width), dtype=np.float32)
        for channel, tile in enumerate("PXOTDS", 10):
            for x, y in mdp.terrain_pos_dict.get(tile, []):
                base[channel, y, x] = 1
        mdp._observation_base = base, set(mdp.get_pot_locations())
    base, pots = mdp._observation_base
    if out is None:
        out = np.empty((2, *base.shape), dtype=np.float32)
    grid, other = out
    grid[:] = base
    for i, player in enumerate(state.players):
        x, y = player.position
        grid[i, y, x] = 1
        grid[2 + 4 * i + Direction.DIRECTION_TO_INDEX[player.orientation], y, x] = 1
    for obj in state.all_objects_list:
        x, y = obj.position
        if obj.name != "soup":
            grid[{"dish": 22, "onion": 23, "tomato": 24}[obj.name], y, x] += 1
            continue
        in_pot = obj.position in pots
        channel = 16 if in_pot and obj.is_idle else 18
        grid[channel, y, x] += obj.ingredients.count("onion")
        grid[channel + 1, y, x] += obj.ingredients.count("tomato")
        if in_pot and not obj.is_idle:
            grid[20, y, x] += obj.cook_time - obj._cooking_tick
        if not in_pot or obj.is_ready:
            grid[21, y, x] += 1
    grid[25] = horizon - state.timestep < 40
    other[:] = grid
    other[:10] = grid[[1, 0, 6, 7, 8, 9, 2, 3, 4, 5]]
    return out
