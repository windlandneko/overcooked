"""The C++ vectorized environment (engine.cpp), bit-exact with core/env.py."""

from __future__ import annotations

import ctypes
import platform
import subprocess
import tempfile
import weakref
from functools import cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from numpy.ctypeslib import ndpointer

from core.catalog import get_map
from core.scenario import ScenarioSpec

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "engine.cpp"
FLAGS = {"Darwin": ["-mcpu=native"], "Linux": ["-march=native", "-fopenmp"]}.get(platform.system(), [])

TERRAIN = {tile: code for code, tile in enumerate(" PXOTDS")} | {"1": 0, "2": 0}
# Per-seat event channels of `VecEnv.events`. drop_* fire when an item fresh from a dispenser or pot is first
# put on a counter, handoff_pickup when a chef picks up what the partner put down, curl is the move's
# clockwise component around the kitchen centre, zone is +1/-1 for the lower-right/upper-left half.
EVENTS = ("pot_onion", "pot_tomato", "start_cook", "dish_pickup", "soup_pickup", "soup_deliver",
          "drop_onion", "handoff_pickup", "drop_soup", "collision", "curl", "zone")
MAX_SLOTS = 64  # counters and pots share one occupancy bitmask


@cache
def load_library() -> ctypes.CDLL:
    # The file name carries a source hash, so a rebuild never replaces a library a running process has loaded.
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    digest = sha256(SOURCE.read_bytes() + " ".join(FLAGS).encode()).hexdigest()[:16]
    path = HERE / f"libengine_{platform.machine().lower()}_{digest}{suffix}"
    if not path.exists():
        with tempfile.TemporaryDirectory(dir=HERE) as build:  # concurrent builders each publish a whole file
            built = Path(build) / path.name
            subprocess.run(["c++", "-O3", "-std=c++20", "-shared", "-fPIC", *FLAGS, str(SOURCE), "-o", str(built)],
                           check=True)
            built.replace(path)
    lib = ctypes.CDLL(str(path))
    handle, i32, f32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    mask = ndpointer(np.bool_, ndim=1, flags="C_CONTIGUOUS")
    lib.env_create.argtypes = [i32, i32, i32, ndpointer(np.uint8, flags="C_CONTIGUOUS"), *[i32] * 7, *[handle] * 7]
    lib.env_create.restype = handle
    lib.env_destroy.argtypes = [handle]
    lib.env_seed.argtypes = [handle, ctypes.c_uint64]
    lib.env_reset.argtypes = [handle, mask, f32]
    lib.env_step.argtypes = [handle, ndpointer(np.int32, ndim=2, flags="C_CONTIGUOUS"), f32]
    lib.env_record.argtypes = [handle, mask, i32]
    lib.env_ring_size.argtypes = [handle]
    return lib


class VecEnv:
    """`count` games stepped together, finished games restarting inside step().

    Every reset and step rewrites the output tensors in place: `obs` (count, 2, 26, H, W) holds both seats'
    grids, `timestep` (count,) the clock, `augmented` whether the running game began mid-game, and `sparse`,
    `shaped`, `done` and `events` (count, 2, 12) the last step's results. Copy anything you keep across calls.

    With `aug_prob`, a restarting game instead begins, with that probability, from a mid-game state: a
    recorded one (see record_states) or a random plausible kitchen.
    """

    def __init__(self, spec: ScenarioSpec, count: int, pin_memory: bool = False):
        grid = get_map(spec.layout_id)["grid"]
        self.spec = spec
        self.count = count
        self.height, self.width = len(grid), len(grid[0])
        terrain = np.array([[TERRAIN[tile] for tile in row] for row in grid], dtype=np.uint8)
        assert np.isin(terrain, (TERRAIN["P"], TERRAIN["X"])).sum() <= MAX_SLOTS
        starts = ["".join(grid).index(seat) for seat in "12"]

        pin = pin_memory and torch.cuda.is_available()
        self.obs = torch.zeros((count, 2, 26, self.height, self.width), pin_memory=pin)
        self.timestep = torch.zeros(count, dtype=torch.int32, pin_memory=pin)
        self.sparse = torch.zeros(count, pin_memory=pin)
        self.shaped = torch.zeros(count, pin_memory=pin)
        self.done = torch.zeros(count, dtype=torch.bool, pin_memory=pin)
        self.events = torch.zeros((count, 2, len(EVENTS)), pin_memory=pin)
        self.augmented = torch.zeros(count, dtype=torch.bool, pin_memory=pin)
        outputs = (self.obs, self.timestep, self.sparse, self.shaped, self.done, self.events, self.augmented)
        self._results = tuple(t.numpy() for t in outputs[2:6])
        self._everyone = np.ones(count, dtype=bool)

        self._lib = load_library()
        self._handle = self._lib.env_create(
            count, self.width, self.height, terrain, *starts,
            spec.onion, spec.tomato, spec.cook_time, spec.reward, spec.horizon,
            *(t.data_ptr() for t in outputs),
        )
        self.close = weakref.finalize(self, self._lib.env_destroy, self._handle)

    def seed(self, seed: int) -> None:
        self._lib.env_seed(self._handle, seed % 2**64)

    def reset(self, mask: np.ndarray | None = None, aug_prob: float = 0.0) -> None:
        """Restart the masked games, or all of them."""
        self._lib.env_reset(self._handle, self._mask(mask), aug_prob)

    def step(self, actions: np.ndarray, aug_prob: float = 0.0) -> tuple[np.ndarray, ...]:
        """Apply joint actions (count, 2); returns numpy views of (sparse, shaped, done, events)."""
        self._lib.env_step(self._handle, np.ascontiguousarray(actions, dtype=np.int32), aug_prob)
        return self._results

    def record_states(self, stride: int = 10, mask: np.ndarray | None = None) -> None:
        """Keep every stride-th masked game's state, if it is clean and mid-game, for augmented restarts."""
        self._lib.env_record(self._handle, self._mask(mask), stride)

    @property
    def ring_size(self) -> int:
        return self._lib.env_ring_size(self._handle)

    def _mask(self, mask: np.ndarray | None) -> np.ndarray:
        return self._everyone if mask is None else np.ascontiguousarray(mask, dtype=bool)
