"""Rule-based Overcooked partners, driven directly from the 26-channel grid observation.

Each partner uses precomputed shortest-path tables for the map (including partner-avoiding and clockwise /
counter-clockwise tables), so stepping a batch inside training adds negligible overhead.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterable
from enum import IntEnum
from functools import cache

import numpy as np

from core.catalog import get_map


class Act(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    STAY = 4
    INTERACT = 5


DIRS = ((0, -1), (0, 1), (1, 0), (-1, 0))
INF = 999

SCRIPTED_NAMES = (
    "greedy_rammer",
    "greedy_polite",
    "cw_looper",
    "ccw_looper",
    "onion_feeder",
    "pot_plater",
    "counter_relay",
    "lazy_blocker",
)


class MapNav:
    """Precomputed shortest-path and adjacency tables for one layout."""

    def __init__(self, layout_id: str):
        grid = get_map(layout_id)["grid"]
        self.layout_id = layout_id
        self.h, self.w = len(grid), len(grid[0])
        self.cells = self.h * self.w
        flat_terrain = np.array(list("".join(grid)))
        self.walkable = np.isin(flat_terrain, (" ", "1", "2"))
        self.floors = np.nonzero(self.walkable)[0].astype(np.int32)
        self.floor_id = np.full(self.cells, -1, dtype=np.int32)
        self.floor_id[self.floors] = np.arange(len(self.floors), dtype=np.int32)

        self.pots, self.counters, self.onions, self.dishes, self.serves = (
            np.nonzero(flat_terrain == ch)[0].astype(np.int32) for ch in "PXODS"
        )
        self.neighbors: tuple[tuple[tuple[int, int], ...], ...] = tuple(
            tuple(
                (d, ny * self.w + nx)
                for d, (dx, dy) in enumerate(DIRS)
                if 0 <= (nx := pos % self.w + dx) < self.w and 0 <= (ny := pos // self.w + dy) < self.h
            )
            for pos in range(self.cells)
        )

        # Connected components of floor tiles (e.g. separate rooms on unident_s_hard).
        self.comp = np.full(self.cells, -1, dtype=np.int32)
        cid = 0
        for f in map(int, self.floors):
            if self.comp[f] >= 0:
                continue
            q = deque([f])
            self.comp[f] = cid
            while q:
                for _, v in self.neighbors[q.popleft()]:
                    if self.walkable[v] and self.comp[v] < 0:
                        self.comp[v] = cid
                        q.append(v)
            cid += 1

        # Prefer counters bridging rooms, else interior counters, else any reachable counter.
        multi, interior, reachable = [], [], []
        for c in map(int, self.counters):
            comps = {int(self.comp[v]) for _, v in self.neighbors[c] if self.walkable[v]}
            if not comps:
                continue
            reachable.append(c)
            if len(comps) > 1:
                multi.append(c)
            y, x = divmod(c, self.w)
            if 0 < y < self.h - 1 and 0 < x < self.w - 1:
                interior.append(c)
        self.relay_counters = np.array(multi or interior or reachable, dtype=np.int32)
        self.reachable_counters = np.array(reachable, dtype=np.int32)

        self.adj_face = np.full((self.cells, self.cells), -1, dtype=np.int8)
        for p in map(int, self.floors):
            for d, t in self.neighbors[p]:
                self.adj_face[t, p] = d

        self.base_dist, self.base_dir = self._build_bfs(blocked=-1)
        self.cw_dist, self.cw_dir = self._build_directed(sign=1.0)
        self.ccw_dist, self.ccw_dir = self._build_directed(sign=-1.0)

        n_f = len(self.floors)
        self.avoid_dist = np.full((n_f, self.cells, self.cells), INF, dtype=np.int16)
        self.avoid_dir = np.full((n_f, self.cells, self.cells), -1, dtype=np.int8)
        for fi, blk in enumerate(map(int, self.floors)):
            self.avoid_dist[fi], self.avoid_dir[fi] = self._build_bfs(blocked=blk)

    def _build_bfs(self, blocked: int) -> tuple[np.ndarray, np.ndarray]:
        dist = np.full((self.cells, self.cells), INF, dtype=np.int16)
        first = np.full((self.cells, self.cells), -1, dtype=np.int8)
        for t in range(self.cells):
            q: deque[int] = deque()
            for _, adj in self.neighbors[t]:
                if self.walkable[adj] and adj != blocked:
                    dist[t, adj] = 0
                    q.append(adj)
            while q:
                u = q.popleft()
                du = int(dist[t, u]) + 1
                for d, v in self.neighbors[u]:
                    if self.walkable[v] and v != blocked and dist[t, v] > du:
                        dist[t, v] = du
                        first[t, v] = d ^ 1
                        q.append(v)
        return dist, first

    def _build_directed(self, sign: float) -> tuple[np.ndarray, np.ndarray]:
        """Dijkstra preferring clockwise (sign=+1) or counter-clockwise (sign=-1) motion around map center."""
        cy, cx = 0.5 * (self.h - 1), 0.5 * (self.w - 1)
        dist = np.full((self.cells, self.cells), INF, dtype=np.int16)
        first = np.full((self.cells, self.cells), -1, dtype=np.int8)
        for t in range(self.cells):
            cost = np.full(self.cells, 1e9, dtype=np.float32)
            pq: list[tuple[float, int]] = []
            for _, adj in self.neighbors[t]:
                if self.walkable[adj]:
                    cost[adj] = 0.0
                    dist[t, adj] = 0
                    heapq.heappush(pq, (0.0, adj))
            while pq:
                cu, u = heapq.heappop(pq)
                if cu > cost[u]:
                    continue
                for d_u_to_v, v in self.neighbors[u]:
                    if not self.walkable[v]:
                        continue
                    d_v_to_u = d_u_to_v ^ 1
                    vy, vx = divmod(v, self.w)
                    dx, dy = DIRS[d_v_to_u]
                    curl = ((cy - vy) * dx + (vx - cx) * dy) * sign
                    # Reverse cost (> 16-tile ring on counter_circuit) forces looping rather than backtracking.
                    step_cost = 1.0 if curl > 0 else 1.5 if curl == 0 else 25.0
                    if (nv := cu + step_cost) < cost[v]:
                        cost[v] = nv
                        dist[t, v] = min(INF - 1, round(nv))
                        first[t, v] = d_v_to_u
                        heapq.heappush(pq, (nv, v))
        return dist, first


@cache
def get_nav(layout_id: str) -> MapNav:
    return MapNav(layout_id)


def _step_toward(
    nav: MapNav, pos: int, facing: int, partner: int, target: int, mode: str, rng: np.random.Generator
) -> int:
    if target < 0:
        return Act.STAY
    if (face := int(nav.adj_face[target, pos])) >= 0:
        return Act.INTERACT if facing == face else face
    if mode == "avoid":
        py, px = divmod(partner, nav.w)
        sy, sx = divmod(pos, nav.w)
        # Small tie-break prevents symmetric livelocks when two bots meet head-on.
        if abs(sy - py) + abs(sx - px) <= 2 and rng.random() < 0.12:
            if free := [nd for nd, v in nav.neighbors[pos] if nav.walkable[v] and v != partner]:
                return free[int(rng.integers(len(free)))]
        if (pfi := int(nav.floor_id[partner])) >= 0:
            if (d := int(nav.avoid_dir[pfi, target, pos])) >= 0:
                return d
            cur = abs(sy - py) + abs(sx - px)
            free_moves = [
                (abs(v // nav.w - py) + abs(v % nav.w - px), nd)
                for nd, v in nav.neighbors[pos]
                if nav.walkable[v] and v != partner
            ]
            best_dist, best_d = max(free_moves, key=lambda x: x[0], default=(cur, Act.STAY))
            return best_d if best_dist > cur else Act.STAY
    table = nav.cw_dir if mode == "cw" else nav.ccw_dir if mode == "ccw" else nav.base_dir
    return d if (d := int(table[target, pos])) >= 0 else Act.STAY


def _nearest(dist_col: np.ndarray, *groups: Iterable[int]) -> int:
    best_t, best_d = -1, INF
    for group in groups:
        for t in group:
            if (d := int(dist_col[t])) < best_d:
                best_d, best_t = d, int(t)
    return best_t


def scripted_action(obs: np.ndarray, nav: MapNav, name: str, rng: np.random.Generator) -> int:
    """Single action for scripted partner `name` from its egocentric (26, H, W) observation."""
    if name == "lazy_blocker":
        if (u := rng.random()) < 0.78:
            return Act.STAY
        if u < 0.86:
            return int(rng.integers(4))
        name = "greedy_rammer"

    flat = obs.reshape(26, nav.cells)
    pos, partner = int(flat[0].argmax()), int(flat[1].argmax())
    facing = int(flat[2:6, pos].argmax())

    mode = {"greedy_polite": "avoid", "cw_looper": "cw", "ccw_looper": "ccw"}.get(name, "base")
    match mode:
        case "avoid" if (pfi := int(nav.floor_id[partner])) >= 0:
            dist_col = nav.avoid_dist[pfi, :, pos]
            if int(dist_col[nav.pots].min()) >= INF:
                dist_col = nav.base_dist[:, pos]
        case "cw":
            dist_col = nav.cw_dist[:, pos]
        case "ccw":
            dist_col = nav.ccw_dist[:, pos]
        case _:
            dist_col = nav.base_dist[:, pos]

    held = next(
        (item for ch, item in ((21, "soup"), (22, "dish"), (23, "onion"), (24, "tomato")) if flat[ch, pos] > 0.5),
        "none",
    )

    ready_pots: list[int] = []
    cooking_pots: list[tuple[float, int]] = []
    full_idle_pots: list[int] = []
    need_onion_pots: list[tuple[int, int]] = []
    for p in map(int, nav.pots):
        if nav.base_dist[p, pos] >= INF:
            continue
        if flat[21, p] > 0.5:
            ready_pots.append(p)
        elif flat[20, p] > 0.5:
            cooking_pots.append((float(flat[20, p]), p))
        elif (n_ing := round(float(flat[16, p] + flat[17, p]))) >= 3:
            full_idle_pots.append(p)
        else:
            need_onion_pots.append((n_ing, p))

    rc = nav.reachable_counters[nav.base_dist[nav.reachable_counters, pos] < INF]
    empty_counters = rc[flat[21:25, rc].sum(axis=0) <= 0.5]
    soup_counters = rc[flat[21, rc] > 0.5]
    dish_counters = rc[flat[22, rc] > 0.5]
    onion_counters = rc[flat[23, rc] > 0.5]

    relay = nav.relay_counters[nav.base_dist[nav.relay_counters, pos] < INF]
    empty_relay = relay[flat[21:25, relay].sum(axis=0) <= 0.5]

    def go(target: int) -> int:
        return _step_toward(nav, pos, facing, partner, target, mode, rng)

    def drop() -> int:
        return go(_nearest(dist_col, empty_counters))

    def wait_at(target: int) -> int:
        return Act.STAY if target < 0 or nav.adj_face[target, pos] >= 0 else go(target)

    def best_onion_pot() -> int:
        return min(need_onion_pots, key=lambda x: (-x[0], int(dist_col[x[1]])))[1]

    def soonest_pot() -> int:
        return min(cooking_pots, key=lambda x: (x[0], int(dist_col[x[1]])))[1]

    if held == "soup":
        return go(_nearest(dist_col, nav.serves))
    if held == "tomato":
        return drop()

    match (name, held):
        case ("onion_feeder", "dish"):
            return drop()
        case ("onion_feeder", "onion"):
            if need_onion_pots:
                return go(best_onion_pot())
            return go(_nearest(dist_col, empty_relay if empty_relay.size else empty_counters))
        case ("onion_feeder", "none"):
            if full_idle_pots:
                return go(_nearest(dist_col, full_idle_pots))
            return go(_nearest(dist_col, nav.onions, onion_counters) if need_onion_pots else _nearest(dist_col, nav.onions))

        case ("pot_plater", "onion"):
            return drop()
        case ("pot_plater", "dish"):
            if ready_pots:
                return go(_nearest(dist_col, ready_pots))
            if full_idle_pots and empty_counters.size:
                return drop()
            return wait_at(soonest_pot() if cooking_pots else _nearest(dist_col, nav.pots))
        case ("pot_plater", "none"):
            if full_idle_pots:
                return go(_nearest(dist_col, full_idle_pots))
            if soup_counters.size:
                return go(_nearest(dist_col, soup_counters))
            return go(_nearest(dist_col, nav.dishes, dish_counters))

        case ("counter_relay", "onion"):
            if empty_relay.size and (not need_onion_pots or int(dist_col[empty_relay].min()) <= 6):
                return go(_nearest(dist_col, empty_relay))
            return go(best_onion_pot()) if need_onion_pots else drop()
        case ("counter_relay", "none"):
            if full_idle_pots:
                return go(_nearest(dist_col, full_idle_pots))
            if soup_counters.size:
                return go(_nearest(dist_col, soup_counters))
            if ready_pots:
                return go(_nearest(dist_col, nav.dishes, dish_counters))
            return go(_nearest(dist_col, nav.onions))

        case (_, "dish"):
            if ready_pots:
                return go(_nearest(dist_col, ready_pots))
            return wait_at(soonest_pot()) if cooking_pots else drop()
        case (_, "onion"):
            return go(best_onion_pot()) if need_onion_pots else drop()
        case _:
            if soup_counters.size:
                return go(_nearest(dist_col, soup_counters))
            if full_idle_pots:
                return go(_nearest(dist_col, full_idle_pots))
            best_dish = _nearest(dist_col, nav.dishes, dish_counters)
            # Compare against true BFS distance (not directed loop cost) so loopers do not oscillate around dish timing.
            soon_ready = (
                bool(cooking_pots)
                and best_dish >= 0
                and min(rem for rem, _ in cooking_pots) <= int(nav.base_dist[best_dish, pos]) + 4
            )
            if best_dish >= 0 and (ready_pots or soon_ready):
                return go(best_dish)
            if need_onion_pots:
                return go(_nearest(dist_col, nav.onions, onion_counters))
            if best_dish >= 0 and cooking_pots:
                return go(best_dish)
            return Act.STAY


def make_scripted_partner(layout_id: str, name: str, seed: int = 0):
    """Callable `act(grid) -> int` matching zsc.evaluate's partner interface."""
    nav = get_nav(layout_id)
    rng = np.random.default_rng(seed)

    def act(grid: np.ndarray) -> int:
        return scripted_action(np.asarray(grid, dtype=np.float32), nav, name, rng)

    return act
