"""One configured MDP, multiple independent same-scenario states."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from core.actions import Action
from core.overcooked_mdp import Recipe

from core.scenario import ScenarioSpec, encode_observations

ENV_ID = "overcooked-739950a-single-order-v1"


def scenario_list(spec):
    specs = [spec] if isinstance(spec, ScenarioSpec) else list(spec)
    if not specs or len({item.fingerprint for item in specs}) != len(specs):
        raise ValueError("Choose one or more distinct scenario configurations")
    return specs


def scenario_entries(spec):
    return [
        {"scenario": asdict(item), "scenario_spec_id": item.fingerprint}
        for item in scenario_list(spec)
    ]


def read_scenarios(path, layout_id=None, scenario_id=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("kind") == "team" and not layout_id and not scenario_id:
        return [
            read_scenario(path, scenario_id=entry["id"]) for entry in data["scenarios"]
        ]
    return [read_scenario(path, layout_id, scenario_id)]


def read_scenario(path: str | Path, layout_id=None, scenario_id=None) -> ScenarioSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = data
    if "scenarios" in data:
        candidates = data["scenarios"]
        if scenario_id:
            candidates = [entry for entry in candidates if entry["id"] == scenario_id]
        if layout_id:
            candidates = [
                entry
                for entry in candidates
                if entry["scenario"]["layout_id"] == layout_id
            ]
        if not scenario_id and not layout_id:
            candidates = [
                entry for entry in candidates if entry["id"] == data["default_scenario"]
            ]
        if len(candidates) != 1:
            raise ValueError(
                "Choose exactly one configured scenario; use --scenario-id when maps repeat"
            )
        selected = candidates[0]
    spec = ScenarioSpec(**selected.get("scenario", selected))
    if layout_id and spec.layout_id != layout_id:
        raise ValueError("Requested map is not included in this configuration")
    if (
        "scenario_spec_id" in selected
        and selected["scenario_spec_id"] != spec.fingerprint
    ):
        raise ValueError(
            "Scenario fingerprint does not match the provided parameters/environment"
        )
    if data.get("env_id", ENV_ID) != ENV_ID or selected.get("env_id", ENV_ID) != ENV_ID:
        raise ValueError("This scenario uses a different environment version")
    return spec


def policy_observation(
    spec, mdp, state, player_index, grid, previous_joint_action=None
):
    return {
        "grid": grid,
        "player_index": player_index,
        "layout_id": spec.layout_id,
        "scenario_spec_id": spec.fingerprint,
        "timestep": state.timestep,
        "horizon": spec.horizon,
        "scenario": asdict(spec),
        "previous_joint_action": previous_joint_action,
    }


class SelfPlayEnv:
    def __init__(self, spec: ScenarioSpec, count: int):
        self.spec = spec
        self.mdp = spec.create_mdp()
        self.states = [self.mdp.get_standard_start_state() for _ in range(count)]
        self.previous_actions = [None for _ in range(count)]

    def get_states(self):
        return [state.to_dict() for state in self.states]

    def get_previous_actions(self):
        return [
            None if value is None else list(value) for value in self.previous_actions
        ]

    def set_states(self, states, previous_actions=None):
        from core.overcooked_mdp import OvercookedState

        if len(states) != len(self.states):
            raise ValueError("Environment state count differs")
        self.states = [
            OvercookedState.from_dict(value) if isinstance(value, dict) else value
            for value in states
        ]
        self.previous_actions = (
            [None for _ in states] if previous_actions is None else previous_actions
        )

    def close(self):
        pass

    def activate(self):
        # The upstream recipe table is process-global. Restore it whenever a
        # different map/session or checkpoint evaluation has run in this process.
        Recipe.configure(self.mdp.recipe_config)

    def observe(self):
        width, height = self.mdp.shape
        observations = np.empty((len(self.states), 2, 26, height, width), dtype=np.float32)
        for state, out in zip(self.states, observations):
            encode_observations(self.mdp, state, self.spec.horizon, out=out)
        return observations

    def centralized(self, observations):
        count, _, channels, height, width = observations.shape
        grids = observations.reshape(count, 2 * channels, height, width)
        time_left = np.asarray(
            [1 - state.timestep / self.spec.horizon for state in self.states],
            dtype=np.float32,
        )
        time_layer = np.broadcast_to(
            time_left[:, None, None, None], (count, 1, height, width)
        )
        return np.concatenate([grids, time_layer], axis=1)

    def step(self, actions):
        sparse, shaped, terminated = [], [], []
        for index, joint_action in enumerate(actions):
            self.states[index], info = self.mdp.get_state_transition(
                self.states[index],
                [Action.ALL_ACTIONS[int(action)] for action in joint_action],
            )
            self.previous_actions[index] = [int(action) for action in joint_action]
            sparse.append(sum(info["sparse_reward_by_agent"]))
            shaped.append(sum(info["shaped_reward_by_agent"]))
            terminated.append(self.states[index].timestep >= self.spec.horizon)
        return (
            np.asarray(sparse, dtype=np.float32),
            np.asarray(shaped, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
        )

    def reset_done(self, terminated):
        for index, done in enumerate(terminated):
            if done:
                self.states[index] = self.mdp.get_standard_start_state()
                self.previous_actions[index] = None
