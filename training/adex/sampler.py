from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Any

import optuna
import scenic
from scenic.domains.driving.roads import ManeuverType


@dataclass
class EnvConfiguration:
    id: Any
    options: dict
    params: dict
    stats: dict = None


class EnvParamSampler(Protocol):

    def __call__(self) -> EnvConfiguration:
        ...

    def update(self, config: EnvConfiguration, score: float) -> None:
        ...


class RandomEnvParamSampler(EnvParamSampler):

    def __init__(self, path: str, max_vehicles: int, max_iterations: int = 10000) -> None:
        self._max_vehicles = max_vehicles
        with open(path) as s:
            self._scenario_code = s.read()
        self._max_iterations = max_iterations
        self._map_path = scenic.scenarioFromFile(path).params["map"]
        self._maneuvers = [
            ManeuverType.STRAIGHT.value,
            ManeuverType.LEFT_TURN.value,
            ManeuverType.RIGHT_TURN.value
        ]
        self._num_samples = 0

    def __call__(self) -> EnvConfiguration:
        params = {
            "MANEUVER_TYPE": random.choice([
                ManeuverType.STRAIGHT.value,
                ManeuverType.LEFT_TURN.value,
                ManeuverType.RIGHT_TURN.value
            ]),
            "NUM_NPCS": random.randint(0, self._max_vehicles + 1),
            "NPC_PARAMS": {
                "target_speed": random.choice([i * 10 for i in range(1, 6)]),
                "ignore_traffic_lights": random.choice([True, False]),
                "ignore_vehicles": random.choice([True, False])
            }
        }
        params["map"] = self._map_path
        scenario = scenic.scenarioFromString(string=self._scenario_code, params=params)
        scene, _ = scenario.generate(maxIterations=self._max_iterations)
        config = EnvConfiguration(
            id=self._num_samples,
            params=params,
            options={
                "scene": {
                    "binary": scenario.sceneToBytes(scene),
                    "code": self._scenario_code,
                    "params": params
                },
            }
        )
        self._num_samples += 1
        return config

    def update(self, config: EnvConfiguration, score: float) -> None:
        super().update(config, score)


class OptunaEnvParamSampler(EnvParamSampler):

    def __init__(self, path: str, max_vehicles: int, max_iters: int = 100000):
        with open(path) as s:
            self._scenario_code = s.read()
        self._map_path = scenic.scenarioFromFile(path).params["map"]
        self._maneuvers = [
            ManeuverType.STRAIGHT.value,
            ManeuverType.LEFT_TURN.value,
            ManeuverType.RIGHT_TURN.value
        ]
        self._max_vehicles = max_vehicles
        self._max_iterations = max_iters
        self._lateral_pid_params = {}
        self._longitudinal_pid_params = {}
        self._num_samples = 0
        self._study = optuna.create_study(direction="maximize")

        self._trials = {}

    def __call__(self, params: dict = None) -> EnvConfiguration:
        if params is None:
            trial = self._study.ask()
            params = {
                "MANEUVER_TYPE": trial.suggest_categorical("maneuver", self._maneuvers),
                "NUM_NPCS": trial.suggest_int("num_npcs", 0, self._max_vehicles),
                "NPC_PARAMS": {
                    "target_speed": trial.suggest_categorical("npc_target_speed",
                                                              [i * 10 for i in range(1, 6)]),
                    "ignore_traffic_lights": trial.suggest_categorical("npc_ignore_traffic_lights",
                                                                       [True, False]),
                    "ignore_vehicles": trial.suggest_categorical("npc_ignore_vehicles",
                                                                 [True, False]),
                }
            }
            self._trials[self._num_samples] = trial
        params["map"] = self._map_path
        scenario = scenic.scenarioFromString(string=self._scenario_code, params=params)
        scene, _ = scenario.generate(maxIterations=self._max_iterations)
        config = EnvConfiguration(
            id=self._num_samples,
            params=params,
            options={
                "scene": {
                    "binary": scenario.sceneToBytes(scene),
                    "code": self._scenario_code,
                    "params": params
                },
            }
        )
        self._num_samples += 1
        return config

    def get_eval_config(self) -> list[EnvConfiguration]:
        params = []
        for maneuver in self._maneuvers:
            cfg = self._get_config(
                num_npcs=0,
                maneuver=maneuver,
                conflict_only=False,
                npc_params=self._get_npc_params(target_speed=0, ignore_traffic_lights=True,
                                                ignore_vehicles=True)
            )
            params.append(cfg)
            for num_npcs in range(1, 3):
                cfg = self._get_config(
                    num_npcs=num_npcs,
                    maneuver=maneuver,
                    conflict_only=True,
                    npc_params=self._get_npc_params(
                        target_speed=0,
                        ignore_traffic_lights=True,
                        ignore_vehicles=True
                    )
                )
                params.append(cfg)

        configs = []
        for p in params:
            configs.append(self(params=p))
        return configs

    def _get_npc_params(self, target_speed: float, ignore_traffic_lights: bool,
                        ignore_vehicles: bool):
        return {
            "target_speed": target_speed,
            "ignore_traffic_lights": ignore_traffic_lights,
            "ignore_vehicles": ignore_vehicles,
        }

    def _get_config(self, num_npcs: int, maneuver: ManeuverType, conflict_only: bool,
                    npc_params: dict) -> dict:
        return {
            "MANEUVER_TYPE": maneuver.value,
            "NUM_NPCS": num_npcs,
            "NPC_MANEUVER_CONFLICT_ONLY": conflict_only,
            "NPC_PARAMS": npc_params
        }

    def update(self, config: EnvConfiguration, score: float) -> None:
        if config.id in self._trials:
            self._study.tell(self._trials[config.id], score)
            self._trials.pop(config.id)


class HedgeOptimizer(EnvParamSampler):

    def __call__(self) -> EnvConfiguration:
        return super().__call__()

    def update(self, config: EnvConfiguration, score: float) -> None:
        super().update(config, score)
