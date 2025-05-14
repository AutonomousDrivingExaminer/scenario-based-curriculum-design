from __future__ import annotations

import logging
import math
from typing import Callable

import carla
import numpy as np
import optuna
import scenic
from scenic.core.scenarios import Scene

import adex_gym
from adex_gym.envs import renderers
from adex_gym.wrappers import CarlaVisualizationWrapper
from examples.example_agents import AutopilotAgent


class ScenicGenerator:

    def __init__(self, scenario: str, sampler: Callable[[optuna.Trial], dict],
                 max_iterations: int = 10000, max_retries=5) -> None:
        self._scenario = scenario
        with open(scenario) as s:
            self._scenario_code = s.read()
        self._map_path = scenic.scenarioFromFile(scenario).params["map"]
        self.param_generator = optuna.create_study(direction="maximize")
        self._sampler = sampler
        self._num_samples = 0
        self._last_trial = None
        self._max_iterations = max_iterations
        self._max_retries = max_retries

    def generate(self, params=None) -> Scene:
        scene = None
        for i in range(self._max_retries):
            try:
                if params is None:
                    trial = self.param_generator.ask()
                    params = self._sampler(trial)
                    self._last_trial = trial
                params["map"] = self._map_path
                scenario = scenic.scenarioFromString(self._scenario_code, params=params)
                scene, _ = scenario.generate(maxIterations=self._max_iterations)
                break
            except Exception as e:
                logging.warning(f"Failed to generate scene: {e}")
                if i == self._max_retries - 1:
                    logging.warning(f"Failed to generate scene after {self._max_retries} retries.")
                pass
        if scene is None:
            return None
        binary = scenario.sceneToBytes(scene)
        config = EnvConfiguration(
            id=self._num_samples,
            scene_binary=binary,
            scenario=self._scenario_code,
            params=params
        )
        self._num_samples += 1

        return config

    def update(self, score: float):
        self.param_generator.tell(self._last_trial, score)


if __name__ == "__main__":
    scenario = "training/scenarios/route_following.scenic"
    generator = ScenicGenerator(scenario=scenario)
    scene = generator.generate()
    env = adex_gym.scenic_env(
        host="localhost",
        port=2000,
        scenes=scene.compile_scene(),
        agent_name_prefixes=["student", "adv"],
        render_mode="human",
        render_config=renderers.camera_pov("student"),
        traffic_manager_port=8000,
    )
    agent = AutopilotAgent(
        role_name="student",
        carla_host="localhost",
        carla_port=2000,
    )


    def route_vis_callback(scenario, world: carla.World) -> None:
        map = world.get_map()
        colors = [(5, 0, 0), (0, 5, 0), (0, 0, 5), (5, 5, 0)]
        for i, route in enumerate([cfg.route for cfg in scenario.config.ego_vehicles]):
            for loc, _ in route:
                wpt = map.get_waypoint(loc)
                wpt_t = wpt.transform
                begin = wpt_t.location + carla.Location(z=0.3)
                angle = math.radians(wpt_t.rotation.yaw)
                end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))

                world.debug.draw_arrow(
                    begin=begin,
                    end=end,
                    arrow_size=0.05,
                    color=carla.Color(*colors[i]),
                    life_time=0.1
                )

        # The callback is then passed to the CarlaVisualizationWrapper. You can register multiple
        # callbacks by passing a list of callbacks.


    env = CarlaVisualizationWrapper(env=env, callbacks=[route_vis_callback])
    for _ in range(10):
        config = generator.generate()
        obs, info = env.reset(options={
            "scene": config.compile_scene()
        })
        cfg = \
            [cfg for cfg in env.current_scenario.config.ego_vehicles if cfg.rolename == "student"][
                0]
        agent.setup(path_to_conf_file=None, route=cfg.route)
        print(f"Scenario: {', '.join([f'{k}={v}' for k, v in config.params.items()])}")
        done = False
        while not done:
            action = {agent: env.action_space(agent).sample() for agent in env.agents}
            ctrl = agent.run_step(obs["student"], 0.0)
            action["student"] = np.array([ctrl.throttle, ctrl.steer, ctrl.brake])
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = all(terminated.values())
