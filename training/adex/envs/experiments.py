from __future__ import annotations

import numpy as np
from omegaconf import DictConfig
from srunner.scenariomanager.scenarioatomics.atomic_criteria import InRouteTest
from srunner.scenariomanager.traffic_events import TrafficEventType

import adex_gym
from adex_gym.envs import renderers
from adex_gym.wrappers.birdseye_view.birdseye import BirdViewCropType, BirdViewMasks
from adex_gym.wrappers.birdview import ObservationConfig
from examples.example_agents.autopilot import AutopilotAgent


def route_following(cfg: DictConfig) -> dict:
    config = {
        "env_args": {
            "scenario_specification": cfg.env.scenario,
            "agent_name_prefixes": [cfg.env.agent_name],
            "render_mode": "rgb_array",
            "resample_scenes": cfg.env.resample,
            "scenes_per_scenario": cfg.env.num_scenes_per_scenario,
            "render_config": renderers.camera_pov(agent=cfg.env.agent_name),
        },
        "birdview": {
            "use_rgb": True,
            "obs_config": ObservationConfig(
                width=96,
                height=96,
                pixels_per_meter=4,
                crop_type=BirdViewCropType.FRONT_AREA_ONLY,
                masks=[BirdViewMasks.ROUTE, BirdViewMasks.EGO, BirdViewMasks.ROAD,
                       BirdViewMasks.CENTERLINES]
            ),
        },
        "criteria": {
            "criteria_fns": [
                lambda scenario: InRouteTest(
                    route=scenario.config.ego_vehicles[0].route,
                    actor=scenario.ego_vehicles[0],
                    offroad_max=5
                )
            ]
        },
        "tasks": {
            "student": {
                "infraction_avoidance": {
                    "infractions": [
                        TrafficEventType.COLLISION_STATIC.name,
                        TrafficEventType.COLLISION_VEHICLE.name,
                        TrafficEventType.COLLISION_PEDESTRIAN.name,
                        TrafficEventType.ON_SIDEWALK_INFRACTION,
                    ],
                },
                "min_velocity": {
                    "target_velocity": 5
                },
                "route_following": {
                    "extra_reward_on_completion": 100
                },
                "weights": [0, 0.5, 0.5],
                "termination_fn": lambda terminated: terminated[0]
            }
        },
        "action_normalization": {
            "agents": ["student"]
        },
        "route_following_preprocessing": {
            "max_episode_time": 15.0,
            "stack": 4
        },
        "visualization": {
            "agent_name": "student"
        }
    }
    return config


if __name__ == "__main__":
    env = adex_gym.scenic_env(
        host="localhost",
        port=2006,
        scenario_specification="training/scenarios/four_way_intersection.scenic",
        agent_name_prefixes=["student"],
        render_mode="rgb_array",
        resample_scenes=True,
        scenes_per_scenario=2,
        render_config=renderers.camera_pov(agent="student"),
        traffic_manager_port=8000,
    )
    env = route_following(env)
    env.reset()
    done = False
    cum_reward = 0.0
    agent = AutopilotAgent(
        role_name="student",
        carla_host="localhost",
        carla_port=2006
    )
    agent.setup(path_to_conf_file=None, route=env.current_scenario.config.ego_vehicles[0].route)

    while not done:
        ctrl = agent.run_step(None, 0.0)
        if ctrl.brake > 0.0:
            acceleration = -ctrl.brake
        else:
            acceleration = ctrl.throttle
        steer = ctrl.steer

        action = {
            "student": np.array([acceleration, steer]),
        }
        action = {id: env.action_space(id).sample() for id in env.agents}

        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward["student"]
        done = terminated["student"]
        print(
            f"Progress: {obs['student']['state'][0]}, Time: {obs['student']['state'][1]}"
        )
        print(f"Reward: {reward['student']}, Cumulative Reward: {cum_reward}")
    env.close()
