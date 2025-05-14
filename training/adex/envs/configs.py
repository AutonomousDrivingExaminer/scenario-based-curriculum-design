from __future__ import annotations

from srunner.scenariomanager.scenarioatomics.atomic_criteria import InRouteTest
from srunner.scenariomanager.traffic_events import TrafficEventType

from adex_gym.wrappers import ObservationConfig, BirdViewCropType, BirdViewMasks


def birdview():
    return {
        "obs_config": ObservationConfig(
            as_rgb=True,
            width=96,
            height=96,
            pixels_per_meter=4,
            crop_type=BirdViewCropType.FRONT_AREA_ONLY,
            masks=[
                BirdViewMasks.ROUTE,
                BirdViewMasks.EGO,
                BirdViewMasks.ROAD,
                BirdViewMasks.CENTERLINES,
                BirdViewMasks.VEHICLES
            ]
        ),
    }


def criteria():
    return {
        "criteria_fns": [
            lambda scenario: InRouteTest(
                route=scenario.config.ego_vehicles[0].route,
                actor=scenario.ego_vehicles[0],
                offroad_max=5
            )
        ]
    }


def tasks():
    return {
        "infraction_avoidance": {
            "infractions": [
                TrafficEventType.COLLISION_STATIC.name,
                TrafficEventType.COLLISION_VEHICLE.name,
                TrafficEventType.COLLISION_PEDESTRIAN.name,
                TrafficEventType.ON_SIDEWALK_INFRACTION,
            ],
            "penalties": 0,
            "terminate_on_infraction": True
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


def route_following_wrappers(agent_names: list[str]) -> dict:
    config = {
        "birdview": birdview(),
        "criteria": criteria(),
        "tasks": {agent_name: tasks() for agent_name in agent_names},
        "action_normalization": {
            "agents": agent_names
        },
        "frame_skip": {
            "skip": 2
        },
        "route_following_preprocessing": {
            "max_episode_time": 15.0,
            "stack": 4,
            "grayscale": True
        },
        "visualization": {
            "agent_names": agent_names
        }
    }
    return config
