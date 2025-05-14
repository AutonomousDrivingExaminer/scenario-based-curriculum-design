from __future__ import annotations

import adex_gym
from adex.envs.wrappers import ActionNormalizationWrapper, FrameSkipWrapper, \
    RouteFollowingPreprocessing
from adex.visualization import make_route_visualization
from adex_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from adex_gym.tasks import TaskCombination, InfractionAvoidanceTask, DriveMinVelocityTask, \
    RouteFollowingTask
from adex_gym.wrappers import CarlaVisualizationWrapper, BirdViewObservationWrapper, \
    CriteriaWrapper, TaskWrapper


def make_scenic_env(env_args):
    env = adex_gym.scenic_env(
        **env_args
    )


def wrap_env(env: BaseScenarioEnvWrapper, eval: bool, config) -> BaseScenarioEnvWrapper:
    if "birdview" in config:
        bv_args = config.pop("birdview")
        env = BirdViewObservationWrapper(
            env=env,
            **bv_args
        )

    if "criteria" in config:
        criteria_args = config.pop("criteria")
        env = CriteriaWrapper(
            env=env,
            **criteria_args
        )

    task_args = config.pop("tasks", {})
    tasks = {}
    for agent, args in task_args.items():
        agent_tasks = []
        if "infraction_avoidance" in args:
            task = InfractionAvoidanceTask(
                agent=agent,
                infractions=args["infraction_avoidance"]["infractions"],
                penalties=args["infraction_avoidance"]["penalties"],
                terminate_on_infraction=args["infraction_avoidance"]["terminate_on_infraction"]
            )
            agent_tasks.append(task)
        if "min_velocity" in args:
            task = DriveMinVelocityTask(
                agent=agent,
                target_velocity=args["min_velocity"]["target_velocity"]
            )
            agent_tasks.append(task)
        if "route_following" in args:
            task = RouteFollowingTask(
                agent=agent,
                extra_reward_on_completion=args["route_following"]["extra_reward_on_completion"]
            )
            agent_tasks.append(task)
        tasks[agent] = TaskCombination(
            agent=agent,
            tasks=agent_tasks,
            weights=args["weights"],
            termination_fn=args["termination_fn"]
        )

    env = TaskWrapper(
        env=env,
        tasks=tasks,
        ignore_wrapped_env_reward=True,
        ignore_wrapped_env_termination=False,
    )

    if "action_normalization" in config:
        args = config.pop("action_normalization")
        env = ActionNormalizationWrapper(env=env, agents=args["agents"])

    if "frame_skip" in config:
        args = config.pop("frame_skip")
        env = FrameSkipWrapper(env=env, skip=args["skip"])

    if "route_following_preprocessing" in config:
        args = config.pop("route_following_preprocessing")
        env = RouteFollowingPreprocessing(
            env=env,
            **args
        )

    if eval and "visualization" in config:
        vis_args = config.pop("visualization")
        route_viss = [make_route_visualization(agent_name) for agent_name in vis_args["agent_names"]]
        env = CarlaVisualizationWrapper(env=env, callbacks=route_viss)

    return env
