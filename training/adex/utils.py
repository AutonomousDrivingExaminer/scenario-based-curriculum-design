from __future__ import annotations

import time
from typing import Callable

import imageio
import numpy as np
import ray
import wandb
from omegaconf import DictConfig

from adex_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from adex_gym.wrappers.server import ServerWrapper
from training.adex.rollouts import EpisodeRolloutWorker, FixedLengthRolloutWorker
from training.adex.rollouts.distributed import DistributedRolloutCollector


class Timer:

    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.time()

    @property
    def duration(self):
        return self.end - self.start

def wrap_server_env(env, port: int, gpu: int) -> BaseScenarioEnvWrapper:
    tm_port = 2000 - port + 8000
    env = ServerWrapper(env, world_port=port, gpus=[str(gpu)], traffic_manager_port=tm_port)
    return env


def make_episode_worker(
        env_fn: Callable[[], BaseScenarioEnvWrapper],
        remote: bool = False,
        render: bool = False
) -> EpisodeRolloutWorker | ray.actor.ActorHandle:
    kwargs = {
        "env_fn": env_fn,
        "render": render
    }
    if remote:
        return ray.remote(EpisodeRolloutWorker).remote(**kwargs)
    else:
        return EpisodeRolloutWorker(**kwargs)


def make_fixed_length_worker(
        env_fn: Callable[[], BaseScenarioEnvWrapper],
        num_steps: int,
        remote: bool = False,
        render: bool = False,
        reset_between_rollouts: bool = False
) -> FixedLengthRolloutWorker | ray.actor.ActorHandle:
    kwargs = {
        "env_fn": env_fn,
        "num_steps": num_steps,
        "render": render,
        "reset_between_rollouts": reset_between_rollouts
    }
    if remote:
        return ray.remote(FixedLengthRolloutWorker).remote(**kwargs)
    else:
        return FixedLengthRolloutWorker(**kwargs)


def make_distributed_rollout_collector(worker_cfg: DictConfig, worker_fn: Callable[
    [int, str], ray.actor.ActorHandle]) -> DistributedRolloutCollector:
    return DistributedRolloutCollector(
        worker_fn=worker_fn,
        available_gpus=worker_cfg.available_gpus,
        num_workers=worker_cfg.num_workers,
        start_port=worker_cfg.start_port
    )

def make_video(path: str, frames: list[np.ndarray], fps: int) -> wandb.Video:
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    video = wandb.Video(path, fps=fps, format="mp4")
    return video
