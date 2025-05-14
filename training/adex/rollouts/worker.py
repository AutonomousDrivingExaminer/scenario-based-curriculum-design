from __future__ import annotations

import dataclasses
from typing import Callable, Any, Protocol

import numpy as np
import torch
from tensordict import TensorDict

from adex.sampler import EnvConfiguration
from training.adex.agent.agent import Actor


@dataclasses.dataclass
class Rollout:
    trajectory: TensorDict
    infos: list[list[dict[str, Any]]]  # list of episode info lists
    videos: list[np.ndarray] = None
    env_configs: EnvConfiguration = None


def get_tensordict(
        obs: dict[str, Any],
        actions: dict[str, Any],
        reward: dict[str, Any],
        next_obs: dict[str, Any],
        terminated: dict[str, Any],
        truncated: dict[str, Any],
        actor_outputs: dict[str, Any]
):
    return TensorDict({
        agent: TensorDict({
            "obs": obs[agent],
            "action": actions[agent],
            "reward": reward[agent],
            "next_obs": next_obs[agent],
            "terminated": terminated[agent],
            "truncated": truncated[agent],
            **actor_outputs[agent],
        }, batch_size=())
        for agent in obs.keys()
    }, batch_size=())


class BaseRolloutWorker(Protocol):

    def update_env(self, options: dict):
        pass

    def rollout(self, policy_mapping_fn: Callable[[str], Actor], render: bool = False) -> Rollout:
        pass

    def close(self):
        pass


class FixedLengthRolloutWorker:

    def __init__(
            self,
            env_fn: Callable,
            num_steps: int,
            reset_between_rollouts: bool = True,
            render: bool = False,
    ):
        self._num_steps = num_steps
        self._reset_between_rollouts = reset_between_rollouts
        self._last_obs = None
        self._needs_reset = True
        self.env = env_fn()
        self._options = {}

    def _setup_actors(self, actors: dict[str, Actor]) -> None:
        for config in self.env.current_scenario.config.ego_vehicles:
            actor = actors[config.rolename]
            actor.setup(config)

    def update_env(self, options: dict):
        self._options = options

    def rollout(self, policy_mapping_fn: Callable[[str], Actor], render: bool = False) -> Rollout:
        trajectory, infos, videos = [], [], []
        episode_infos, episode_frames = [], []
        done = self._needs_reset or self._reset_between_rollouts
        actors = {}
        if not done:
            assert self.env.current_scenario is not None
            actors = {agent: policy_mapping_fn(agent) for agent in self.env.agents}
            self._setup_actors(actors)
        obs = self._last_obs
        for t in range(self._num_steps):
            if done:
                if len(episode_infos) > 0:
                    infos.append(episode_infos)
                if len(episode_frames) > 0:
                    videos.append(np.stack(episode_frames, axis=0))
                obs, info = self.env.reset(options=self._options)
                actors = {agent: policy_mapping_fn(agent) for agent in self.env.agents}
                self._setup_actors(actors)
                episode_infos = [info]
                if render:
                    episode_frames = [self.env.render()]

            obs = TensorDict(obs, batch_size=())
            actions, actor_ouputs = {}, {}
            for id, actor in actors.items():
                action, actor_output = actor(obs[id])
                actions[id] = action
                actor_ouputs[id] = actor_output

            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            if t == self._num_steps - 1:
                truncated = {agent: True for agent in obs.keys()}

            step = get_tensordict(
                obs=obs,
                actions=actions,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
                actor_outputs=actor_ouputs
            )

            trajectory.append(step)
            episode_infos.append(info)
            if render:
                episode_frames.append(self.env.render())
            done = all(terminated.values())
            obs = next_obs

        self._last_obs = obs
        self._needs_reset = done
        trajectory = torch.stack(trajectory, dim=0).to_tensordict()
        return Rollout(
            trajectory=trajectory,
            infos=infos,
            videos=videos if render else None
        )

    def close(self):
        self.env.close()


class EpisodeRolloutWorker(BaseRolloutWorker):

    def __init__(self, env_fn: Callable):
        self.env = env_fn()
        self._options = {}

    def update_env(self, options: dict):
        self._options = options

    def rollout(self, policy_mapping_fn: Callable[[str], Actor], render: bool = False) -> Rollout:
        obs, info = self.env.reset(options=self._options)
        actors = {agent: policy_mapping_fn(agent) for agent in self.env.agents}
        for config in self.env.current_scenario.config.ego_vehicles:
            actor = actors[config.rolename]
            actor.setup(config)
        trajectory, infos, frames = [], [info], []
        if render:
            frames.append(self.env.render())
        done = False
        while not done:
            actions, actor_ouputs = {}, {}
            obs = TensorDict(obs, batch_size=())
            for id, actor in actors.items():
                actions[id], actor_ouputs[id] = actor(obs[id])
            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            step = get_tensordict(
                obs=obs,
                actions=actions,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
                actor_outputs=actor_ouputs
            )
            trajectory.append(step)
            infos.append(info)
            if render:
                frames.append(self.env.render())
            done = all(terminated.values())
            obs = next_obs
        return Rollout(
            trajectory=torch.stack(trajectory, dim=0).to_tensordict(),
            infos=[infos],
            videos=[np.stack(frames, axis=0)] if render else None
        )

    def close(self):
        self.env.close()
