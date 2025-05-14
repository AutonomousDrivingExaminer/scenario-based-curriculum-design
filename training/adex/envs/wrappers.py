from __future__ import annotations

from collections import deque
from typing import Any

import carla
import gymnasium
import numpy as np
import torch
import torchvision
from gymnasium.spaces import Box
from pettingzoo.utils.env import AgentID, ObsType, ActionType

from adex_gym import BaseScenarioEnv
from adex_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from adex_gym.wrappers.vectorized import VecEnvWrapper


class ActionNormalizationWrapper(BaseScenarioEnvWrapper):
    def __init__(self, env: BaseScenarioEnv, agents: list[AgentID] = None):
        super().__init__(env)
        self._agents = agents

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        original_space = self.env.action_space(agent)
        if agent in self._agents:
            return gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=original_space.dtype
            )
        else:
            return original_space

    def step(
            self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        actions = actions.copy()
        for agent, action in actions.items():
            if agent in self._agents:
                acceleration, steer = action[0], action[1]
                brake = abs(min(acceleration, 0))
                throttle = max(acceleration, 0)
                actions[agent] = np.array([throttle, steer, brake]).clip(
                    -1.0, 1.0
                )
        return self.env.step(actions)


class RouteFollowingPreprocessing(BaseScenarioEnvWrapper):
    def __init__(
            self,
            env: BaseScenarioEnv,
            max_episode_time: float = None,
            stack: int = 1,
            grayscale: bool = True,
    ):
        super().__init__(env)
        self._num_stack = stack
        self._max_duration = max_episode_time
        self._obs_stack = deque(maxlen=self._num_stack)
        self._progress = {agent: 0.0 for agent in self.env.possible_agents}
        self._grayscale = grayscale

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        obs_space = self.env.observation_space(agent)
        bv_space = obs_space["birdview"]
        shape = bv_space.shape
        channels = self._num_stack * (1 if self._grayscale else 3)
        obs_space["birdview"] = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=(channels, shape[0], shape[1]),
            dtype=np.float32
        )
        obs_space["state"] = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,) if self._max_duration else (1,),
            dtype=np.float32
        )
        return obs_space

    def _process_birdview(self, birdview: np.ndarray) -> dict[str, np.ndarray]:
        birdview = torch.from_numpy(birdview.transpose(2, 0, 1)) / 255.0
        if self._grayscale:
            birdview = torchvision.transforms.Grayscale(num_output_channels=1)(
                birdview
            )
        birdview = birdview.numpy()
        return birdview

    def reset(
            self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:

        observation, info = self.env.reset(seed=seed, options=options)
        self._obs_stack.clear()
        self._obs_stack.append(observation)
        self._progress = {agent: 0.0 for agent in observation}
        obs = {}
        for agent in observation:
            self._progress[agent] = self._get_progress(agent, info)
            obs[agent] = self._get_observation(agent, observation[agent], info)
        return obs, info

    def step(
            self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self._obs_stack.append(obs)
        new_obs = {}
        for agent in actions.keys():
            self._progress[agent] = self._get_progress(agent, info)
            new_obs[agent] = self._get_observation(agent, obs[agent], info)
        return new_obs, reward, terminated, truncated, info

    def _get_progress(self, agent: str, info: dict) -> float:
        for event in info[agent]["events"]:
            if event["event"] == "ROUTE_COMPLETION":
                if "route_completed" in event:
                    progress = event["route_completed"]
                    return progress
                else:
                    return 0.0
        return 0.0

    def _get_observation(
            self, agent: str, obs: dict, info: dict
    ) -> dict[AgentID, np.ndarray]:
        birdviews = []
        for frame in self._obs_stack:
            birdview = frame[agent]["birdview"].copy()
            birdview = self._process_birdview(birdview)
            birdviews.append(birdview)

        birdview = np.concatenate(birdviews, axis=0)
        if len(birdview) != self._num_stack * (1 if self._grayscale else 3):
            pad = np.zeros(
                (self._num_stack * (1 if self._grayscale else 3) - len(birdview),
                 *birdview.shape[1:])
            )
            birdview = np.concatenate((pad, birdview), axis=0)

        state = [self._progress[agent] / 100.0]
        if self._max_duration:
            current_time = info["__common__"]["simulation_time"]
            state.append(current_time / self._max_duration)
        state = np.array(state)
        new_obs = obs.copy()
        new_obs["birdview"] = birdview.astype(np.float32)
        new_obs["state"] = state.astype(np.float32)
        return new_obs


class FrameSkipWrapper(BaseScenarioEnvWrapper):
    def __init__(self, env: BaseScenarioEnvWrapper, skip: int = 1, render: bool = False):
        super().__init__(env)
        self._render = render
        self._frames = []
        self._skip = skip

    def step(
            self, actions: dict
    ) -> tuple[
        dict,
        dict[Any, float],
        dict[Any, bool],
        dict[Any, bool],
        dict[Any, dict],
    ]:
        cum_reward = {id: 0 for id in actions.keys()}
        terminated = {id: False for id in actions.keys()}
        truncated = {id: False for id in actions.keys()}
        #self._frames.clear()

        for _ in range(self._skip):
            obs, reward, term, trun, info = self.env.step(actions)
            cum_reward = {id: r + cum_reward[id] for id, r in reward.items()}
            terminated = {id: t or terminated[id] for id, t in term.items()}
            truncated = {id: t or truncated[id] for id, t in trun.items()}
            #if self._render:
            #    self._frames.append(self.env.render())
            if all(term.values()) or all(trun.values()):
                break
        return obs, cum_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        self._frames.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        if self._render:
            self._frames.append(self.env.render())
        return obs, info


class AdversaryWrapper(BaseScenarioEnvWrapper):

    def __init__(self, env: BaseScenarioEnvWrapper, adversary_prefix: str):
        super().__init__(env)
        self._adversary_prefix = adversary_prefix

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        if not agent.startswith(self._adversary_prefix):
            return self.env.observation_space(agent)
        else:
            space = self.env.observation_space(agent)
            return gymnasium.spaces.Dict({
                **space,
                "student_location": Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            })

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.action_space(agent)

    def _get_relative_location(self, agent: str, location: carla.Location):
        tf: carla.Transform = self.env.actors[agent].get_transform()
        relative_loc = np.array(tf.get_inverse_matrix()) @ np.array([location.x, location.y, location.z, 1])
        return relative_loc[:2]

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[
        dict[AgentID, ObsType], dict[AgentID, dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, observation: dict):
        student_location = self.env.actors["student"].get_location()
        for agent in observation:
            if agent.startswith(self._adversary_prefix):
                relative_location = self._get_relative_location(agent, location=student_location)
                observation[agent]["student_location"] = relative_location
        return observation


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeReward(BaseScenarioEnvWrapper, gymnasium.utils.RecordConstructorArgs):

    def __init__(
            self,
            env: VecEnvWrapper,
            gamma: float = 0.99,
            epsilon: float = 1e-8,
    ):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.return_rms = [{agent: RunningMeanStd() for agent in env.agents} for _ in
                           range(self.num_envs)]
        self.returns = [{agent: 0.0 for agent in env.agents} for _ in range(self.num_envs)]
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        for i in range(self.num_envs):
            for agent in self.returns[i]:
                rew = rews[i][agent]
                ret = self.returns[i][agent]
                new_ret = rew + self.gamma * ret * (1 - terminateds[i][agent])
                self.returns[i][agent] = new_ret
                self.return_rms[i][agent].update(np.array((new_ret,)))
                var = self.return_rms[i][agent].var
                norm_rew = rew / np.sqrt(var + self.epsilon)
                infos[i][agent]["original_reward"] = rew
                rews[i][agent] = norm_rew

        return obs, rews, terminateds, truncateds, infos
