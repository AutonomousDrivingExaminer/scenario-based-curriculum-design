from __future__ import annotations

import random

import numpy as np

from adex.buffers import PrioritizedLevelReplayBuffer
from adex.sampler import EnvConfiguration
from training.adex.agent.agent import Agent
from training.adex.buffers.env_buffer import EnvironmentBuffer


class AgentBuffer:

    def __init__(self, lamda: float, max_agents: int, max_envs_per_agent: int, plr_temp: float,
                 plr_p: float) -> None:
        self._agents = {}
        self._scores = {}
        self._env_buffers = {}
        self._lamda = lamda
        self._plr_temp = plr_temp
        self._plr_p = plr_p
        self._max_agents = max_agents
        self._max_envs_per_agent = max_envs_per_agent

    def add(self, id: str, agent: Agent) -> None:
        self._agents[id] = agent
        self._scores[id] = {}
        self._env_buffers[id] = PrioritizedLevelReplayBuffer(
            max_size=self._max_envs_per_agent,
            temperature=self._plr_temp,
            p=self._plr_p
        )

    def sample(self) -> tuple[str, Agent]:
        scores = {}
        max_regret = -np.inf
        for agent_id, level_scores in self._scores.items():
            for scene, score in level_scores.items():
                scores[(agent_id, scene)] = score
                if score > max_regret:
                    max_regret = score

        if len(scores) == 0:
            return random.choice(list(self._agents.values()))
                
        weights = {}
        N = len(scores)
        num_max = len([pair for pair, score in scores.items() if score >= max_regret])
        for pair, score in scores.items():
            if score < max_regret:
                weights[pair] = self._lamda / N
            else:
                weights[pair] = 1 - self._lamda * (N - num_max) / N

        pairs, weights = zip(*weights.items())
        i = np.random.choice(len(pairs), p=weights)
        agent_id, _ = pairs[i]
        return agent_id, self._agents[agent_id]

    def get_env_buffer(self, id: str) -> EnvironmentBuffer:
        return self._env_buffers[id]

    def update(self, id: str, env_configuration: EnvConfiguration, score: float) -> None:
        self._scores[id][env_configuration] = score
        self._env_buffers[id].update(env_configuration, score=score)
