from __future__ import annotations

import numpy as np

from training.adex.buffers import EnvironmentBuffer
from training.adex.sampler import EnvConfiguration, EnvParamSampler


class PrioritizedLevelReplayBuffer(EnvironmentBuffer):

    def __init__(
            self,
            generator: EnvParamSampler,
            max_size: int,
            replay_rate: float,
            p: float,
            temperature: float,
            update_sampler: bool = False
    ) -> None:
        self._max_size = max_size
        self._level_index = {}
        self._levels = []
        self._scores = []
        self._staleness = []
        self._p = p
        self._replay_rate = replay_rate
        self._temperature = temperature
        self._generator = generator
        self._update_sampler = update_sampler

    def get_buffer_stats(self) -> dict:
        return {
            "size": len(self._levels),
            "mean_p": np.mean(self._p),
            "std_p": np.std(self._p),
            "temperature": self._temperature,
            "mean_score": np.mean(self._scores),
            "std_score": np.std(self._scores),
            "mean_staleness": np.mean(self._staleness),
            "std_staleness": np.std(self._staleness),
        }

    def get_level_stats(self) -> dict:
        score_probs = self._compute_score_prob()
        staleness_probs = self._compute_staleness_prob()
        return {
            level.id: {
                "score": self._scores[i],
                "staleness": self._staleness[i],
                "score_prob": score_probs[i],
                "staleness_prob": staleness_probs[i],
                "params": level.params,
            }
            for i, level in enumerate(self._levels)
        }

    def checkpoint(self) -> dict:
        return {
            "levels": self._levels,
            "scores": self._scores,
            "staleness": self._staleness,
        }

    def add(self, env_configuration: EnvConfiguration, score: float = 0) -> None:
        self._level_index[env_configuration.id] = len(self._levels)
        self._levels.append(env_configuration)
        self._scores.append(score)
        self._staleness.append(0)

    def get_next_level(self, num_levels: int = None) -> list[tuple[EnvConfiguration, bool]]:
        levels = []
        for _ in range(num_levels or 1):
            replay_decision = np.random.rand() < self._replay_rate and len(self._levels) > 0
            staleness = np.array(self._staleness)
            if replay_decision:
                config = self.sample()
                idx = self._level_index[config.id]
                staleness[idx] = 0
            else:
                config = self._generator()

            staleness += 1
            self._staleness = staleness.tolist()
            levels.append((config, replay_decision))
        return levels[0] if num_levels is None else levels

    def sample(self) -> EnvConfiguration:
        # Compute probabilities
        p_score = self._compute_score_prob()
        c_score = self._compute_staleness_prob()
        probs = (1 - self._p) * p_score + self._p * c_score
        i = np.random.choice(len(self._levels), p=probs)

        # Update staleness
        staleness = np.array(self._staleness)
        staleness[i] = 0
        self._staleness = staleness.tolist()
        return self._levels[i]

    def generate(self) -> EnvConfiguration:
        return self._generator()

    def update(self, env_configuration: EnvConfiguration, score: float) -> None:
        if env_configuration.id not in self._level_index:
            self.add(env_configuration, score=score)
        else:
            i = self._level_index[env_configuration.id]
            self._scores[i] = score
        if self._update_sampler:
            self._generator.update(env_configuration, score)

    def __len__(self):
        return len(self._levels)

    def _compute_score_prob(self) -> np.ndarray:
        scores = np.array(self._scores)
        ranks =  np.argsort(scores) + 1
        inv_ranks = 1 / ranks
        probs = np.power(inv_ranks, 1 / self._temperature)
        probs /= probs.sum()
        return probs
    
    def _compute_staleness_prob(self) -> np.ndarray:
        staleness = np.array(self._staleness)
        if staleness.sum() == 0:
            return np.ones(len(self._levels)) / len(self._levels)
        probs = staleness / staleness.sum()
        return probs
