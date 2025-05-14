from __future__ import annotations

from scenic.core.scenarios import random

from adex.sampler import EnvConfiguration
from training.adex.buffers.env_buffer import EnvironmentBuffer


class RandomReplayBuffer(EnvironmentBuffer):
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._levels = []

    def add(self, env_configuration: EnvConfiguration) -> None:
        self._levels.append(env_configuration)
        if len(self._levels) > self._max_size:
            self._levels.pop(0)

    def sample(self) -> EnvConfiguration:
        return random.choice(self._levels)
    
    def update(self, env_configuration: EnvConfiguration, **kwargs) -> None:
        pass
    
