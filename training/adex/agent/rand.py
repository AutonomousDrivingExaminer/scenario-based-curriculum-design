from __future__ import annotations

import gymnasium
import numpy as np
from tensordict import TensorDict

from training.adex.agent.agent import Actor, Agent


class RandomAgent(Agent):
    def __init__(self, id: str) -> None:
        super().__init__(id)
        self.id = id
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def update(self, trajectory: TensorDict) -> dict:
        return {}
    
    def checkpoint(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def get_actor(self) -> Actor:
        return RandomActor(self.action_space)
    

class RandomActor(Actor):

    def __init__(self, action_space: gymnasium.spaces.Box) -> None:
        self.action_space = action_space

    def __call__(self, obs: TensorDict) -> tuple[np.ndarray, TensorDict]:
        return self.action_space.sample(), TensorDict({}, batch_size=obs.batch_size)
    
    def update(self, params: dict) -> None:
        pass
    