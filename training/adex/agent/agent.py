from __future__ import annotations

import abc

import numpy as np
from tensordict import TensorDict

from adex_gym.scenarios.actor_configuration import ActorConfiguration


class Agent(abc.ABC):

    @abc.abstractmethod
    def update(self, trajectory: TensorDict) -> dict:
        """
        Updates the agent's parameters based on the trajectory.
        :param trajectory: A tensordict with keys 'obs', 'action', 'reward', 'next_obs',
        'terminated', 'truncated', and possibly additional keys (e.g. 'value').
        :return: A dictionary containing metrics (e.g. loss, etc.).
        """

    @abc.abstractmethod
    def checkpoint(self) -> dict:
        """
        Returns a dictionary containing the agent's current training state.
        :return: A dictionary containing the agent's parameters, optimizer state, etc.
        """

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the agent's training state from the given path.
        :param path: The path to the checkpoint.
        """

    @abc.abstractmethod
    def get_actor(self, eval: bool = False) -> Actor:
        """
        Returns an actor that can be used for evaluation or training.
        :param eval: If true, the actor should be in evaluation mode (e.g. deterministic).
        :return: An actor instance.
        """


class Actor(abc.ABC):

    @abc.abstractmethod
    def __call__(self, obs: TensorDict) -> tuple[np.ndarray, TensorDict]:
        """
        Receives the observation and returns the action and additional information (e.g. action
        distribution parameters, value estimate, etc.).
        :param obs: A tensordict containing the observation.
        :return: A tuple containing the action and additional information.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def setup(self, config: ActorConfiguration) -> None:
        """
        Setup will be called after the environment has been reset. The actor configuration
        contains simulation related information such as the route and the role name.
        :param config: The actor configuration.
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, params: dict) -> None:
        """
        The update function can be used to update the actor's parameters or hyperparameters.
        :param params: A dictionary containing the parameters.
        :return:
        """
        raise NotImplementedError
