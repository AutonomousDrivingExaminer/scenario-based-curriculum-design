import abc

from training.adex.sampler import EnvConfiguration


class EnvironmentBuffer(abc.ABC):

    @abc.abstractmethod
    def add(self, env_configuration: EnvConfiguration) -> None:
        pass

    @abc.abstractmethod
    def sample(self) -> EnvConfiguration:
        pass

    @abc.abstractmethod
    def update(self, env_configuration: EnvConfiguration, score: float) -> None:
        pass

    @abc.abstractmethod
    def __len__(self):
        pass
