from __future__ import annotations

from typing import Any

from tensordict import TensorDict


class RegretApproximation:
    def __call__(self, trajectory: Any) -> float:
        raise NotImplementedError


class PositiveValueLoss(RegretApproximation):
    def __call__(self, trajectory: dict | TensorDict) -> float:
        assert "advantage" in trajectory.keys()
        return trajectory["advantage"].clamp(min=0).mean().item()


class MonteCarloRegret(RegretApproximation):
    def __call__(self, trajectory: dict | TensorDict) -> float:
        assert "value" in trajectory.keys()
