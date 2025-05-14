from __future__ import annotations
from einops import rearrange
import numpy as np
from tensordict.nn import TensorDictModule
from torch import nn, optim, vmap
import gymnasium
import optree
import torch
import torchrl
from torchrl.data.tensor_specs import TensorDict, TensorSpec
from torchrl.modules import MLP, MaskedCategorical, TruncatedNormal
from torchrl.objectives.ppo import GAE
from training.adex.envs import make_route_following_env
from training.adex.models.common import LambdaLayer
from training.adex.models.observations import DictObservationEncoder

import torch.functional as F
from torchrl.data.replay_buffers import (
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
    ListStorage,
)
import logging

from training.adex.rollouts.actor import Actor


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def gae(rewards, values, episode_ends, truncated, gamma, lam):
    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = np.zeros((N,))
    advantages = np.zeros((N, T))
    for t in reversed(range(T - 1)):
        # First compute delta, which is the one-step TD error
        delta = (
            rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
        )
        # Then compute the current step's GAE by discounting the previous step
        # of GAE, resetting it to zero if the episode ended, and adding this
        # step's delta
        gae_step = delta + gamma * lam * episode_ends[:, t] * truncated[:, t] * gae_step
        # And store it
        advantages[:, t] = gae_step
    return advantages


class ContinuousPPOTrainer:
    def __init__(
        self,
        model: PPOContinuous,
        device: torch.device = torch.device("cpu"),
        lr: float = 1e-4,
        clip_ratio: float = 0.2,
        max_grad_norm: float = 0.5,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        gae_lambda: float = 0.95,
        discount_factor: float = 0.99,
        num_epochs: int = 4,
        batch_size: int = 32,
        normalize_advantages: bool = True,
        num_minibatches: int = 4,
        target_kl: float = 0.01,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.lmbda = gae_lambda
        self.gamma = discount_factor
        self.clip_ratio = clip_ratio
        self.max_grad_norm = max_grad_norm
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_minibatches = num_minibatches
        self.target_kl = target_kl
        self.normalize_advantages = normalize_advantages
        self.device = device

    def update(self, rollout: TensorDict) -> dict:
        self.model.to(self.device).train()
        rollout = rollout.to(self.device)
        agents = sorted(list(rollout["obs"].keys()))
        for item in rollout.keys():
            rollout[item] = torch.stack(
                [rollout[item][agent] for agent in agents], dim=2
            )
        rollout = TensorDict(rollout, batch_size=(*rollout.batch_size, len(agents)))
        rollout["action"][..., 0] = rollout["action"][..., 0] * 2 - 1
        rollout["action"][..., 2] = rollout["action"][..., 2] * 2 - 1
        self.compute_advantages(rollout)
        rollout = rollout.view(-1)
       
        with torch.no_grad():
            pi, _ = self.model(rollout["obs"])
            logpi_old = pi.log_prob(rollout["action"])

        rollout["logpi"] = logpi_old

        metrics = []
        B = rollout.batch_size[0]
        minibatch_size = B // self.num_minibatches
        for epoch in range(self.num_epochs):
            indices = np.arange(B)
            np.random.shuffle(indices)
            start = 0
            for i in range(1,self.num_minibatches+1):
                end = start + minibatch_size * i
                end = min(end, B)
                minibatch = rollout[indices[start:end]]
                pi, values = self.model(minibatch["obs"])
                #todo: logprob over minibatch gives different results than logprob over batch
                logpi = pi.log_prob(minibatch["action"]).view(-1, 1)
                logratio = logpi - minibatch["logpi"].view(-1, 1)
                ratio = torch.exp(logratio)
                pg_loss1 = -minibatch["advantages"] * ratio
                pg_loss2 = -minibatch["advantages"] * torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = torch.mean(0.5 * (values - minibatch["returns"]) ** 2)
                entropy_loss = pi.entropy().mean()
                loss = (
                    pg_loss
                    - self.entropy_coeff * entropy_loss
                    + self.value_coeff * value_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()

                metrics = {
                    "pg_loss": pg_loss,
                    "value_loss": value_loss,
                    "entropy_loss": entropy_loss,
                    "loss": loss,
                    "approx_kl": approx_kl,
                    "clipfrac": clipfracs,
                }

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    logging.info(
                        f"Early stopping at epoch {epoch} due to reaching max kl."
                    )
                    break

        metrics = optree.tree_map(lambda x: x.detach().cpu().numpy().item(), metrics)
        self.model.eval()
        return metrics

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def compute_advantages(self, rollout: TensorDict) -> TensorDict:
        lmbda = 0.95
        gamma = 0.99
        num_steps = rollout.batch_size[1]
        rewards = rollout["reward"]
        dones = rollout["terminated"].to(rewards)
        trunc = rollout["truncated"].to(rewards)
        with torch.no_grad():
            values = self.model.get_value(rollout["obs"]).to(rewards)
            values = values.reshape(rollout.batch_size)

        gae_step = 0.0
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(num_steps - 1)):
            delta = (
                rewards[..., t, :]
                + gamma * values[:, t + 1, :] * (1 - dones[:, t, :])
                - values[:, t, :]
            )
            gae_step = delta + gamma * lmbda * (1 - dones[:, t, :]) * (1 - trunc[:, t, :]) * gae_step
            advantages[:, t] = gae_step

        returns = advantages + values
        rollout["returns"] = returns

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollout["advantages"] = advantages
        return rollout

    @staticmethod
    def from_config(model: PPOContinuous, device, config: dict) -> ContinuousPPOTrainer:
        return ContinuousPPOTrainer(
            model=model,
            device=device,
            lr=config["lr"],
            clip_ratio=config["clip_ratio"],
            max_grad_norm=config["max_grad_norm"],
            value_coeff=config["value_coeff"],
            entropy_coeff=config["entropy_coeff"],
            gae_lambda=config["gae_lambda"],
            discount_factor=config["discount_factor"],
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
            normalize_advantages=config["normalize_advantages"],
            num_minibatches=config["num_minibatches"],
            target_kl=config["target_kl"],
        )


class PPOContinuous(nn.Module):
    def __init__(
        self,
        obs_space: gymnasium.spaces.Dict,
        action_space: gymnasium.Space,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self._encoder = DictObservationEncoder(
            obs_space=obs_space, keys=["birdview", "velocity"], embed_dim=embed_dim
        )

        self._policy_net = nn.Sequential(
            DictObservationEncoder(
                obs_space=obs_space,
                keys=["birdview", "velocity"],
                embed_dim=embed_dim
            ),
            layer_init(nn.Linear(embed_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_space.shape[0]*2), std=1.0)
        )

        self._vf_net = nn.Sequential(
            DictObservationEncoder(
                obs_space=obs_space,
                keys=["birdview", "velocity"],
                embed_dim=embed_dim
            ),
            layer_init(nn.Linear(embed_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def get_value(self, embed: torch.Tensor) -> torch.Tensor:
        values = self._vf_net(embed)
        return values

    def get_action_dist(
        self, embed: torch.Tensor
    ) -> TruncatedNormal:
        params = self._policy_net(embed)
        loc, scale = torch.split(params, params.shape[-1] // 2, dim=-1)
        scale = scale.exp()
        return TruncatedNormal(loc=loc, scale=scale)

    def forward(self, obs: TensorDict) -> tuple[TruncatedNormal, torch.Tensor]:
        dist = self.get_action_dist(obs)
        values = self.get_value(obs)
        return dist, values


class ContinuousPPOActor(Actor):
    def __init__(
        self, model: PPOContinuous, device: torch.device = torch.device("cpu")
    ) -> None:
        self._model = model.to(device)
        self._device = device

    def act(self, obs: dict) -> np.ndarray:
        agents = sorted(list(obs.keys()))
        obs = TensorDict(obs, batch_size=()).to(self._device)
        obs = torch.stack([obs[agent] for agent in agents], dim=0)
        with torch.no_grad():
            dist, _ = self._model(obs)
            action = dist.sample()

        action = action.cpu().numpy()
        action[: 0] *= 0.5
        action[:, 0] += 0.5
        action[: 2] *= 0.5
        action[:, 2] += 0.5
        
        action = {
            agent: action[i] for i, agent in enumerate(agents)
        }
        return action

    def update(self, params: dict) -> None:
        self._model.load_state_dict(params)


if __name__ == "__main__":
    env = make_route_following_env("training/scenarios/coop_intersection.scenic", continuous=True)
    model = ContinuousPPOTrainer(
        obs_space=env.observation_space(env.possible_agents[0]),
        action_space=env.action_space(env.possible_agents[0]),
    )
    trainer = ContinuousPPOTrainer(model=model)
    metrics = trainer.update(rollout)
    print(metrics)
