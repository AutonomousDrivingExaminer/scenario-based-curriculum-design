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
from torchrl.modules import MLP, MaskedCategorical
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


def gae(rewards, values, episode_ends, gamma, lam):
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
        gae_step = delta + gamma * lam * episode_ends[:, t] * gae_step
        # And store it
        advantages[:, t] = gae_step
    return advantages


class MAPPOTrainer:
    def __init__(
            self,
            model: MAPPO,
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
        if rollout.device != self.device:
            rollout = rollout.to(self.device)

        agents = sorted(list(rollout["obs"].keys()))
        for item in rollout.keys():
            rollout[item] = torch.stack(
                [rollout[item][agent] for agent in agents], dim=2
            )

        self.compute_advantages(rollout)
        rollout = TensorDict(
            optree.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), rollout.to_dict()),
            batch_size=(rollout.batch_size[0] * rollout.batch_size[1],),
        )

        with torch.no_grad():
            embeds = self.model._encoder(rollout["obs"])
            actions = rollout["action"]
            action_masks = rollout["obs"]["action_mask"].to(torch.bool)
            advantages = rollout["advantages"]
            returns = rollout["returns"]

            dist = self.model.get_action_dist(embeds, action_masks)
            logpi_old = dist.log_prob(actions)

        rollout["logpi"] = logpi_old

        metrics = []
        for epoch in range(self.num_epochs):
            indices = torch.randperm(rollout.batch_size[0])
            minibatch_sz = int(rollout.batch_size[0] // self.num_minibatches)
            for i in range(self.num_minibatches):
                minibatch_idx = indices[i * minibatch_sz: (i + 1) * minibatch_sz]
                mb_actions = actions[minibatch_idx]
                mb_embeds = embeds[minibatch_idx]
                mb_action_masks = action_masks[minibatch_idx]
                mb_advantages = advantages[minibatch_idx]
                mb_returns = returns[minibatch_idx]
                mb_logpi_old = logpi_old[minibatch_idx]

                pi = self.model.get_action_dist(mb_embeds, mb_action_masks)
                values = self.model.get_value(mb_embeds)

                logpi = pi.log_prob(mb_actions).view(-1, 1)
                logratio = logpi - mb_logpi_old.view(-1, 1)
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()

                pg_loss1 = - mb_advantages * ratio
                pg_loss2 = - mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = torch.mean(0.5 * (values - mb_returns) ** 2)
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

    def compute_advantages(self, rollout: TensorDict, gamma: float, lmbda: float) -> TensorDict:
        values = rollout["value"]
        next_value = rollout["next_value"]
        rewards = rollout["reward"]

        advantages = torch.zeros_like(rollout["reward"])
        batch, num_steps = rollout.batch_size[0], rollout.batch_size[1]
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            terminated = batch["terminated"][:, t]
            truncated = batch["truncated"][:, t]
            nextnonterminal = 1.0 - (terminated | truncated).float()
            nextvalues = next_value[:, t]
            delta = rewards[:, t] + gamma * nextvalues * (1.0 - terminated.float()) - values[:, t]
            lastgaelam = delta + gamma * lmbda * nextnonterminal * lastgaelam
            advantages[:, t] = lastgaelam
        returns = values + advantages
        return rollout, returns

    @staticmethod
    def from_config(model: MAPPO, device, config: dict) -> MAPPOTrainer:
        return MAPPOTrainer(
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


class MAPPO(nn.Module):
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

        self._policy_net = MLP(
            in_features=embed_dim,
            out_features=action_space.n,
            num_cells=[64, 64],
            activation_class=nn.ELU,
        )

        self._vf_net = nn.Sequential(
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=8,
                    dim_feedforward=256,
                    dropout=0,
                    batch_first=True,
                ),
                num_layers=4,
            ),
            LambdaLayer(lambda x: x.mean(1)),
            MLP(in_features=embed_dim, out_features=1, num_cells=128, depth=2),
        )

    def get_value(self, embed: torch.Tensor) -> torch.Tensor:
        values = self._vf_net(embed).mean(1)
        return values

    def get_action_dist(
            self, embed: torch.Tensor, mask: torch.Tensor
    ) -> MaskedCategorical:
        logits = self._policy_net(embed)
        return MaskedCategorical(logits=logits, mask=mask)

    def forward(self, obs: TensorDict) -> tuple[MaskedCategorical, torch.Tensor]:
        embed = self._encoder(obs)
        mask = obs["action_mask"].to(torch.bool)
        dist = self.get_action_dist(embed, mask)
        # value function is shared across agents, first dim is sequence length
        values = self.get_value(embed)

        return dist, values


class MAPPOActor(Actor):
    def __init__(
            self, model: MAPPO, device: torch.device = torch.device("cpu")
    ) -> None:
        self._model = model.to(device)
        self._device = device

    def act(self, obs: dict) -> np.ndarray:
        agents = sorted(list(obs.keys()))
        obs = TensorDict(obs, batch_size=()).to(self._device)
        obs = torch.stack([obs[agent] for agent in agents], dim=0)
        obs = torch.unsqueeze(obs, dim=0)
        with torch.no_grad():
            dist, _ = self._model(obs)
            action = dist.sample()
            action = action.squeeze(0)
        action = {
            agent: action[i].cpu().numpy().item() for i, agent in enumerate(agents)
        }
        return action

    def update(self, params: dict) -> None:
        self._model.load_state_dict(params)
