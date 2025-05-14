from __future__ import annotations

import copy
from typing import Callable, Iterator

import gymnasium.spaces
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn

from adex.agent.common import ObservationModel, MaskedCategoricalDistributionHead, \
    IndependentNormalDistributionHead
from adex_gym.scenarios.actor_configuration import ActorConfiguration
from training.adex.agent.agent import Actor, Agent


def _make_encoder(encoder_cfg, obs_space):
    encoders = {}
    for key in encoder_cfg:
        model = hydra.utils.instantiate(encoder_cfg[key])
        if key == "birdview":
            encoders[key] = model(input_channels=obs_space[key].shape[0])
        else:
            encoders[key] = model
    encoder = ObservationModel(models=encoders, obs_space=obs_space)
    return encoder


class ContinuousActor(nn.Module):

    def __init__(self, cfg: DictConfig, action_dim: int, obs_space: gymnasium.spaces.Dict):
        super().__init__()
        self.encoder = _make_encoder(cfg.observation_model, obs_space)
        ActorNet = hydra.utils.instantiate(cfg.actor.net)
        self.actor_net = ActorNet(
            in_features=self.encoder.embedding_dim,
            out_features=action_dim * 2
        )
        self.dist_head = IndependentNormalDistributionHead()

    def forward(self, obs: dict):
        embed = self.encoder(obs)
        return self.dist_head(self.actor_net(embed))


class DiscreteActor(nn.Module):

    def __init__(self, cfg: DictConfig, num_actions: int, obs_space: gymnasium.spaces.Dict):
        super().__init__()
        self.encoder = _make_encoder(cfg.observation_model, obs_space)
        ActorNet = hydra.utils.instantiate(cfg.actor.net)
        self.actor_net = ActorNet(
            in_features=self.encoder.embedding_dim,
            out_features=num_actions
        )
        self.dist_head = MaskedCategoricalDistributionHead()

    def forward(self, obs: dict):
        embed = self.encoder(obs)
        logits = self.actor_net(embed)
        mask = obs.get("action_mask", torch.ones_like(logits)).to(logits.device)
        return self.dist_head(logits, mask=mask)


class Critic(nn.Module):

    def __init__(self, cfg: DictConfig, obs_space: gymnasium.spaces.Dict):
        super().__init__()
        self.encoder = _make_encoder(cfg.observation_model, obs_space)
        CriticNet = hydra.utils.instantiate(cfg.critic.net)
        self.critic_net = CriticNet(
            in_features=self.encoder.embedding_dim,
            out_features=1
        )

    def forward(self, obs: dict):
        embed = self.encoder(obs)
        return self.critic_net(embed)


class PPOAgent(Agent):
    def __init__(
            self,
            is_discrete: bool,
            actor: nn.Module,
            critic: nn.Module,
            device: str,
            ppo_epochs: int,
            num_minibatches: int,
            max_grad_norm: float,
            normalize_advantages: bool,
            value_coef: float,
            clip_coef: float,
            gamma: float,
            lmbda: float,
            optimizer_fn: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer],
    ) -> None:
        self.is_discrete = is_discrete
        self.device = device
        self.epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.max_grad_norm = max_grad_norm
        self.num_updates = 0
        self.normalize_advantages = normalize_advantages
        self.value_coef = value_coef
        self.clip_coef = clip_coef
        self.gamma = gamma
        self.lmbda = lmbda
        self.actor = actor
        self.critic = critic
        self.models = nn.ModuleList([self.actor, self.critic]).to(self.device)
        self.opt = optimizer_fn(self.models.parameters())

    @classmethod
    def from_config(cls, cfg: DictConfig, obs_space: gymnasium.spaces.Dict,
                    action_space: gymnasium.Space, device: str) -> PPOAgent:
        if isinstance(action_space, gymnasium.spaces.Discrete):
            actor = DiscreteActor(cfg, action_space.n, obs_space)
            is_discrete = True
        elif isinstance(action_space, gymnasium.spaces.Box):
            is_discrete = False
            actor = ContinuousActor(cfg, action_space.shape[0], obs_space)
        else:
            raise NotImplementedError

        critic = Critic(cfg, obs_space)
        optimizer_fn = hydra.utils.instantiate(cfg.optimizer)
        return PPOAgent(
            is_discrete=is_discrete,
            actor=actor,
            critic=critic,
            device=device,
            ppo_epochs=cfg.ppo_epochs,
            num_minibatches=cfg.num_minibatches,
            max_grad_norm=cfg.max_grad_norm,
            normalize_advantages=cfg.normalize_advantages,
            value_coef=cfg.value_coef,
            clip_coef=cfg.clip_coef,
            gamma=cfg.gamma,
            lmbda=cfg.lmbda,
            optimizer_fn=optimizer_fn
        )

    def compute_advantages(self, trajectory: TensorDict, gamma: float, lmbda: float) -> TensorDict:
        with torch.no_grad():
            values = self.critic(trajectory["obs"].view(-1))
            next_values = self.critic(trajectory["next_obs"].view(-1))
            values = values.view(*trajectory.batch_size)
            next_value = next_values.view(*trajectory.batch_size)

        rewards = trajectory["reward"]
        advantages = torch.zeros_like(trajectory["reward"])
        batch_size, num_steps = trajectory.batch_size[0], trajectory.batch_size[1]
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            terminated = trajectory["terminated"][:, t]
            truncated = trajectory["truncated"][:, t]
            nextvalues = next_value[:, t]
            delta = rewards[:, t] + gamma * nextvalues * (1.0 - terminated) - values[:, t]
            lastgaelam = delta + gamma * lmbda * (1.0 - terminated) * (1.0 - truncated) * lastgaelam
            advantages[:, t] = lastgaelam
        returns = values + advantages
        trajectory["advantage"] = advantages
        trajectory["return"] = returns
        trajectory["value"] = values
        trajectory["next_value"] = next_value
        return trajectory

    def update(self, trajectory: TensorDict) -> dict:
        metrics = {}
        trajectory = trajectory.to(self.device, dtype=torch.float32)
        trajectory = self.compute_advantages(trajectory, gamma=self.gamma, lmbda=self.lmbda)
        advantages = trajectory["advantage"]
        returns = trajectory["return"]
        values = trajectory["value"]

        trajectory = trajectory.view(-1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        values = values.view(-1)
        indices = np.arange(returns.size(0))

        batch_size = returns.size(0)
        minibatch_size = batch_size // self.num_minibatches
        clipfracs = []
        for i in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_indices = indices[start:start+minibatch_size]
                minibatch = trajectory[mb_indices]
                minibatch = minibatch.to(self.device)

                pi_new = self.actor(minibatch["obs"])
                mb_logprobs = pi_new.log_prob(minibatch["action"])
                mb_logprobs_old = minibatch["log_prob"]
                ratio = (mb_logprobs - mb_logprobs_old).exp()

                A = advantages[mb_indices]

                if self.normalize_advantages:
                    A = (A - A.mean()) / (A.std() + 1e-8)

                pg_loss1 = -A * ratio
                pg_loss2 = -A * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                mb_values = self.critic(minibatch["obs"])
                mb_targets = returns[mb_indices]
                vf_loss = 0.5 * (mb_values - mb_targets).pow(2).mean()

                loss = pg_loss + self.value_coef * vf_loss

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.models.parameters(),
                    max_norm=self.max_grad_norm
                )
                self.opt.step()

        y_pred, y_true = values.cpu(), returns.cpu()
        var_y = torch.var(y_true).item()
        explained_var = torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y
        metrics["pg_loss"] = pg_loss
        metrics["vf_loss"] = vf_loss
        metrics["loss"] = loss
        metrics["mean_value"] = values.mean()
        metrics["approx_kl"] = approx_kl
        metrics["clipfrac"] = torch.mean(torch.tensor(clipfracs))
        metrics["explained_variance"] = explained_var

        for k, v in metrics.items():
            metrics[k] = v.detach().cpu().numpy().item()

        self.num_updates += 1
        return metrics

    def get_actor(self, eval: bool = False) -> Actor:
        model = copy.deepcopy(self.actor).to("cpu")
        return PPOActor(model, discrete=self.is_discrete, eval=eval)

    def checkpoint(self) -> dict:
        return {
            "models": self.models.state_dict(),
            "opt": self.opt.state_dict(),
            "num_updates": self.num_updates,
        }

    def load(self, path: str) -> None:
        state = torch.load(path)
        if "actor" in state.keys():
            self.actor.load_state_dict(state["actor"])
        if "critic" in state.keys():
            self.critic.load_state_dict(state["critic"])
        if "opt" in state.keys():
            self.opt.load_state_dict(state["opt"])
        if "num_updates" in state.keys():
            self.num_updates = state["num_updates"]


class PPOActor(Actor):

    def __init__(self, actor: nn.Module, discrete: bool, eval: bool = False, ) -> None:
        self.actor = actor
        self.eval = eval
        self.discrete = discrete

    def update(self, params: dict) -> None:
        self.actor.load_state_dict(params)

    def setup(self, config: ActorConfiguration) -> None:
        pass

    def __call__(self, obs: TensorDict) -> tuple[np.ndarray, TensorDict]:
        if len(obs.shape) == 0:
            obs = obs.unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            pi = self.actor(obs)
        if self.eval:
            action = pi.mode if self.discrete else pi.mean
        else:
            action = pi.sample()

        agent_output = TensorDict({
            "action": action.squeeze(0),
            "log_prob": pi.log_prob(action).squeeze(0)
        }, batch_size=())
        action = action.cpu().numpy()
        if action.shape[0] == 1:
            action = action.squeeze(0)
        return action, agent_output