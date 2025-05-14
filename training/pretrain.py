from __future__ import annotations

import logging
import os
import time
from functools import partial
from typing import Callable

import GPUtil
import hydra
import numpy as np
import optree
import ray
import torch
import wandb
from adex.agent.autopilot import AutoPilotActor
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.utils.data import Dataset
from torchrl.modules import IndependentNormal
from tqdm import tqdm

import adex_gym
from adex.agent.agent import Actor
from adex.agent.ppo import PPOAgent
from adex.envs import experiments
from adex.rollouts import DistributedRolloutCollector, get_eval_metrics
from adex.rollouts.worker import (
    FixedLengthRolloutWorker,
    EpisodeRolloutWorker,
)
from adex.visualization import make_route_visualization
from adex_gym.envs import renderers
from adex_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from adex_gym.wrappers import ServerWrapper, CarlaVisualizationWrapper


class TrajectoryDataset(Dataset):
    def __init__(self, path: str):
        self._data = np.load(path, mmap_mode="r")
        self._obs = TensorDict(
            {
                "birdview": torch.from_numpy(self._data["birdview"]),
                "state": torch.from_numpy(self._data["state"]),
            },
            batch_size=self._data["action"].shape[:2],
        ).view(-1)
        self._action = torch.from_numpy(self._data["action"]).view(-1, 2)

    def __len__(self) -> int:
        return self._action.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        item = TensorDict(
            {"observation": self._obs[index], "action": self._action[index]},
            batch_size=(),
        )
        return item


@hydra.main(
    version_base=None, config_path="configs", config_name="imitation_learning"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    ray.init(local_mode=cfg.experiment.debug)
    run = wandb.init(
        project="adex",
        name=cfg.experiment.name,
        mode=cfg.logger.mode if not cfg.experiment.debug else "offline",
        config=dict(cfg),
    )
    device = cfg.experiment.device if torch.cuda.is_available() else "cpu"
    agent_name = cfg.env.agent_name
    actor = AutoPilotActor(role_name=agent_name)
    collector, eval_collector = make_collectors(cfg)
    # collect_data(cfg=cfg, actor=actor, collector=collector)

    collector.close()
    eval_rollout = eval_collector.collect_rollouts(
        batch_size=cfg.experiment.eval_episodes,
        actors={agent_name: actor},
    )
    metrics = get_eval_metrics(eval_rollout)
    logging.info(f"Expert average return: {metrics['mean_return']:.2f}.")
    eval_env = env_fn(cfg=cfg, port=cfg.worker.start_port, eval=True)
    agent = PPOAgent(
        cfg=cfg,
        obs_space=eval_env.observation_space(agent_name),
        action_space=eval_env.action_space(agent_name),
        device=device,
    )
    eval_actor = agent.get_actor(eval=True)
    if cfg.experiment.checkpoint is not None:
        agent.actor.load_state_dict(torch.load(cfg.experiment.checkpoint))
    dataset_dir = cfg.dataset.directory
    files = [f"{dataset_dir}/{f}" for f in os.listdir(dataset_dir)]
    dataset = TrajectoryDataset(path="trajectories.npz")

    dataloader = torch.utils.data.DataLoader(
        batch_size=cfg.imitation.batch_size,
        dataset=dataset,
        collate_fn=lambda x: x,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        agent.actor.parameters(),
        lr=cfg.imitation.lr,
        weight_decay=cfg.imitation.l2_coef,
    )
    best_score = -np.inf
    for epoch in range(cfg.experiment.epochs):
        losses = []
        agent.actor.to(device)
        loop = tqdm(dataloader)
        for batch in loop:
            batch = torch.stack(batch).to(device)
            action = batch.pop("action")
            obs = batch
            action = action.to(device)
            output = agent.actor(obs)
            dist = IndependentNormal(loc=output["loc"], scale=output["scale"])
            entropy = dist.entropy().mean()
            loss = -(
                dist.log_prob(action).mean() + cfg.imitation.ent_coef * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
            loop.set_description(f"Epoch [{epoch}/{cfg.experiment.epochs}]")
            loop.set_postfix(loss=np.mean(losses).item())
        wandb.log(
            {"mean_loss": np.mean(losses), "std_loss": np.std(losses)},
            step=epoch,
        )

        if epoch % cfg.experiment.eval_interval == 0:
            eval_actor = agent.get_actor(eval=True)
            eval_actor.actor.to("cpu")
            eval_rollout = eval_collector.collect_rollouts(
                batch_size=cfg.experiment.eval_episodes,
                actors={agent_name: eval_actor},
            )
            metrics = get_eval_metrics(eval_rollout)
            if metrics["mean_return"] > best_score:
                logging.info(
                    f"New best model with average return {metrics['mean_return']:.2f}."
                )
                best_score = metrics["mean_return"]
                torch.save(eval_actor.actor.state_dict(), "best_actor.pt")
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=epoch)


def get_free_gpus():
    gpus = GPUtil.getGPUs()
    return [gpu.id for gpu in gpus if gpu.memoryFree > 6000]


def env_fn(cfg: DictConfig, port: int, eval: bool) -> BaseScenarioEnvWrapper:
    tm_port = port - 2000 + 8000
    env = adex_gym.scenic_env(
        scenario_specification=cfg.env.scenario,
        agent_name_prefixes=[cfg.env.agent_name],
        render_mode="rgb_array",
        resample_scenes=cfg.env.resample,
        scenes_per_scenario=cfg.env.num_scenes_per_scenario,
        render_config=renderers.camera_pov(agent=cfg.env.agent_name),
        traffic_manager_port=tm_port,
    )

    env = experiments.route_following(
        env=env,
        time_limit=cfg.env.time_limit,
        terminate_on_route_deviation=cfg.env.terminate_on_route_deviation,
    )
    gpus = [str(i) for i in cfg.worker.gpus]
    env = ServerWrapper(env, world_port=port, gpus=gpus)

    if eval:
        route_vis = make_route_visualization(cfg.env.agent_name)
        env = CarlaVisualizationWrapper(env=env, callbacks=[route_vis])
    return env


def make_worker_fn(
    cfg: DictConfig, eval: bool
) -> Callable[[int], ray.actor.ActorHandle]:
    RemoteRolloutWorker = ray.remote(FixedLengthRolloutWorker)
    EvalRemoteRolloutWorker = ray.remote(EpisodeRolloutWorker)

    def worker_fn(port: int) -> ray.actor.ActorHandle:
        return RemoteRolloutWorker.remote(
            env_fn=partial(env_fn, cfg=cfg, port=port, eval=eval),
            num_steps=cfg.dataset.sequence_length,
            reset_between_rollouts=False,
            render=False,
        )

    def eval_worker_fn(port: int) -> ray.actor.ActorHandle:
        return EvalRemoteRolloutWorker.remote(
            env_fn=partial(env_fn, cfg=cfg, port=port, eval=eval), render=True
        )

    return worker_fn if not eval else eval_worker_fn


def print_logs(logs: dict) -> None:
    for k, v in logs.items():
        if np.isscalar(v):
            print(f"{k}: {v}")


def make_collectors(
    cfg: DictConfig,
) -> tuple[DistributedRolloutCollector, DistributedRolloutCollector]:
    collector = DistributedRolloutCollector(
        worker_fn=make_worker_fn(cfg, eval=False),
        num_workers=cfg.worker.num_workers,
        start_port=cfg.worker.start_port,
    )
    eval_collector = DistributedRolloutCollector(
        worker_fn=make_worker_fn(cfg, eval=True),
        num_workers=cfg.worker.num_workers,
        start_port=cfg.worker.start_port,
    )
    return collector, eval_collector


def collect_data(
    cfg: DictConfig, actor: Actor, collector: DistributedRolloutCollector
) -> None:
    os.makedirs(cfg.dataset.directory, exist_ok=True)
    trajectories = []
    iters = 0
    num_collected = 0
    while num_collected < cfg.dataset.num_sequences:
        rollouts = collector.collect_rollouts(
            batch_size=cfg.worker.num_workers,
            actors={cfg.env.agent_name: actor},
        )
        num_collected += len(rollouts)
        data = torch.stack([r.trajectory for r in rollouts], dim=0)
        trajectories.append(data)
        iters += 1
        if iters % 10 == 0:
            path = f"{cfg.dataset.directory}/trajectories-{time.time_ns()}.pt"
            data = torch.cat(trajectories, dim=0)
            data = data[cfg.env.agent_name]
            data = optree.tree_map(lambda x: x.cpu().numpy(), data.to_dict())
            np.savez_compressed(path, **data)
            trajectories = []




if __name__ == "__main__":
    main()
