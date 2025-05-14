from __future__ import annotations

import logging
import os
import time

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

import adex_gym
import wandb
from adex.agent.ppo import PPOAgent
from adex.envs import env_factory
from adex.envs.configs import route_following_wrappers
from adex.rollouts import get_eval_metrics, get_batch_metrics
from adex_gym.envs import renderers
from adex_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from adex_gym.wrappers import ServerWrapper
from training.adex import utils


@hydra.main(
    version_base=None, config_path="configs", config_name="independent_ppo_negotiation"
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

    dummy_env = make_env(cfg=cfg, port=2000, gpu=0, eval=False)

    # note: here the agent_name denotes the prefix of multiple agents which we assume have same actions/observations
    # so we create the agent based on the first agent's name
    agent_name = dummy_env.possible_agents[0]


    assert all(dummy_env.observation_space(agent_name) == dummy_env.observation_space(agent) for agent in
               dummy_env.possible_agents), "expected all agents to have same observation space"
    assert all(dummy_env.action_space(agent_name) == dummy_env.action_space(agent) for agent in
               dummy_env.possible_agents), "expected all agents to have same action space"

    agent = PPOAgent.from_config(
        cfg=cfg.agent,
        obs_space=dummy_env.observation_space(agent_name),
        action_space=dummy_env.action_space(agent_name),
        device=device,
    )

    if cfg.experiment.checkpoint_restore is not None:
        agent.load(cfg.experiment.checkpoint_restore)

    collector = utils.make_distributed_rollout_collector(
        worker_cfg=cfg.workers.train,
        worker_fn=lambda port, gpu: utils.make_fixed_length_worker(
            env_fn=lambda: make_env(cfg=cfg, port=port, gpu=gpu, eval=False),
            num_steps=cfg.agent.num_steps,
            remote=True,
            render=False,
            reset_between_rollouts=cfg.workers.train.reset_between_rollouts,
        ),
    )

    eval_collector = utils.make_distributed_rollout_collector(
        worker_cfg=cfg.workers.eval,
        worker_fn=lambda port, gpu: utils.make_episode_worker(
            env_fn=lambda: make_env(cfg=cfg, port=port, gpu=gpu, eval=True),
            remote=True,
            render=True,
        ),
    )

    total_timesteps = 0
    max_avg_reward = -np.inf
    for epoch in range(cfg.experiment.epochs):
        start = time.time()
        actor_critic = agent.get_actor(eval=False)
        rollouts = collector.collect_rollouts(
            batch_size=cfg.agent.batch_size, policy_mapping_fn=lambda agent_id: actor_critic
        )
        sample_duration = time.time() - start
        batch: TensorDict = torch.stack([r.trajectory[agent_id] for r in rollouts for agent_id in r.trajectory.keys()],
                                        dim=0)
        num_timesteps = np.prod(batch.batch_size).item()
        total_timesteps += num_timesteps

        start = time.time()
        update_metrics = agent.update(batch)
        update_duration = time.time() - start

        batch_metrics = get_batch_metrics(rollouts)

        metrics = {
            "train/update_duration": update_duration,
            "train/sample_duration": sample_duration,
            "train/total_timesteps": total_timesteps,
            "train/steps_per_second": num_timesteps / sample_duration,
        }
        metrics.update({f"train/{k}": v for k, v in update_metrics.items()})

        metrics.update({f"train/{k}": v for k, v in batch_metrics.items()})

        if epoch % cfg.experiment.eval_interval == 0:
            logging.info(f"Starting evaluation at epoch {epoch}.")
            eval_actor = agent.get_actor(eval=True)
            eval_rollouts = eval_collector.collect_rollouts(
                batch_size=cfg.experiment.eval_episodes,
                policy_mapping_fn=lambda agent_id: eval_actor,
            )
            eval_metrics = get_eval_metrics(eval_rollouts)
            metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
            avg_return = np.mean(eval_metrics["mean_return"])
            if avg_return > max_avg_reward:
                logging.info(f"New best model with average return {avg_return:.2f}.")
                max_avg_reward = avg_return
                best_path = os.path.join(run.dir, "best_ppo_model.pt")
                torch.save(agent.checkpoint(), best_path)

        if epoch % cfg.experiment.checkpoint_interval == 0:
            checkpoint_dir = os.path.join(run.dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{epoch:06d}.pt")
            logging.info(f"Saving checkpoint at epoch {epoch}: {ckpt_path}.")
            checkpoints = os.listdir(checkpoint_dir)
            if len(checkpoints) == cfg.experiment.num_checkpoints:
                checkpoints = sorted(checkpoints)
                checkpoint_to_remove = checkpoints[0]
                logging.info(f"Removing oldest checkpoint: {checkpoint_to_remove}.")
                os.remove(os.path.join(checkpoint_dir, checkpoint_to_remove))
            ckpt = {
                **agent.checkpoint(),
                "run_id": run.id,
                "epoch": epoch,
                "best_avg_return": max_avg_reward,
                "total_timesteps": total_timesteps,
            }
            torch.save(ckpt, ckpt_path)

        wandb.log(metrics)
        print_logs(metrics)


def make_env(cfg: DictConfig, port: int, gpu: int, eval: bool = False) -> BaseScenarioEnvWrapper:
    tm_port = 8000 + port - 2000
    render_cfg = renderers.camera_pov(agent=cfg.render.agent, width=cfg.render.width, height=cfg.render.height)
    env = adex_gym.scenic_env(
        scenario_specification=cfg.experiment.scenario,
        agent_name_prefixes=[cfg.agent.name],
        render_mode="rgb_array",
        resample_scenes=True,
        scenes_per_scenario=8,
        render_config=render_cfg,
        traffic_manager_port=tm_port
    )
    wrap_config = route_following_wrappers(agent_names=env.possible_agents)
    env = env_factory.wrap_env(env=env, eval=eval, config=wrap_config)
    env = ServerWrapper(env=env, world_port=port, gpus=[str(gpu)])

    #default_action = {agent: np.array([-1.0, 0.0], dtype=np.float32) for agent in env.agents}
    #env = BlackDeathWrapper(env=env, default_action=default_action)

    return env


def print_logs(logs: dict) -> None:
    for k, v in logs.items():
        if np.isscalar(v):
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
