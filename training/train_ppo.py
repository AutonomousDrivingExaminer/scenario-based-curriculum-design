from __future__ import annotations

import copy
import logging
import os
from collections import defaultdict
from functools import partial

import hydra
import numpy as np
import optree
import torch
from omegaconf import DictConfig, OmegaConf
from srunner.scenariomanager.traffic_events import TrafficEventType
from tensordict import TensorDict

import adex_gym
import wandb
from adex.agent.agent import Actor
from adex.agent.ppo import PPOAgent
from adex.envs import env_factory, configs
from adex.envs.wrappers import NormalizeReward
from adex.utils import Timer, make_video
from adex_gym.envs import renderers
from adex_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from adex_gym.wrappers import ServerWrapper, BlackDeathWrapper
from adex_gym.wrappers.vectorized import VecEnvWrapper


def make_env(cfg: DictConfig, port: int, gpu: int, eval: bool = False) -> BaseScenarioEnvWrapper:
    tm_port = port - 2000 + 8000
    env = adex_gym.scenic_env(
        scenario_specification=cfg.experiment.scenario,
        agent_name_prefixes=[cfg.student.prefix],
        render_mode="rgb_array",
        resample_scenes=True,
        scenes_per_scenario=1,
        render_config=renderers.camera_pov(agent=cfg.eval.render_agent),
        traffic_manager_port=tm_port,
        params=cfg.experiment.scenario_params,
    )
    wrap_config = configs.route_following_wrappers(agent_names=env.agents)
    env = env_factory.wrap_env(env=env, eval=eval, config=wrap_config)
    env = ServerWrapper(env=env, world_port=port, gpus=[str(gpu)], wait_time=15.0)
    env = BlackDeathWrapper(env=env, default_action={
        agent: np.array([-1.0, 0.0])
        for agent in env.agents
    })
    return env


@hydra.main(version_base=None, config_path="configs", config_name="ppo_route_following")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    logging.basicConfig(
        level=logging.DEBUG if cfg.experiment.debug else logging.INFO,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s"
    )
    run = wandb.init(
        project="adex",
        name=cfg.experiment.name,
        mode=cfg.logger.mode if not cfg.experiment.debug else "offline",
        config=dict(cfg),
    )
    device = cfg.experiment.device if torch.cuda.is_available() else "cpu"

    # Create dummy env to get observation and action spaces
    dummy_env = make_env(cfg=cfg, port=2000, gpu=0, eval=False)
    agent_name = dummy_env.agents[0]
    ppo_agent = PPOAgent.from_config(
        cfg=cfg.student,
        obs_space=dummy_env.observation_space(agent_name),
        action_space=dummy_env.action_space(agent_name),
        device=device
    )

    if cfg.experiment.checkpoint_restore is not None:
        ppo_agent.load(cfg.experiment.checkpoint_restore)

    # Create vectorized environment
    start_port = cfg.workers.start_port
    gpus = cfg.workers.available_gpus
    env_fns = []
    for i in range(cfg.workers.num_workers):
        env_fn = partial(make_env, cfg=cfg, port=start_port + i * 3, gpu=gpus[i % len(gpus)])
        env_fns.append(env_fn)

    # Create vectorized environment. Terminate when all agents are done.
    vec_env = VecEnvWrapper(
        env_fns=env_fns,
        termination_fn=lambda term, trun: all(term.values()) or all(trun.values()),
        timeout=60.0
    )
    vec_env = NormalizeReward(vec_env)
    next_obs, _ = vec_env.reset()
    total_timesteps = 0
    max_avg_reward = -np.inf
    for epoch in range(cfg.experiment.epochs):
        logging.info(f"Starting epoch {epoch}.")

        if epoch % cfg.eval.interval == 1:
            obs, info = vec_env.reset()
        else:
            obs = next_obs

        actors = {
            name: ppo_agent.get_actor(eval=False)
            for name in dummy_env.agents
        }
        with Timer() as rollout_timer:
            batch, infos = rollout(
                actors=actors,
                vec_env=vec_env,
                num_steps=cfg.student.num_steps,
                initial_obs=copy.deepcopy(obs),
                render=False
            )
        next_obs_batch = TensorDict(
            {agent: batch[agent]["next_obs"] for agent in batch.sorted_keys},
            batch_size=batch.batch_size)
        next_obs = next_obs_batch[:, -1]
        batch_analysis = analyze_rollout(batch, infos, skip_first=True)

        # Stack the multi-agent batch into a single batch
        batch = torch.cat([batch[name] for name in batch.sorted_keys], dim=0)
        num_timesteps = np.prod(batch.batch_size).item()
        total_timesteps += num_timesteps

        with Timer() as update_timer:
            update_metrics = ppo_agent.update(batch)

        metrics = {
            "train/update_duration": update_timer.duration,
            "train/sample_duration": rollout_timer.duration,
            "train/total_timesteps": total_timesteps,
            "train/steps_per_second": num_timesteps / rollout_timer.duration,
            **{f"train/{k}": v for k, v in update_metrics.items()},
            **{f"train/{k}": v for k, v in batch_analysis.items()}
        }

        wandb.log(metrics, step=total_timesteps)
        print_logs(metrics)

        if epoch % cfg.eval.interval == 0:
            logging.info(f"Starting evaluation at epoch {epoch}.")
            eval_actors = {
                name: ppo_agent.get_actor(eval=True)
                for name in dummy_env.agents
            }
            mean_return, eval_metrics = eval(
                actors=eval_actors,
                vec_env=vec_env,
                num_steps=cfg.eval.num_steps
            )
            wandb.log(eval_metrics, step=total_timesteps)
            print_logs(eval_metrics)

            if mean_return > max_avg_reward:
                logging.info(f"New best model with average return {mean_return:.2f}.")
                max_avg_reward = mean_return
                best_path = os.path.join(run.dir, "best_ppo_model.pt")
                torch.save(ppo_agent.checkpoint(), best_path)

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
                **ppo_agent.checkpoint(),
                "run_id": run.id,
                "epoch": epoch,
                "best_avg_return": max_avg_reward,
                "total_timesteps": total_timesteps
            }
            torch.save(ckpt, ckpt_path)


def eval(actors: dict[str, Actor], vec_env: VecEnvWrapper, num_steps: int) -> tuple[float, dict]:
    eval_batch, eval_infos, videos = rollout(
        actors=actors,
        vec_env=vec_env,
        num_steps=num_steps,
        render=True
    )
    videos = make_video(videos, fps=20)
    analysis = analyze_rollout(eval_batch, eval_infos)

    ep_returns = analysis["ep_returns"]
    ep_events = analysis["ep_events"]
    avg_return = np.mean(np.array([ep_returns[agent] for agent in ep_returns]))

    metrics = defaultdict(float)
    for agent in ep_events:
        agent_events = ep_events[agent]
        for event in agent_events:
            metrics[event] += agent_events[event] / len(eval_batch.sorted_keys)
    metrics = dict(metrics)
    metrics = {
        f"eval/{k.name}": metrics.get(k.name, 0.0)
        for k in TrafficEventType
    }
    metrics["eval/mean_return"] = avg_return
    metrics["eval/videos"] = videos
    return avg_return, metrics


def rollout(
        actors: dict[str, Actor],
        vec_env: VecEnvWrapper,
        num_steps: int = None,
        render: bool = False,
        initial_obs: list[dict[str, np.ndarray]] = None
) -> tuple[TensorDict, list] | tuple[TensorDict, list, np.ndarray]:
    if initial_obs is None:
        obs, info = vec_env.reset()
    else:
        obs = initial_obs

    agents = list(obs[0].keys())
    batch, b_infos, frames = [], [], []
    for t in range(num_steps):
        obs = torch.stack([TensorDict(o, batch_size=()) for o in obs], dim=0)
        actions, actor_outputs = {}, {}
        for name in obs.keys():
            actor = actors[name]
            action, actor_output = actor(obs[name])
            actions[name] = action
            actor_outputs[name] = actor_output
        actions = [dict(zip(actions, t)) for t in zip(*actions.values())]
        true_next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

        if render:
            imgs = vec_env.render()
            frames.append([img.astype(np.uint8) for img in imgs])

        next_obs = []
        for i, info in enumerate(infos):
            if final := info.get("__final__", None):
                next_obs.append(final["obs"])
            else:
                next_obs.append(true_next_obs[i])

        b_infos.append(infos)
        next_obs = torch.stack([TensorDict(o, batch_size=()) for o in next_obs], dim=0)
        rewards = torch.stack([TensorDict(r, batch_size=()) for r in rewards], dim=0)
        terminateds = torch.stack([TensorDict(t, batch_size=()) for t in terminateds], dim=0)
        truncateds = torch.stack([TensorDict(t, batch_size=()) for t in truncateds], dim=0)

        step = TensorDict({
            agent: {
                "obs": obs[agent],
                "next_obs": next_obs[agent],
                "reward": rewards[agent],
                "terminated": terminateds[agent],
                "truncated": truncateds[agent],
                **actor_outputs[agent]
            } for agent in agents
        }, batch_size=len(obs))

        batch.append(step)
        obs = true_next_obs.copy()
    batch = torch.stack(batch, dim=1)
    if render:
        frames = np.stack(frames, axis=0).transpose(1, 0, 2, 3, 4) if render else None
        frames = np.concatenate(frames, axis=0)
        return batch, b_infos, frames
    return batch, b_infos

def analyze_rollout(trajectory: TensorDict, infos: list, skip_first: bool = False) -> dict:
    ep_returns, ep_returns_unnormalized, ep_lengths, ep_events = [], [], [], []
    infractions = []
    for i in range(len(infos[0])):
        prev_idx = 0
        for t in range(len(infos)):
            info = infos[t][i]
            if not "__final__" in info:
                continue
            if skip_first and prev_idx == 0:
                prev_idx = t + 1
                continue

            final = info["__final__"]
            ep_return = {
                agent: trajectory[agent]["reward"][i, prev_idx:t + 1].sum().item()
                for agent in trajectory.keys()
            }
            unnormed_return = {
                agent: np.sum([i[agent]["original_reward"] for i in infos[prev_idx:t + 1][i]])
                for agent in trajectory.keys()
            }
            ep_returns.append(ep_return)
            ep_returns_unnormalized.append(unnormed_return)
            ep_lengths.append(t - prev_idx + 1)
            event_summary = {}
            for agent in trajectory.keys():
                events = final["info"][agent]["events"]
                agent_events = defaultdict(int)
                for event in events:
                    type = event["event"]
                    if type == "ROUTE_COMPLETION":
                        agent_events[type] = max(agent_events[type], event.get("route_completed", 0))
                    else:
                        agent_events[type] += 1
                event_summary[agent] = dict(agent_events)
            ep_events.append(event_summary)
            prev_idx = t + 1
    if len(ep_returns) == 0:
        return {}

    agent_events = defaultdict(lambda: defaultdict(int))
    for events in ep_events:
        for agent in events:
            for type, count in events[agent].items():
                agent_events[agent][type] += count / len(ep_events)


    return {
        "ep_returns": optree.tree_map(lambda *xs: np.mean(xs), *ep_returns),
        "ep_returns_unnormalized": optree.tree_map(lambda *xs: np.mean(xs),
                                                   *ep_returns_unnormalized),
        "ep_lengths": np.mean(ep_lengths),
        "ep_events": dict(agent_events)
    }





def print_logs(logs: dict) -> None:
    for k, v in logs.items():
        if np.isscalar(v):
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
