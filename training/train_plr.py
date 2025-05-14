from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import random
from collections import defaultdict
from functools import partial

import hydra
import numpy as np
import optree
import ray
import torch
from flatten_dict import flatten
from omegaconf import DictConfig
from srunner.scenariomanager.traffic_events import TrafficEventType
from tensordict import TensorDict

import adex_gym
import wandb
from adex.agent import PPOAgent
from adex.buffers import PrioritizedLevelReplayBuffer
from adex.envs import configs, env_factory
from adex.envs.wrappers import FrameSkipWrapper
from adex.rollouts.worker import Rollout
from adex.sampler import OptunaEnvParamSampler, EnvConfiguration, RandomEnvParamSampler, \
    EnvParamSampler
from adex.utils import Timer, make_video
from adex_gym.agents.meta_actions_agent import Action
from adex_gym.envs import renderers
from adex_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from adex_gym.wrappers import ServerWrapper, BlackDeathWrapper, MetaActionWrapper

torch.multiprocessing.set_sharing_strategy("file_system")


def to_dict(x):
    return TensorDict(
        source=optree.tree_map(lambda *xs: np.stack(xs, axis=0), *x),
        batch_size=(len(x),)
    )


def to_lists(x):
    return [dict(zip(x, t)) for t in zip(*x.values())]


@dataclasses.dataclass
class RolloutState:
    WAIT_FOR_INIT = 0
    WAIT_FOR_ROLLOUT = 1
    WAIT_FOR_LEVEL = 2
    FINISHED = 3
    FAILED = 4


class Worker:

    def __init__(self, env_fn):
        self.env_fn = env_fn
        self.env = env_fn()

    def collect_episode(self, actors, level: EnvConfiguration, eval: bool = False):
        obs, info = self.env.reset(options=level.options)
        episode = []
        done = False
        while not done:
            obs = TensorDict(obs, batch_size=())
            actions, actor_outputs = {}, {}
            for id, actor in actors.items():
                action, actor_output = actor(obs[id])
                actions[id] = action
                actor_outputs[id] = actor_output
            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            done = all(terminated.values())
            logging.info(f"observations: {obs[next(iter(obs.keys()))]}")
            step = TensorDict({
                agent: TensorDict({
                    "obs": obs[agent],
                    "action": actions[agent],
                    "reward": reward[agent],
                    "next_obs": next_obs[agent],
                    "terminated": terminated[agent],
                    "truncated": truncated[agent],
                    **actor_outputs[agent],
                }, batch_size=())
                for agent in obs.keys()
            }, batch_size=())
            frame = self.env.render() if eval else None
            episode.append((step, info, frame))
            obs = next_obs
        # self.env.close()
        return episode


class RolloutCollector:

    def __init__(self, env_fns, buffer: PrioritizedLevelReplayBuffer, env_sampler: EnvParamSampler):
        self.env_fns = env_fns
        self.workers = [
            ray.remote(Worker).remote(fn)
            for fn
            in env_fns
        ]
        self.episodes = defaultdict(list)
        self.episode_count = 0
        self.buffer = buffer
        self.env_sampler = env_sampler

    def collect(self, agent: PPOAgent, actors, num_steps, eval=False):
        batch, episodes = [], []
        busy_workers = {}
        free_workers = list(range(len(self.workers)))
        steps = 0
        pending, finished = [], []
        while steps < num_steps:
            for i in range(len(free_workers)):
                if eval:
                    logging.info("Sampling new level for evaluation.")
                    level = self.env_sampler()
                    sampled = True
                else:
                    logging.info("Sampling new level for training.")
                    level, sampled = self.buffer.get_next_level()
                idx = free_workers.pop()
                future = self.workers[idx].collect_episode.remote(actors, level, eval=eval)
                busy_workers[future.hex()] = (idx, level, sampled)
                pending.append(future)

            finished, pending = ray.wait(pending)
            idx, finished_level, sampled = busy_workers[finished[0].hex()]
            try:
                episode = ray.get(finished)[0]
                logging.info(f"Finished rollout on worker {idx} for level {finished_level.id} ({'sampled' if sampled else 'generated'}).")
                steps += len(episode)
                timesteps, infos, frames = list(zip(*episode))
                trajectory = torch.stack(timesteps, dim=0)
                if eval:
                    frames = np.stack(frames, axis=0)
                rollout = Rollout(
                    trajectory=trajectory,
                    infos=infos,
                    videos=frames,
                    env_configs=finished_level
                )
                score, rollout = self.compute_level_metrics(agent, rollout)
                if not eval:
                    logging.info(f"Updating buffer with level {finished_level.id} with score {score}.")
                    self.buffer.update(rollout.env_configs, score)
                episodes.append(rollout)
                if sampled:
                    batch.append(trajectory)
            except Exception as e:
                logging.info(f"Worker {idx} failed for level {finished_level.id}. Reason: {e}")
                actor = self.workers[idx]
                try:
                    ray.kill(actor)
                except Exception as e:
                    logging.info(e)
                self.workers[idx] = ray.remote(Worker).remote(self.env_fns[idx])

            free_workers.append(idx)
        batch = torch.cat(batch, dim=0)
        batch = batch[:num_steps]
        return batch, episodes

    def compute_level_metrics(self, agent: PPOAgent, rollout: Rollout):
        steps = rollout.trajectory["student"]
        steps = steps.to(agent.device, dtype=torch.float32).unsqueeze(0)
        steps = agent.compute_advantages(
            trajectory=steps,
            gamma=agent.gamma,
            lmbda=agent.lmbda
        )
        if rollout.env_configs.stats is None:
            rollout.env_configs.stats = {}
        trajectory = steps.cpu()
        episode_return = trajectory["reward"].sum().numpy().item()
        prev_return = rollout.env_configs.stats.get("max_return", -np.inf)
        R_max = max(episode_return, prev_return)
        rollout.env_configs.stats["max_return"] = R_max
        values = trajectory["value"].squeeze(0)
        score = R_max - values
        score = score.mean().numpy().item()
        rollout.env_configs.stats["score"] = score
        return score, rollout


@hydra.main(version_base=None, config_path="configs", config_name="ued_route_following")
def main(cfg: DictConfig) -> None:
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    seed = cfg.experiment.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # full_episodes = torch.load("full_episodes.pt")
    # get_episode_metrics(full_episodes)

    ray.init(local_mode=False)

    logging.basicConfig(level=logging.DEBUG)
    run = wandb.init(
        project="adex",
        name=cfg.experiment.name,
        mode=cfg.logger.mode if not cfg.experiment.debug else "offline",
        config=dict(cfg)
    )
    cfg.eval.video_dir = f"/tmp/adex/{run.id}/videos"
    os.makedirs(cfg.eval.video_dir, exist_ok=True)

    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")

    if cfg.env_generator.sampler == "random":
        logging.info("Using random sampler.")
        sampler = RandomEnvParamSampler(
            path=cfg.env_generator.scenario,
            max_vehicles=cfg.env_generator.max_vehicles
        )
    else:
        logging.info("Using optuna sampler.")
        sampler = OptunaEnvParamSampler(
            path=cfg.env_generator.scenario,
            max_vehicles=cfg.env_generator.max_vehicles
        )

    level_buffer = PrioritizedLevelReplayBuffer(
        replay_rate=cfg.env_buffer.replay_rate,
        p=cfg.env_buffer.p,
        temperature=cfg.env_buffer.temperature,
        generator=sampler,
        max_size=cfg.env_buffer.max_size,
        update_sampler=cfg.env_buffer.update_sampler
    )

    dummy_env = make_env(
        seed=cfg.experiment.seed,
        discrete=cfg.student.discrete,
        scenario=cfg.env_generator.scenario,
        agent=cfg.student.name,
        port=cfg.workers.start_port,
        gpu=0,
        image=f"carlasim/carla:{cfg.experiment.carla_version}"
    )
    agent_name = dummy_env.agents[0]
    ppo_agent = PPOAgent.from_config(
        cfg=cfg.student,
        obs_space=dummy_env.observation_space(agent_name),
        action_space=dummy_env.action_space(agent_name),
        device=device
    )

    if cfg.experiment.checkpoint_restore is not None:
        ppo_agent.load(cfg.experiment.checkpoint_restore)

    env_fns = []
    for i in range(cfg.workers.num_workers):
        port = cfg.workers.start_port + (i * 3)
        gpu = cfg.workers.available_gpus[i % len(cfg.workers.available_gpus)]
        env_fn = partial(
            make_env,
            seed=cfg.experiment.seed + port,
            discrete=cfg.student.discrete,
            scenario=cfg.env_generator.scenario,
            agent=cfg.student.name,
            port=port,
            gpu=gpu,
            wait_time=cfg.workers.wait_time,
            image=f"carlasim/carla:{cfg.experiment.carla_version}"
        )
        env_fns.append(env_fn)

    collector = RolloutCollector(
        env_fns=env_fns,
        buffer=level_buffer,
        env_sampler=RandomEnvParamSampler(
            path=cfg.env_generator.scenario,
            max_vehicles=cfg.env_generator.max_vehicles
        )
    )

    total_timesteps = 0

    best_reward = -np.inf
    for epoch in range(cfg.experiment.epochs):
        logging.info(f"Epoch {epoch}.")
        metrics = {}
        actors = {agent: ppo_agent.get_actor(eval=False) for agent in dummy_env.agents}
        with Timer() as rollout_timer:
            batch, full_episodes = collector.collect(
                agent=ppo_agent,
                actors=actors,
                num_steps=cfg.student.num_steps * cfg.student.batch_size,
                eval=False,
            )

        batch_size = np.prod(batch.batch_size)
        save_params(full_episodes, timestep=epoch, run=run, prefix="train")
        logging.info(f"Updating agent with {batch_size} steps.")
        batch = batch.to(device, dtype=torch.float32)
        train_metrics = ppo_agent.update(trajectory=batch[agent_name].unsqueeze(0))
        metrics.update(train_metrics)

        # collect stats
        total_timesteps += batch_size
        episode_metrics = get_episode_metrics(full_episodes)
        buffer_metrics = get_buffer_metrics(level_buffer)
        metrics.update(episode_metrics)
        metrics.update(buffer_metrics)
        metrics.update({
            "steps_per_second": sum([len(e.infos) for e in full_episodes]) / rollout_timer.duration,
            "total_timesteps": total_timesteps
        })

        if epoch % cfg.eval.interval == 0:
            logging.info("Evaluating agent.")
            actors = {agent: ppo_agent.get_actor(eval=True) for agent in dummy_env.agents}
            _, episodes = collector.collect(
                agent=ppo_agent,
                actors=actors,
                eval=True,
                num_steps=cfg.student.num_steps,
            )

            save_params(episodes, timestep=epoch, run=run, prefix="eval")
            videos, birdviews = [], []
            env_configs = {}
            for episode in episodes:
                videos.append(episode.videos)
                bv = episode.trajectory["student"]["obs"]["birdview"].cpu().numpy().transpose(0, 2, 3, 1)
                birdviews.append(bv[..., -1] * 255)
                env_configs[episode.env_configs.id] = episode.env_configs

            video = make_video(path=f"{cfg.eval.video_dir}/video.mp4", frames=np.concatenate(videos, axis=0), fps=10)
            bv = np.concatenate(birdviews, axis=0)
            obs_video = make_video(path=f"{cfg.eval.video_dir}/bv.mp4", frames=bv, fps=10)
            eval_metrics = get_episode_metrics(episodes)
            best_reward = max(best_reward, eval_metrics["episode_reward"])
            metrics.update({
                **{"eval/" + k: v for k, v in eval_metrics.items()},
                "eval/video": video,
                "eval/birdview": obs_video
            })

        log(metrics, step=total_timesteps)

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
                **level_buffer.checkpoint(),
                "run_id": run.id,
                "epoch": epoch,
                "best_avg_return": best_reward,
                "total_timesteps": total_timesteps
            }
            torch.save(ckpt, ckpt_path)


def get_buffer_metrics(buffer: PrioritizedLevelReplayBuffer) -> dict:
    buffer_metrics = buffer.get_buffer_stats()
    param_stats = defaultdict(float)
    buffer_size = buffer_metrics["size"]
    for config in buffer.get_level_stats().values():
        params = flatten(config["params"])
        for param, v in params.items():
            if param == "map":
                continue
            param_stats[f"params/{'/'.join(param)}/{v}"] += 1 / buffer_size
    buffer_metrics.update(param_stats)
    return buffer_metrics


def get_episode_metrics(rollouts: list[Rollout]) -> dict:
    def get_route_completion(last_info):
        agents = [a for a in last_info.keys() if not a.startswith("__")]
        completions = {}
        for agent in agents:
            if completion := last_info[agent].get("completion", None):
                completions[agent] = completion
                continue

            events = filter(lambda x: x["event"] == "ROUTE_COMPLETION", last_info[agent]["events"])
            route_completion_event = list(events)
            if len(route_completion_event) > 0:
                completions[agent] = route_completion_event[0].get("route_completed", 0.0)
        return completions

    def get_infractions(last_info):
        agents = [a for a in last_info.keys() if not a.startswith("__")]
        infractions = {t.name: 0 for t in TrafficEventType}
        for agent in agents:
            events = last_info[agent]["events"]
            for event in filter(lambda x: x["event"] != "ROUTE_COMPLETION", events):
                infractions[event["event"]] += 1
        return infractions

    def get_params(config):
        params = flatten(config.params)
        stats = defaultdict(int)
        for k, v in params.items():
            if k == "map":
                continue
            stats[f"params/{'-'.join(k)}-{v}"] += 1
        return dict(stats)

    metrics = defaultdict(list)
    for episode in rollouts:
        metrics["episode_length"].append(len(episode.trajectory))
        last_info = episode.infos[-1]

        agents = list(episode.trajectory.keys())
        mean_return = np.mean([episode.trajectory[agent, "reward"].sum() for agent in agents])
        route_completion = np.mean(list(get_route_completion(last_info).values()))
        metrics["route_completion"].append(route_completion)
        metrics["episode_reward"].append(mean_return)

        infractions = get_infractions(last_info)
        for infraction in infractions:
            metrics[f"infractions/{infraction}"].append(infractions[infraction])

        config = episode.env_configs
        params = get_params(episode.env_configs)
        metrics.update({
            "params/score": config.stats["score"],
            "params/max_return": config.stats["max_return"],
            **{f"{k}": v for k, v in optree.tree_map(lambda x: x / len(rollouts), params).items()}
        })

    episode_stats = {
        k: np.mean(v) for k, v in metrics.items()
    }
    return episode_stats


def log(metrics: dict, step: int = None):
    wandb.log(metrics, step=step)
    for k, v in metrics.items():
        if np.isscalar(v):
            print(f"{k} : {v:.3f}")


def save_params(episodes: list[Rollout], timestep, run, prefix=""):
    params, scenes = [], []
    for episode in episodes:
        config = copy.deepcopy(episode.env_configs.__dict__)
        scene = config.pop("options")
        params.append(config)
    params_dir = os.path.join(run.dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    prefix = f"{prefix}-" if prefix else ""
    params_path = os.path.join(params_dir, f"{prefix}params-{timestep:06d}.json")
    with open(params_path, "w") as f:
        json.dump(params, f)


def make_env(scenario: str, discrete: bool, agent: str, seed: int, image: str, port: int = 2000,
             gpu: int = 0, wait_time=40.0) -> BaseScenarioEnvWrapper:
    tm_port = port - 2000 + 8000
    env = adex_gym.scenic_env(
        seed=seed,
        scenario_specification=scenario,
        agent_name_prefixes=[agent],
        render_mode="rgb_array",
        resample_scenes=True,
        scenes_per_scenario=1,
        render_config=renderers.camera_pov(agent=agent, width=512, height=512),
        traffic_manager_port=tm_port,
    )
    wrap_config = configs.route_following_wrappers(agent_names=env.agents)

    if discrete:
        wrap_config.pop("action_normalization")
        wrap_config.pop("frame_skip")
        env = env_factory.wrap_env(env=env, eval=True, config=wrap_config)
        env = MetaActionWrapper(
            env=env,
            action_frequency=20,
            planner_options={
                "target_speed": 35,
                "ignore_vehicles": True,
                "ignore_traffic_light": True,
            },
            agent_names=[agent]
        )
        env = FrameSkipWrapper(env=env, skip=20, render=True)
        env = BlackDeathWrapper(env=env, default_action={
            agent: Action.STOP
            for agent in env.agents
        })
    else:
        skip_config = wrap_config.pop("frame_skip")
        env = FrameSkipWrapper(env=env, **skip_config, render=True)
        env = env_factory.wrap_env(env=env, eval=True, config=wrap_config)
        env = BlackDeathWrapper(env=env, default_action={
            agent: np.array([-1.0, 0.0])
            for agent in env.agents
        })
    env = ServerWrapper(
        env=env,
        world_port=port,
        gpus=[str(gpu)],
        wait_time=wait_time,
        server_kwargs={"image": image}
    )


    return env


if __name__ == "__main__":
    main()
