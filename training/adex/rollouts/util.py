from __future__ import annotations

from collections import defaultdict

import numpy as np

from adex import utils
from adex.rollouts.worker import Rollout


def get_eval_metrics(rollouts: list[Rollout]) -> dict:
    metrics = defaultdict(list)
    videos = []
    for episode in rollouts:
        infos = episode.infos[0]  # Only one list of infos per episode
        for agent in episode.trajectory.keys():
            returns = episode.trajectory[agent, "reward"].sum().numpy().item()
            completion = infos[-1][agent].get("completion", 0.0)
            metrics["return"].append(returns)
            metrics["route_completion"].append(completion)
        if episode.videos is not None:
            videos.append(episode.videos[0])
    metrics = dict(metrics)
    summary = {}
    for k, v in metrics.items():
        summary[f"mean_{k}"] = np.mean(v)
        summary[f"std_{k}"] = np.std(v)

    if len(videos) > 0:
        videos = np.concatenate(videos, axis=0).astype(np.uint8)
        summary["video"] = utils.make_video(videos, fps=10)

    return summary


def get_batch_metrics(rollouts: list[Rollout]) -> dict:
    metrics = defaultdict(list)
    infraction_count = defaultdict(int)
    for rollout in rollouts:
        for agent in rollout.trajectory.keys():
            terminated = rollout.trajectory[agent, "terminated"]
            rewards = rollout.trajectory[agent, "reward"]
            term_idx = terminated.argwhere().squeeze(-1)
            terminals = term_idx.tolist()
            start = 0
            for end in terminals:
                ep_returns = rewards[start:end].sum(-1).numpy().item()
                metrics["return"].append(ep_returns)
                metrics["episode_length"].append(end - start)
                start = end
        for ep_infos in rollout.infos:
            last_info = ep_infos[-1]
            for agent in rollout.trajectory.keys():
                if "completion" in last_info[agent]:
                    metrics["route_completion"].append(last_info[agent]["completion"])
                for infraction in last_info[agent].get("infractions", []):
                    infraction_count[infraction] += 1

    metrics = dict(metrics)
    keys = list(metrics.keys())
    for k in keys:
        v = metrics[k]
        metrics[f"mean_{k}"] = np.mean(v)
        metrics[f"std_{k}"] = np.std(v)
        metrics.pop(k)

    infraction_count = dict(infraction_count)
    num_episodes = sum([len(rollout.infos) for rollout in rollouts])
    for k, v in infraction_count.items():
        metrics[f"{k.lower()}_per_episode"] = v / num_episodes
    return metrics
