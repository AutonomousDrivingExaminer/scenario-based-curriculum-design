from __future__ import annotations

import logging
from itertools import cycle
from typing import Callable

import pettingzoo
import ray

from .worker import Rollout
from ..agent.agent import Actor

EnvFn = Callable[[int, int], pettingzoo.ParallelEnv]


class DistributedRolloutCollector:
    def __init__(
            self,
            worker_fn: Callable[[int, str], ray.actor.ActorHandle],
            num_workers: int,
            available_gpus: list[str] = None,
            worker_kwargs: dict = None,
            start_port: int = 2000,
            max_restarts: int = -1
    ):
        self._worker_ctor = worker_fn
        self._num_workers = num_workers
        self._max_restarts = max_restarts
        self._start_port = start_port
        self._available_gpus = cycle(available_gpus or ["0"])
        self._workers = {}
        self._worker_kwargs = worker_kwargs or {}
        for i in range(self._num_workers):
            port = self._start_port + (i * 3)
            gpu = next(self._available_gpus)
            worker = worker_fn(port, gpu)
            self._workers[port] = worker

    def collect_rollouts(
            self,
            batch_size: int,
            policy_mapping_fn: Callable[[str], Actor],
            env_options: dict | list[dict] = None,
            render: bool | list[bool] = False
    ) -> list[Rollout]:
        restarts = [0] * len(self._workers)
        env_options = env_options or {}
        if isinstance(env_options, dict):
            env_options = [env_options] * batch_size

        if isinstance(render, bool):
            render = [render] * batch_size

        assert len(
            env_options) == batch_size, "If env_options is a list, it must have the same length as batch_size."
        rollouts = []
        running, free_workers = [], list(self._workers.keys())
        option_indices = list(range(len(env_options)))
        busy_workers = {}
        while len(rollouts) < batch_size:
            num_tasks = min(
                batch_size - len(rollouts) - len(busy_workers), len(free_workers)
            )
            if num_tasks == 0:
                logging.debug(f"Waiting for {len(running)} rollouts to finish.")
            else:
                logging.debug(
                    f"Tasks to distribute: {num_tasks}. Available workers: {len(free_workers)}."
                )
            for _ in range(num_tasks):
                port = free_workers.pop()
                opt_idx = option_indices.pop()
                logging.debug(f"Starting rollout on worker with port {port}.")
                self._workers[port].update_env.remote(env_options[opt_idx])
                future = self._workers[port].rollout.remote(policy_mapping_fn, render[opt_idx])
                busy_workers[future.hex()] = (port, opt_idx)
                running.append(future)

            finished, running = ray.wait(running)
            port, opt_idx = busy_workers.pop(finished[0].hex())
            logging.debug(f"Finished rollout on worker with port {port}.")

            try:
                rollout = ray.get(finished)[0]
                rollouts.append((rollout, opt_idx))
                logging.info(f"Rollout successful. Total number of rollouts: {len(rollouts)}.")
            except Exception as e:
                logging.warning(e)
                logging.info(f"Rollout failed for worker with port {port}. Skipping.")
                actor = self._workers[port]
                try:
                    ray.kill(actor)
                except Exception as e:
                    logging.info(e)
                self._workers[port] = self._worker_ctor(port, next(self._available_gpus))
                restarts[opt_idx] += 1
                if self._max_restarts > 0 and restarts[opt_idx] > self._max_restarts:
                    logging.warning(
                        f"Rollout for option {opt_idx} failed {restarts[opt_idx]} times. Aborting.")
                    rollouts.append((None, opt_idx))
                else:
                    option_indices.append(opt_idx)

            free_workers.append(port)
            logging.debug(f"Number of available workers: {len(free_workers)}.")

        logging.info(f"Finished collecting {len(rollouts)} rollouts.")
        rollouts = sorted(rollouts, key=lambda x: x[1])
        return [r[0] for r in rollouts]

    def close(self):
        logging.debug(f"Closing {len(self._workers)} workers.")
        for worker in self._workers.values():
            worker.close.remote()
            ray.kill(worker, no_restart=True)
