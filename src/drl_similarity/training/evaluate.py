"""Deterministic evaluation suites over fixed seed sets."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from drl_similarity.agents.base import LoadedAgent
from drl_similarity.agents.registry import get_algorithm_adapter
from drl_similarity.config import (
    EvalSuiteConfig,
    EvaluationConfig,
    ExperimentConfig,
    ResolvedConfig,
)
from drl_similarity.environments.atari import make_environment
from drl_similarity.errors import ConfigError
from drl_similarity.study_io import load_run, resolve_device


# Columns that are constant for a run
RUN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "training_seed",
    "algorithm_id",
    "adapter_id",
    "policy_id",
    "backend_version",
    "environment_fingerprint",
    "code_commit",
    "config_hash",
)


def evaluate_agent(
    agent: LoadedAgent,
    env_factory: Callable[[int], gym.Env],
    suite: EvalSuiteConfig,
    evaluation: EvaluationConfig,
) -> list[dict[str, Any]]:
    """Play ``episodes_per_seed`` episodes per seed; one row of raw outcomes per episode."""
    rows: list[dict[str, Any]] = []

    for seed in suite.seeds:
        env = env_factory(seed)
        try:
            for episode_index in range(suite.episodes_per_seed):
                # The first reset pins the seed; later episodes continue its RNG stream
                observation, info = env.reset(seed=seed if episode_index == 0 else None)
                episode = _play_episode(agent, env, observation, info, evaluation)
                rows.append({"eval_seed": seed, "episode_index": episode_index, **episode})
        finally:
            env.close()

    return rows


def run_final_evaluation(resolved: ResolvedConfig) -> Path:
    """Measure the selected checkpoints on the disjoint final suite (replaces old final rows)."""
    experiment = resolved.value
    if not isinstance(experiment, ExperimentConfig):
        raise ConfigError("evaluate requires an experiment config")

    run = load_run(Path(experiment.artifacts.root) / experiment.id)
    eval_path = run.run_dir / "evaluations.parquet"
    table = pd.read_parquet(eval_path)
    meta = table.iloc[0][list(RUN_COLUMNS)].to_dict()
    adapter = get_algorithm_adapter(run.algorithm.adapter)
    device = resolve_device(experiment.device)

    rows: list[dict[str, Any]] = []
    for checkpoint in run.selected.values():
        agent = adapter.load(checkpoint.path, run.algorithm, None, device=device)
        episodes = evaluate_agent(
            agent,
            lambda seed: make_environment(run.environment, seed=seed),
            experiment.evaluation.final,
            experiment.evaluation,
        )
        rows += [
            {
                **meta,
                "checkpoint_id": checkpoint.checkpoint_id,
                "training_step": checkpoint.training_step,
                "suite": "final",
                **episode,
            }
            for episode in episodes
        ]

    kept = table[table["suite"] != "final"]
    write_evaluations(eval_path, pd.concat([kept, pd.DataFrame(rows)], ignore_index=True))

    return eval_path


def write_evaluations(path: Path, table: pd.DataFrame) -> None:
    """Replace the evaluations table atomically."""
    tmp = path.with_suffix(".tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)


def _play_episode(
    agent: LoadedAgent,
    env: gym.Env,
    observation: np.ndarray,
    info: dict[str, Any],
    evaluation: EvaluationConfig,
) -> dict[str, Any]:
    """Roll out one episode and count declared game events."""
    wrapped_return, length, lives_lost, goals = 0.0, 0, 0, 0
    lives = info.get("lives")
    done = False
    started = time.perf_counter()

    while not done and length < evaluation.max_agent_steps:
        action = agent.predict(observation, deterministic=evaluation.deterministic_actions)
        observation, reward, terminated, truncated, info = env.step(action)
        wrapped_return += float(reward)
        length += 1
        goals += reward > 0

        # A drop in the emulator's life counter is an event, not an episode boundary
        new_lives = info.get("lives", lives)
        lives_lost += lives is not None and new_lives < lives
        lives = new_lives
        done = terminated or truncated

    return {
        "return": _raw_return(env, wrapped_return),
        "length": length,
        "lives_lost": int(lives_lost),
        "goals": int(goals),
        "wall_time": time.perf_counter() - started,
    }


def _raw_return(env: gym.Env, wrapped_return: float) -> float:
    """Unclipped return from ``record_episode_statistics`` when that wrapper is present."""
    try:
        return float(env.get_wrapper_attr("episode_returns"))
    except AttributeError:
        return wrapped_return
