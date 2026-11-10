"""Algorithm-neutral training runner: build or resume, checkpoint, evaluate, record provenance.

Run directory layout (``<artifacts.root>/<experiment id>/``)::

    resolved-config.yaml            fully resolved experiment
    environment_fingerprint.json    observation + runtime fingerprints
    versions.json                   package versions and git commit
    evaluations.parquet             one row per checkpoint and evaluation episode
    checkpoints/checkpoint_<step>.zip|.json   backend checkpoint + metadata sidecar
    manifest.json                   checksums of everything above
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
import yaml

from drl_similarity.agents.base import TrainableAgent
from drl_similarity.agents.registry import get_algorithm_adapter
from drl_similarity.artifacts.manifest import (
    ArtifactManifest,
    ArtifactOutput,
    capture_package_versions,
    sha256_file,
    try_git_commit,
)
from drl_similarity.config import (
    REPO_ROOT,
    AlgorithmConfig,
    EnvironmentConfig,
    ExperimentConfig,
    ResolvedConfig,
    write_resolved_config,
)
from drl_similarity.environments.atari import compute_environment_fingerprints, make_environment
from drl_similarity.errors import CheckpointError, ConfigError
from drl_similarity.logging import get_logger, log_extra
from drl_similarity.study_io import checkpoint_file, resolve_device
from drl_similarity.training.callbacks import CheckpointEvaluationCallback
from drl_similarity.training.evaluate import evaluate_agent, write_evaluations


VERSION_PACKAGES = ["numpy", "torch", "gymnasium", "ale_py", "stable_baselines3", "pandas"]


def run_training(resolved: ResolvedConfig, *, resume: Path | None = None) -> Path:
    """Train (or resume) one experiment and return its run directory."""
    experiment = resolved.value
    if not isinstance(experiment, ExperimentConfig):
        raise ConfigError("train requires an experiment config")
    environment, algorithm = experiment.environment, experiment.algorithm
    assert isinstance(environment, EnvironmentConfig) and isinstance(algorithm, AlgorithmConfig)

    logger = get_logger("train")
    extra = log_extra(stage="train", config_hash=resolved.config_hash)
    run_dir = Path(experiment.artifacts.root) / experiment.id
    eval_path = run_dir / "evaluations.parquet"
    _check_run_dir(run_dir, resolved, resume)

    # Validate before allocating a replay buffer or GPU memory
    adapter = get_algorithm_adapter(algorithm.adapter)
    fingerprints = compute_environment_fingerprints(environment)
    _seed_everything(experiment)
    env = make_environment(environment, seed=experiment.seed)

    try:
        device = resolve_device(experiment.device)
        agent = adapter.build(algorithm, env, seed=experiment.seed, device=device, resume=resume)

        # Persist provenance before the first step
        repro = experiment.reproducibility
        commit = try_git_commit(REPO_ROOT) if repro.capture_git else None
        versions = capture_package_versions(VERSION_PACKAGES) if repro.capture_packages else {}
        run_dir.mkdir(parents=True, exist_ok=True)
        write_resolved_config(run_dir / "resolved-config.yaml", resolved.canonical)
        _write_json(
            run_dir / "environment_fingerprint.json",
            {"observation": fingerprints.observation, "runtime": fingerprints.runtime},
        )
        _write_json(run_dir / "versions.json", {**versions, "git_commit": commit})

        run_meta = {
            "run_id": experiment.id,
            "training_seed": experiment.seed,
            "algorithm_id": algorithm.id,
            "adapter_id": algorithm.adapter,
            "policy_id": algorithm.policy,
            "backend_version": adapter.capabilities(algorithm).backend_version,
            "environment_fingerprint": fingerprints.runtime,
            "code_commit": commit,
            "config_hash": resolved.config_hash,
        }
        # A resumed run keeps evaluations up to the resumed step
        rows = _prior_rows(eval_path, agent.num_timesteps) if resume else []

        def on_checkpoint(step: int) -> None:
            """Save atomically, run the selection suite, and flush evaluations."""
            checkpoint_id = f"checkpoint_{step}"
            path = checkpoint_file(run_dir, checkpoint_id)
            _save_checkpoint(agent, path)

            episodes = evaluate_agent(
                agent.loaded(),
                lambda seed: make_environment(environment, seed=seed),
                experiment.evaluation.selection,
                experiment.evaluation,
            )
            rows.extend(
                {**run_meta, "checkpoint_id": checkpoint_id, "training_step": step, "suite": "selection", **ep}
                for ep in episodes
            )
            write_evaluations(eval_path, pd.DataFrame(rows))

            _write_json(
                path.with_suffix(".json"),
                {
                    "checkpoint_id": checkpoint_id,
                    "training_step": step,
                    "evaluation_status": "complete",
                    "sha256": sha256_file(path),
                    "environment_fingerprint": fingerprints.runtime,
                    "versions": versions,
                },
            )
            mean_return = float(np.mean([ep["return"] for ep in episodes]))
            logger.info("checkpoint %s mean selection return %.2f", checkpoint_id, mean_return, extra=extra)

        callback = CheckpointEvaluationCallback(
            interval_steps=experiment.evaluation.selection.interval_steps
            or experiment.training.checkpoint_interval_steps,
            log_interval_steps=experiment.training.log_interval_steps,
            on_checkpoint=on_checkpoint,
            config_hash=resolved.config_hash,
        )
        logger.info("training %s from step %d on %s", experiment.id, agent.num_timesteps, device, extra=extra)
        agent.learn(
            experiment.training.total_timesteps,
            callback=callback,
            progress_bar=experiment.training.progress_bar,
        )
    finally:
        env.close()

    _write_manifest(run_dir, resolved)

    return run_dir


def _check_run_dir(run_dir: Path, resolved: ResolvedConfig, resume: Path | None) -> None:
    """A run directory is immutable: reuse it only to resume one of its own checkpoints."""
    if resume is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ConfigError(
                f"Run directory already exists: {run_dir}",
                context={"hint": "pass --resume <checkpoint.zip> or change the experiment id"},
            )
        return

    if resume.resolve().parent != (run_dir / "checkpoints").resolve():
        raise CheckpointError(f"--resume must point at a checkpoint inside {run_dir / 'checkpoints'}")
    stored = (run_dir / "resolved-config.yaml").read_text(encoding="utf-8")
    if resolved.canonical != yaml.safe_load(stored):
        raise ConfigError("Cannot resume: the experiment config changed since this run started")


def _prior_rows(eval_path: Path, step: int) -> list[dict[str, Any]]:
    """Selection rows already recorded at or before ``step``."""
    if not eval_path.exists():
        return []
    table = pd.read_parquet(eval_path)
    kept = table[(table["suite"] == "selection") & (table["training_step"] <= step)]

    return kept.to_dict("records")


def _save_checkpoint(agent: TrainableAgent, path: Path) -> None:
    """Write to a temporary name, then rename, so an interrupt never leaves a partial zip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    agent.save(tmp)
    tmp.replace(path)


def _seed_everything(experiment: ExperimentConfig) -> None:
    """Seed Python, NumPy, and PyTorch and apply the determinism flags."""
    random.seed(experiment.seed)
    np.random.seed(experiment.seed)
    th.manual_seed(experiment.seed)
    th.use_deterministic_algorithms(experiment.reproducibility.torch_deterministic, warn_only=True)
    th.backends.cudnn.benchmark = experiment.reproducibility.cudnn_benchmark


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_manifest(run_dir: Path, resolved: ResolvedConfig) -> None:
    """Record checksums of the immutable run files (evaluations change when `evaluate` runs)."""
    files = [
        run_dir / "resolved-config.yaml",
        run_dir / "environment_fingerprint.json",
        *sorted((run_dir / "checkpoints").glob("checkpoint_*.zip")),
    ]
    outputs = [
        ArtifactOutput(
            path=path.relative_to(run_dir).as_posix(),
            sha256=sha256_file(path),
            role="checkpoint" if path.suffix == ".zip" else "run_metadata",
            bytes=path.stat().st_size,
        )
        for path in files
    ]
    manifest = ArtifactManifest(
        stage="train",
        artifact_id=run_dir.name,
        config_hash=resolved.config_hash,
        git_commit=try_git_commit(REPO_ROOT),
        package_versions=capture_package_versions(VERSION_PACKAGES),
        outputs=outputs,
    )
    manifest.write(run_dir / "manifest.json")
