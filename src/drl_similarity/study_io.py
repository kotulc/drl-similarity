"""Shared study-level helpers: run directories, stage hashes, and artifact versions.

These helpers are used by scenario collection/validation, activation extraction,
analysis, and the dashboard readers. They never import a training backend.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from drl_similarity.artifacts.manifest import ArtifactInput, sha256_file
from drl_similarity.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    ExperimentConfig,
    ResolvedConfig,
    StudyConfig,
    hash_config,
)
from drl_similarity.errors import CheckpointError, ConfigError


CAPABILITIES: tuple[str, ...] = ("weak", "medium", "strong")

ACTIVATIONS_ARTIFACT_ID = "activations"
METRICS_ARTIFACT_ID = "metrics"


@dataclass(frozen=True)
class SelectedCheckpoint:
    capability: str
    checkpoint_id: str
    training_step: int
    score: float | None
    uncertainty: float | None
    path: Path


@dataclass(frozen=True)
class RunInfo:
    """A completed experiment run referenced by a study."""

    index: int
    run_dir: Path
    run_id: str
    experiment: ExperimentConfig
    environment: EnvironmentConfig
    algorithm: AlgorithmConfig
    selected: dict[str, SelectedCheckpoint]
    selection_payload: dict[str, Any]
    observation_fingerprint: str | None
    runtime_fingerprint: str | None

    @property
    def evaluation_seeds(self) -> set[int]:
        evaluation = self.experiment.evaluation
        return set(evaluation.selection.seeds) | set(evaluation.final.seeds)


def require_study(resolved: ResolvedConfig, *, stage: str) -> StudyConfig:
    if not isinstance(resolved.value, StudyConfig):
        raise ConfigError(f"{stage} requires a study config", context={"kind": resolved.value.kind})
    return resolved.value


def study_root(study: StudyConfig) -> Path:
    """Directory holding every artifact stage of one study."""
    return Path(study.artifacts.root) / study.id


def scenario_data_dir(manifest_path: Path) -> Path:
    """Data directory that accompanies a scenario manifest (``<stem>/`` beside it)."""
    manifest_path = Path(manifest_path)
    return manifest_path.parent / manifest_path.stem


def observations_path(study: StudyConfig) -> Path:
    return scenario_data_dir(Path(study.scenarios.manifest)) / "observations.npz"


def checkpoint_file(run_dir: Path, checkpoint_id: str) -> Path:
    return Path(run_dir) / "checkpoints" / f"{checkpoint_id}.zip"


def load_run(run_dir: Path, index: int = 0, *, require_selected: bool = True) -> RunInfo:
    """Read the resolved experiment and checkpoint selection stored in a run directory."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise CheckpointError(
            f"Missing run directory: {run_dir}",
            context={"hint": "train first with: drl-sim train --config <experiment.yaml>"},
        )
    config_path = run_dir / "resolved-config.yaml"
    if not config_path.exists():
        raise CheckpointError(
            f"Run directory has no resolved-config.yaml: {run_dir}",
            context={"hint": "re-run drl-sim train to regenerate run metadata"},
        )
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        experiment = ExperimentConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(
            f"Invalid resolved experiment config: {config_path}",
            context={"error": str(exc)},
        ) from exc
    if not isinstance(experiment.environment, EnvironmentConfig) or not isinstance(
        experiment.algorithm, AlgorithmConfig
    ):
        raise ConfigError(
            f"Resolved experiment config must inline environment and algorithm: {config_path}"
        )

    selected: dict[str, SelectedCheckpoint] = {}
    payload: dict[str, Any] = {}
    selected_path = run_dir / "selected-checkpoints.yaml"
    if selected_path.exists():
        payload, selected = _read_selected(run_dir, selected_path)
    elif require_selected:
        raise CheckpointError(
            f"Missing {selected_path}",
            context={"hint": "run: drl-sim select-checkpoints --config <experiment.yaml>"},
        )

    observation_fp: str | None = None
    runtime_fp: str | None = None
    fingerprint_path = run_dir / "environment_fingerprint.json"
    if fingerprint_path.exists():
        try:
            fingerprints = json.loads(fingerprint_path.read_text(encoding="utf-8"))
            observation_fp = fingerprints.get("observation")
            runtime_fp = fingerprints.get("runtime")
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(
                f"Invalid environment fingerprint file: {fingerprint_path}",
                context={"error": str(exc)},
            ) from exc

    return RunInfo(
        index=index,
        run_dir=run_dir,
        run_id=experiment.id,
        experiment=experiment,
        environment=experiment.environment,
        algorithm=experiment.algorithm,
        selected=selected,
        selection_payload=payload,
        observation_fingerprint=observation_fp,
        runtime_fingerprint=runtime_fp,
    )


def _read_selected(
    run_dir: Path, selected_path: Path
) -> tuple[dict[str, Any], dict[str, SelectedCheckpoint]]:
    try:
        payload = yaml.safe_load(selected_path.read_text(encoding="utf-8")) or {}
        entries = payload["selected"]
        selected: dict[str, SelectedCheckpoint] = {}
        for capability in CAPABILITIES:
            item = entries[capability]
            checkpoint_id = str(item["checkpoint_id"])
            score = item.get("score")
            uncertainty = item.get("uncertainty")
            selected[capability] = SelectedCheckpoint(
                capability=capability,
                checkpoint_id=checkpoint_id,
                training_step=int(item["training_step"]),
                score=None if score is None else float(score),
                uncertainty=None if uncertainty is None else float(uncertainty),
                path=checkpoint_file(run_dir, checkpoint_id),
            )
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(
            f"Invalid checkpoint selection file: {selected_path}",
            context={"error": repr(exc), "required": list(CAPABILITIES)},
        ) from exc
    return payload, selected


def load_runs(study: StudyConfig, *, require_selected: bool = True) -> list[RunInfo]:
    runs = [
        load_run(Path(ref.run_dir), index, require_selected=require_selected)
        for index, ref in enumerate(study.experiments)
    ]
    seen: dict[str, Path] = {}
    for run in runs:
        if run.run_id in seen:
            raise ConfigError(
                "Study references two runs with the same experiment id",
                context={"run_id": run.run_id, "run_dirs": [str(seen[run.run_id]), str(run.run_dir)]},
            )
        seen[run.run_id] = run.run_dir
    return runs


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch as th

    return "cuda" if th.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- stage hashes


def _scenario_stage_view(canonical: dict[str, Any]) -> dict[str, Any]:
    scenarios = dict(canonical.get("scenarios", {}))
    # Collection settings only produce candidates; curated clips are tracked by checksum.
    scenarios.pop("collection", None)
    return scenarios


def extraction_stage_hash(resolved: ResolvedConfig) -> str:
    """Hash of the configuration that determines activation arrays."""
    canonical = resolved.canonical
    return hash_config(
        {
            "stage": "extract",
            "id": canonical.get("id"),
            "experiments": canonical.get("experiments"),
            "scenarios": _scenario_stage_view(canonical),
            "extraction": canonical.get("extraction"),
        }
    )


def analysis_stage_hash(resolved: ResolvedConfig) -> str:
    """Hash of the configuration that determines metric tables."""
    canonical = resolved.canonical
    return hash_config(
        {
            "stage": "analyze",
            "extract": extraction_stage_hash(resolved),
            "analysis": canonical.get("analysis"),
        }
    )


# --------------------------------------------------------------------------- artifact versions


def artifact_versions(root: Path, artifact_id: str) -> list[tuple[int, Path]]:
    """Published versions of an artifact directory, oldest first.

    ``ArtifactStore`` never overwrites: the first publish is ``<artifact_id>`` and
    every forced re-publish is ``<artifact_id>-vN``.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    pattern = re.compile(rf"^{re.escape(artifact_id)}(?:-v(\d+))?$")
    versions: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = pattern.match(child.name)
        if not match or not child.is_dir() or not (child / "manifest.json").exists():
            continue
        versions.append((int(match.group(1) or 1), child))
    return sorted(versions, key=lambda item: item[0])


def latest_artifact_dir(root: Path, artifact_id: str) -> Path | None:
    versions = artifact_versions(root, artifact_id)
    return versions[-1][1] if versions else None


def relative_input_path(path: Path, artifact_dir: Path) -> str:
    """Path of an input relative to the artifact directory (portable across checkouts)."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(artifact_dir).resolve())).as_posix()
    except ValueError:  # different drive on Windows
        return Path(path).resolve().as_posix()


def make_input(path: Path, *, role: str, artifact_dir: Path) -> ArtifactInput:
    return ArtifactInput(
        path=relative_input_path(path, artifact_dir),
        sha256=sha256_file(Path(path)),
        role=role,
    )
