"""Generate the SYNTHETIC, schema-valid test fixture study without Atari or training.

Writes a small self-contained study under ``tests/fixtures/study/`` used by the
dashboard and data-loading tests.

Evaluations, observations, and activations are random numbers with a planted
structure (the "strong" checkpoint separates scenarios more than "weak").
Checkpoint selection, scenario validation, and metric analysis are then run with
the real pipeline code, so every table has the production schema. Nothing here
is an experimental result: every generated directory carries a
``SYNTHETIC_FIXTURE.json`` marker and manifests record ``synthetic: true``. The
generator refuses to touch a directory that lacks its marker.

Usage::

    python scripts/generate_fixture_study.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from drl_similarity.activations.store import (  # noqa: E402
    ActivationEntry,
    activation_relpath,
    publish_activations,
)
from drl_similarity.analysis import run_analysis  # noqa: E402
from drl_similarity.artifacts.manifest import sha256_file  # noqa: E402
from drl_similarity.config import (  # noqa: E402
    ArtifactsPathConfig,
    ExperimentConfig,
    ResolvedConfig,
    StudyConfig,
    canonicalize_config,
    hash_config,
    load_config,
    write_resolved_config,
)
from drl_similarity.errors import DrlSimilarityError  # noqa: E402
from drl_similarity.scenarios.schema import (  # noqa: E402
    ClipRecord,
    CollectorInfo,
    ScenarioManifest,
    assign_splits,
    dump_manifest,
    load_manifest,
    observation_sha256,
)
from drl_similarity.scenarios.validate import validate_scenarios  # noqa: E402
from drl_similarity.study_io import (  # noqa: E402
    CAPABILITIES,
    load_run,
    scenario_data_dir,
    study_root,
)
from drl_similarity.training.selection import select_checkpoints  # noqa: E402


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "study"
EXPERIMENT_TEMPLATE = REPO_ROOT / "configs" / "experiments" / "frogger-dqn-demo.yaml"
MARKER = "SYNTHETIC_FIXTURE.json"
GENERATOR = "scripts/generate_fixture_study.py"
N_CHECKPOINTS = 8

SMALL_DIMS = {"conv1": 8, "conv2": 16, "conv3": 16, "conv_flat": 64, "policy_features": 32, "q_values": 5}
ORIGINAL_SHAPES = {"conv1": (32, 20, 20), "conv2": (64, 9, 9), "conv3": (64, 7, 7)}

FIXTURE_STUDY: dict[str, Any] = {
    "schema_version": 1,
    "kind": "study",
    "id": "fixture-study-v1",
    "experiments": [{"run_dir": "fixture-dqn-run", "checkpoint_set": "selected"}],
    "scenarios": {
        "manifest": "scenarios/fixture-clips.yaml",
        "labels": ["baseline", "danger", "goal"],
        "minimum_clips_per_label": 4,
        "clip_steps": 6,
        "burn_in_steps": 1,
        "collection": {
            "experiment_index": 0,
            "checkpoint": "strong",
            "seeds": [3001, 3002],
            "episodes_per_seed": 1,
            "max_agent_steps": 64,
            "retain_source_rgb": False,
            "write_video_previews": False,
            "candidate_windows": {
                "baseline": {"anchor": "life_start", "offset_steps": 0},
                "danger": {"anchor": "life_lost", "offset_steps": -5},
                "goal": {"anchor": "positive_goal_event", "offset_steps": -5},
            },
        },
        "review": {"require_approved": True, "minimum_reviewers": 1},
        "split": {"unit": "clip", "train_fraction": 0.5, "seed": 3501},
    },
    "extraction": {
        "batch_size": 16,
        "device": "cpu",
        "dtype": "float32",
        "layers": [
            {"name": "conv3", "source": "conv3", "pooling": "global_average"},
            {"name": "policy_features", "source": "policy_features", "pooling": "none"},
            {"name": "q_values", "source": "q_values", "pooling": "none"},
        ],
    },
    "analysis": {
        "pair_sets": ["cross_clip_all", "cross_clip_aligned_step", "clip_centroids"],
        "metrics": [
            {"id": "cosine"},
            {"id": "correlation"},
            {"id": "normalized_euclidean", "kwargs": {"standardize_on": "train_clips"}},
            {"id": "centroid_dispersion"},
            {"id": "temporal_cosine", "kwargs": {"lags": [1, 2]}},
            {"id": "within_between_gap"},
            {"id": "silhouette"},
            {"id": "nearest_centroid_accuracy"},
            {"id": "linear_cka"},
            {"id": "rsa_spearman", "kwargs": {"first_order_distance": "correlation"}},
            {"id": "svcca", "kwargs": {"variance_retained": 0.99}},
            {"id": "orthogonal_procrustes"},
        ],
        "bootstrap": {"unit": "clip", "replicates": 20, "confidence_level": 0.95, "seed": 4001},
        "embeddings": [{"id": "pca", "kwargs": {"components": 2}}],
    },
    "dashboard": {"ref": "dashboard.yaml"},
    "artifacts": {"root": "results", "collision": "error"},
}

FIXTURE_DASHBOARD: dict[str, Any] = {
    "schema_version": 1,
    "kind": "dashboard",
    "id": "fixture-dashboard-v1",
    "title": "Fixture representation study (synthetic)",
    "default_page": "similarity_matrix",
    "default_filters": {
        "checkpoint": "strong",
        "layer": "policy_features",
        "row_scenario": "danger",
        "column_scenario": "danger",
        "frames_per_case": 3,
        "primary_metric": "cosine",
        "pair_set": "cross_clip_aligned_step",
    },
    "theme": {
        "capability_order": ["weak", "medium", "strong"],
        "scenario_order": ["baseline", "danger", "goal"],
    },
    "charts": {
        "show_raw_points": True,
        "confidence_intervals": True,
        "max_heatmap_samples": 500,
        "clip_fps": 8,
    },
    "downloads": {"enabled": True, "formats": ["csv", "png"]},
}


class GeneratorError(RuntimeError):
    """The generator refused to overwrite data it did not create."""


@dataclass(frozen=True)
class Plan:
    name: str
    config_path: Path
    clips_per_label: int
    feature_dims: dict[str, int]
    seed: int


# --------------------------------------------------------------------------- guarded filesystem


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_ours(directory: Path) -> bool:
    return (directory / MARKER).exists()


def _reset_directory(directory: Path) -> None:
    """Empty a directory we own; refuse to delete anything without our marker."""
    if directory.exists():
        if any(directory.iterdir()) and not _is_ours(directory):
            raise GeneratorError(
                f"{directory} exists and was not created by {GENERATOR} (no {MARKER}); "
                "refusing to overwrite it"
            )
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def _write_marker(directory: Path, plan: Plan) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MARKER).write_text(
        json.dumps(
            {
                "synthetic": True,
                "generator": GENERATOR,
                "target": plan.name,
                "seed": plan.seed,
                "created_at": _now(),
                "warning": "Random data with planted structure. Not an experimental result.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _guard_manifest(manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    try:
        synthetic = load_manifest(manifest_path).synthetic
    except DrlSimilarityError:
        synthetic = False
    if not synthetic:
        raise GeneratorError(
            f"{manifest_path} exists and is not marked synthetic; refusing to overwrite it"
        )
    manifest_path.unlink()


# --------------------------------------------------------------------------- run directory


def _fingerprint(payload: Any) -> str:
    return hash_config({"synthetic_fingerprint": payload})


def _resolved_experiment(run_dir: Path) -> ResolvedConfig:
    """Resolved experiment for the synthetic run, derived from the demo experiment YAML."""
    template = load_config(EXPERIMENT_TEMPLATE, expected_kind="experiment")
    experiment = template.value
    assert isinstance(experiment, ExperimentConfig)
    interval = experiment.evaluation.selection.interval_steps or experiment.training.checkpoint_interval_steps
    experiment = experiment.model_copy(
        update={
            "id": run_dir.name,
            "training": experiment.training.model_copy(
                update={"total_timesteps": interval * N_CHECKPOINTS}
            ),
            "artifacts": ArtifactsPathConfig(root=run_dir.parent, collision="error"),
        }
    )
    canonical = canonicalize_config(experiment)
    return ResolvedConfig(
        value=experiment,
        source_paths=template.source_paths,
        config_hash=hash_config(canonical),
        canonical=canonical,
    )


def _write_run(run_dir: Path, plan: Plan) -> None:
    resolved = _resolved_experiment(run_dir)
    experiment = resolved.value
    assert isinstance(experiment, ExperimentConfig)
    _reset_directory(run_dir)
    _write_marker(run_dir, plan)
    write_resolved_config(run_dir / "resolved-config.yaml", resolved.canonical)

    environment = resolved.canonical["environment"]
    fingerprints = {
        "observation": _fingerprint({"observation": environment["expected"], "env": environment["env_id"]}),
        "runtime": _fingerprint({"runtime": environment}),
        "synthetic": True,
    }
    (run_dir / "environment_fingerprint.json").write_text(
        json.dumps(fingerprints, indent=2), encoding="utf-8"
    )

    rng = np.random.default_rng(plan.seed)
    algorithm = experiment.algorithm
    run_meta = {
        "run_id": experiment.id,
        "training_seed": experiment.seed,
        "algorithm_id": algorithm.id,  # type: ignore[union-attr]
        "adapter_id": algorithm.adapter,  # type: ignore[union-attr]
        "policy_id": algorithm.policy,  # type: ignore[union-attr]
        "backend_version": "synthetic",
        "environment_fingerprint": fingerprints["runtime"],
        "code_commit": None,
        "config_hash": resolved.config_hash,
        "wall_time": None,
    }
    interval = experiment.training.total_timesteps // N_CHECKPOINTS
    rows: list[dict[str, Any]] = []
    for index in range(1, N_CHECKPOINTS + 1):
        step = interval * index
        mean_return = 6.0 + 60.0 * (1.0 - np.exp(-index / 3.0))
        for seed in experiment.evaluation.selection.seeds:
            for episode in range(experiment.evaluation.selection.episodes_per_seed):
                value = float(max(0.0, rng.normal(mean_return, 3.0)))
                rows.append(
                    {
                        **run_meta,
                        "checkpoint_id": f"checkpoint_{step}",
                        "training_step": step,
                        "suite": "selection",
                        "eval_seed": int(seed),
                        "episode_index": episode,
                        "return": value,
                        "length": int(rng.integers(200, 900)),
                        "lives_lost": int(rng.integers(1, 5)),
                        "goals": int(value // 10),
                    }
                )
    final_mean = 6.0 + 60.0 * (1.0 - np.exp(-N_CHECKPOINTS / 3.0))
    for seed in experiment.evaluation.final.seeds:
        for episode in range(experiment.evaluation.final.episodes_per_seed):
            value = float(max(0.0, rng.normal(final_mean, 3.0)))
            rows.append(
                {
                    **run_meta,
                    "checkpoint_id": "checkpoint_final",
                    "training_step": experiment.training.total_timesteps,
                    "suite": "final",
                    "eval_seed": int(seed),
                    "episode_index": episode,
                    "return": value,
                    "length": int(rng.integers(200, 900)),
                    "lives_lost": int(rng.integers(1, 5)),
                    "goals": int(value // 10),
                }
            )
    pd.DataFrame(rows).to_parquet(run_dir / "evaluations.parquet", index=False)
    (run_dir / "versions.json").write_text(
        json.dumps({"synthetic": True, "numpy": np.__version__, "pandas": pd.__version__}, indent=2),
        encoding="utf-8",
    )
    # Real selection rules applied to the synthetic evaluation table.
    select_checkpoints(resolved)


# --------------------------------------------------------------------------- scenarios


def _synthetic_clip(
    rng: np.random.Generator, label_index: int, n_frames: int, shape: tuple[int, ...]
) -> np.ndarray:
    """Draw a toy grayscale 'road crossing' clip shaped ``(n_frames, *shape)`` (uint8)."""
    stack, height, width = (shape[0], shape[-2], shape[-1]) if len(shape) == 3 else (1, shape[-2], shape[-1])
    total = n_frames + stack - 1
    lanes = 6
    lane_height = height // (lanes + 2)
    speeds = rng.integers(1, 4, size=lanes) * rng.choice([-1, 1], size=lanes)
    offsets = rng.integers(0, width, size=lanes)
    frog_x = int(rng.integers(8, width - 14))
    frames = np.zeros((total, height, width), dtype=np.uint8)
    for t in range(total):
        frame = frames[t]
        frame[:lane_height] = 40  # goal bank
        frame[height - lane_height :] = 25  # start bank
        for lane in range(lanes):
            top = lane_height * (lane + 1)
            car_x = int((offsets[lane] + speeds[lane] * t) % width)
            frame[top + 2 : top + lane_height - 2, car_x : min(car_x + 12, width)] = 140 + 15 * (lane % 3)
        progress = t / max(total - 1, 1)
        if label_index % 3 == 0:  # baseline: waiting on the start bank
            frog_y = height - lane_height + 2 + int(rng.integers(0, 2))
        elif label_index % 3 == 1:  # danger: drifting into traffic
            frog_y = int(height - lane_height - progress * lane_height * 3)
            car_x = int(frog_x - (1.0 - progress) * 30)
            lane_top = (frog_y // lane_height) * lane_height
            frame[lane_top + 2 : lane_top + lane_height - 2, max(car_x, 0) : max(car_x, 0) + 12] = 200
        else:  # goal: crossing to the far bank
            frog_y = int((height - lane_height) * (1.0 - progress)) + 1
        frog_y = int(np.clip(frog_y, 0, height - 7))
        frame[frog_y : frog_y + 6, frog_x : frog_x + 6] = 255
    if len(shape) != 3:
        return frames[:n_frames].reshape(n_frames, *shape)
    return np.stack([frames[t : t + stack] for t in range(n_frames)], axis=0)


def _write_scenarios(study: StudyConfig, plan: Plan) -> ScenarioManifest:
    cfg = study.scenarios
    manifest_path = Path(cfg.manifest)
    data_dir = scenario_data_dir(manifest_path)
    _guard_manifest(manifest_path)
    _reset_directory(data_dir)
    _write_marker(data_dir, plan)

    run = load_run(Path(study.experiments[0].run_dir), 0)
    obs_shape = tuple(run.environment.expected.observation_shape)
    rng = np.random.default_rng(plan.seed + 1)
    created_at = _now()
    arrays: dict[str, np.ndarray] = {}
    labels_by_clip: dict[str, str] = {}
    drafts: list[dict[str, Any]] = []
    seeds = list(cfg.collection.seeds)
    for label_index, label in enumerate(cfg.labels):
        window = cfg.collection.candidate_windows.get(label)
        offset = window.offset_steps if window else 0
        anchor_event = window.anchor if window else "synthetic"
        for index in range(plan.clips_per_label):
            clip_id = f"{label}-{index + 1:04d}"
            frames = _synthetic_clip(rng, label_index, cfg.burn_in_steps + cfg.clip_steps, obs_shape)
            arrays[clip_id] = frames
            labels_by_clip[clip_id] = label
            start = max(cfg.burn_in_steps, -offset) + 10 + index
            drafts.append(
                {
                    "clip_id": clip_id,
                    "scenario": label,
                    "source_episode_id": f"synthetic-{clip_id}",
                    "collection_seed": int(seeds[index % len(seeds)]),
                    "start_step": start,
                    "anchor_step": start - offset,
                    "end_step": start + cfg.clip_steps - 1,
                    "burn_in_steps": cfg.burn_in_steps,
                    "anchor_event": anchor_event,
                    "observation_key": clip_id,
                    "observation_sha256": observation_sha256(frames),
                }
            )
    splits = assign_splits(labels_by_clip, train_fraction=cfg.split.train_fraction, seed=cfg.split.seed)
    reviewers = ", ".join(f"synthetic-reviewer-{i + 1}" for i in range(max(cfg.review.minimum_reviewers, 1)))
    clips = [
        ClipRecord(
            **draft,
            split=splits[draft["clip_id"]],
            review_status="approved",
            curator=reviewers,
            notes="SYNTHETIC toy frames from scripts/generate_fixture_study.py",
            created_at=created_at,
        )
        for draft in drafts
    ]
    manifest = ScenarioManifest(
        dataset_id=manifest_path.stem,
        environment_fingerprint=str(run.observation_fingerprint),
        collector=CollectorInfo(policy="synthetic-generator", run_id=run.run_id),
        created_at=created_at,
        synthetic=True,
        notes="SYNTHETIC dataset. Frames are drawn toy images, not Atari observations.",
        clips=clips,
    )
    np.savez_compressed(data_dir / "observations.npz", **arrays)
    dump_manifest(
        manifest,
        manifest_path,
        header=(
            f"SYNTHETIC scenario manifest generated by {GENERATOR}.\n"
            "Frames are toy images with planted structure, not Atari observations.\n"
            "Do not use for analysis conclusions."
        ),
    )
    return manifest


# --------------------------------------------------------------------------- activations


_SEPARATION = {"weak": 0.35, "medium": 1.0, "strong": 1.8}
_NOISE = {"weak": 1.0, "medium": 0.7, "strong": 0.45}
_SMOOTHNESS = {"weak": 0.3, "medium": 0.6, "strong": 0.85}
_DRIFT = {"weak": 0.9, "medium": 0.45, "strong": 0.0}
_LATENT = 8


def _synthetic_activations(
    plan: Plan,
    manifest: ScenarioManifest,
    study: StudyConfig,
) -> dict[tuple[str, str], np.ndarray]:
    """Planted-structure activations keyed by (capability, layer)."""
    labels = list(study.scenarios.labels)
    clips = manifest.active_clips(require_approved=True)
    n_clips = len(clips)
    n_steps = study.scenarios.clip_steps
    rng = np.random.default_rng(plan.seed + 2)
    scenario_means = rng.normal(0.0, 1.0, size=(len(labels), _LATENT))
    label_index = np.asarray([labels.index(clip.scenario) for clip in clips])
    clip_effect = rng.normal(0.0, 0.5, size=(n_clips, _LATENT))
    innovations = rng.normal(0.0, 1.0, size=(n_clips, n_steps, _LATENT))

    outputs: dict[tuple[str, str], np.ndarray] = {}
    for layer in study.extraction.layers:
        width = plan.feature_dims.get(layer.name, 32)
        base = rng.normal(0.0, 1.0, size=(_LATENT, width)) / np.sqrt(_LATENT)
        bias = rng.uniform(0.2, 0.8, size=width)
        for capability in CAPABILITIES:
            rho = _SMOOTHNESS[capability]
            temporal = np.zeros_like(innovations)
            temporal[:, 0] = innovations[:, 0]
            for step in range(1, n_steps):
                temporal[:, step] = rho * temporal[:, step - 1] + np.sqrt(1 - rho**2) * innovations[:, step]
            latent = (
                _SEPARATION[capability] * scenario_means[label_index][:, None, :]
                + clip_effect[:, None, :]
                + _NOISE[capability] * temporal
            )
            drift = rng.normal(0.0, 1.0, size=(_LATENT, width)) / np.sqrt(_LATENT)
            projection = base + _DRIFT[capability] * drift
            values = latent @ projection + bias
            if layer.name != "q_values":
                values = np.maximum(values, 0.0) + 0.01  # ReLU-like, never an all-zero vector
            outputs[(capability, layer.name)] = values.astype(np.float32)
    return outputs


def _write_activations(resolved: ResolvedConfig, manifest: ScenarioManifest, plan: Plan) -> Path:
    study = resolved.value
    assert isinstance(study, StudyConfig)
    run = load_run(Path(study.experiments[0].run_dir), 0)
    clips = manifest.active_clips(require_approved=True)
    clips_frame = pd.DataFrame(
        {
            "clip_index": np.arange(len(clips), dtype=np.int64),
            "clip_id": [clip.clip_id for clip in clips],
            "scenario": [clip.scenario for clip in clips],
            "split": [clip.split for clip in clips],
            "source_episode_id": [clip.source_episode_id for clip in clips],
            "collection_seed": np.asarray([clip.collection_seed for clip in clips], dtype=np.int64),
            "observation_sha256": [clip.observation_sha256 for clip in clips],
        }
    )
    synthetic = _synthetic_activations(plan, manifest, study)
    entries: list[ActivationEntry] = []
    arrays: dict[str, np.ndarray] = {}
    for capability in CAPABILITIES:
        checkpoint = run.selected[capability]
        for layer in study.extraction.layers:
            array = synthetic[(capability, layer.name)]
            width = int(array.shape[-1])
            path = activation_relpath(run.run_id, capability, layer.name)
            arrays[path] = array
            if layer.pooling == "global_average":
                original = ORIGINAL_SHAPES.get(layer.name, (width, 7, 7))
                original = (width, *original[1:])
            else:
                original = (width,)
            entries.append(
                ActivationEntry(
                    run_id=run.run_id,
                    algorithm_id=run.algorithm.id,
                    adapter_id=run.algorithm.adapter,
                    training_seed=int(run.experiment.seed),
                    capability=capability,
                    checkpoint_id=checkpoint.checkpoint_id,
                    training_step=int(checkpoint.training_step),
                    checkpoint_sha256="synthetic",
                    layer=layer.name,
                    source_layer=layer.source or layer.name,
                    pooling=layer.pooling,
                    original_shape=tuple(int(dim) for dim in original),
                    feature_dim=width,
                    dtype="float32",
                    path=path,
                )
            )
    obs_file = scenario_data_dir(Path(study.scenarios.manifest)) / "observations.npz"
    return publish_activations(
        resolved=resolved,
        study=study,
        clips=clips_frame,
        clip_steps=study.scenarios.clip_steps,
        entries=entries,
        arrays=arrays,
        input_files=[
            (Path(study.scenarios.manifest), "scenario_manifest"),
            (obs_file, "observations"),
            (run.run_dir / "resolved-config.yaml", "run_config"),
            (run.run_dir / "selected-checkpoints.yaml", "checkpoint_selection"),
        ],
        metadata={
            "synthetic": True,
            "generator": GENERATOR,
            "device": "none",
            "environment_fingerprint": manifest.environment_fingerprint,
            "observations_sha256": sha256_file(obs_file),
            "recurrent": False,
        },
        packages=["numpy", "pandas"],
    )


# --------------------------------------------------------------------------- orchestration


def populate(plan: Plan) -> dict[str, Path]:
    """Generate every artifact a study config needs; returns the created locations."""
    resolved = load_config(plan.config_path, expected_kind="study", allow_missing_scenario_manifest=True)
    study = resolved.value
    assert isinstance(study, StudyConfig)
    if len(study.experiments) != 1:
        raise GeneratorError("the synthetic generator supports exactly one experiment per study")
    run_dir = Path(study.experiments[0].run_dir)
    manifest_path = Path(study.scenarios.manifest)
    artifacts = study_root(study)

    # Check every destination before writing anything.
    for directory in (run_dir, scenario_data_dir(manifest_path), artifacts):
        if directory.exists() and any(directory.iterdir()) and not _is_ours(directory):
            raise GeneratorError(
                f"{directory} exists and was not created by {GENERATOR} (no {MARKER}); "
                "refusing to overwrite it"
            )
    if manifest_path.exists():
        try:
            if not load_manifest(manifest_path).synthetic:
                raise GeneratorError(
                    f"{manifest_path} exists and is not marked synthetic; refusing to overwrite it"
                )
        except DrlSimilarityError as exc:
            raise GeneratorError(f"{manifest_path} exists and could not be read: {exc}") from exc

    _write_run(run_dir, plan)
    manifest = _write_scenarios(study, plan)
    validate_scenarios(resolved)
    _reset_directory(artifacts)
    _write_marker(artifacts, plan)
    activations = _write_activations(resolved, manifest, plan)
    metrics = run_analysis(resolved)

    from drl_similarity.dashboard.data import validate_dashboard_artifacts

    validate_dashboard_artifacts(resolved)
    return {
        "study_config": plan.config_path,
        "run_dir": run_dir,
        "scenario_manifest": manifest_path,
        "observations": scenario_data_dir(manifest_path) / "observations.npz",
        "activations": activations,
        "metrics": metrics,
    }


def _write_fixture_configs() -> Path:
    if FIXTURE_ROOT.exists() and any(FIXTURE_ROOT.iterdir()) and not _is_ours(FIXTURE_ROOT):
        raise GeneratorError(
            f"{FIXTURE_ROOT} exists and was not created by {GENERATOR} (no {MARKER}); "
            "refusing to overwrite it"
        )
    if FIXTURE_ROOT.exists():
        shutil.rmtree(FIXTURE_ROOT)
    FIXTURE_ROOT.mkdir(parents=True)
    header = f"# SYNTHETIC fixture configuration generated by {GENERATOR}. Do not edit by hand.\n"
    (FIXTURE_ROOT / "study.yaml").write_text(
        header + yaml.safe_dump(FIXTURE_STUDY, sort_keys=False), encoding="utf-8"
    )
    (FIXTURE_ROOT / "dashboard.yaml").write_text(
        header + yaml.safe_dump(FIXTURE_DASHBOARD, sort_keys=False), encoding="utf-8"
    )
    # Manifests record checksums of these files; stop git from rewriting line endings.
    (FIXTURE_ROOT / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    return FIXTURE_ROOT / "study.yaml"


def generate_fixture(seed: int) -> dict[str, Path]:
    config_path = _write_fixture_configs()
    plan = Plan(name="fixture", config_path=config_path, clips_per_label=4, feature_dims=SMALL_DIMS, seed=seed)
    _write_marker(FIXTURE_ROOT, plan)
    return populate(plan)


def _report(name: str, created: dict[str, Path]) -> None:
    print(f"[{name}] synthetic study generated:")
    for role, path in created.items():
        try:
            shown = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            shown = str(path)
        print(f"  {role:18s} {shown}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20261109, help="seed for the synthetic data")
    args = parser.parse_args(argv)

    _report("fixture", generate_fixture(args.seed))
    print("NOTE: all generated values are synthetic; they are not experimental results.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GeneratorError, DrlSimilarityError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
