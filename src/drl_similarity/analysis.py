"""Compute representation metrics and embeddings from extracted activations.

Outputs under ``artifacts/<study_id>/metrics/``:

``summaries.parquet``
    Within-checkpoint estimates with clip-bootstrap percentile intervals.
``cross_checkpoint.parquet``
    Linear CKA, RSA, SVCCA, and Procrustes between aligned sample sets.
``frame_pairs.parquet``
    Diagnostic pair-level values: clip pairs, representative frame pairs, and
    temporal (lagged) pairs with an input-pixel change control.
``embeddings.parquet``
    PCA / optional UMAP coordinates for frames and clip centroids.
``manifest.json``
    Checksums and lineage for every input and output.

Analysis rules implemented here:

* Cross-clip pair sets never compare a clip with itself.
* Feature standardization and the nearest-centroid classifier are fitted on
  training clips only and applied to held-out clips.
* Confidence intervals resample whole clips, stratified by scenario.
* NaN values are never dropped silently; they are counted in ``n_nan`` /
  ``warnings`` columns and in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drl_similarity.activations.store import (
    EXTRACT_COMMAND,
    ActivationBundle,
    ActivationEntry,
    load_activation_bundle,
)
from drl_similarity.artifacts.manifest import (
    ArtifactManifest,
    ArtifactOutput,
    capture_package_versions,
    sha256_file,
    try_git_commit,
    verify_artifact_dir,
)
from drl_similarity.artifacts.store import ArtifactStore
from drl_similarity.config import REPO_ROOT, ResolvedConfig, StudyConfig, write_resolved_config
from drl_similarity.errors import AnalysisError, ArtifactError
from drl_similarity.logging import get_logger, log_extra
from drl_similarity.metrics import geometry, vector
from drl_similarity.metrics.bootstrap import (
    ESTIMATE_OUTSIDE_INTERVAL,
    BootstrapResult,
    bootstrap_statistic,
    resample_counts,
    stratified_clip_resample,
    summarize,
    weighted_clip_mean,
    weighted_pair_mean,
)
from drl_similarity.metrics.pairs import (
    CENTROID_STEP,
    balanced_clip_indices,
    clip_pair_id,
    frame_pair_id,
    representative_frames,
    scenario_members,
    validate_pair_set,
)
from drl_similarity.metrics.registry import (
    describe_metrics,
    get_metric,
    validate_metric_kwargs,
)
from drl_similarity.scenarios.schema import load_manifest
from drl_similarity.study_io import (
    ACTIVATIONS_ARTIFACT_ID,
    METRICS_ARTIFACT_ID,
    analysis_stage_hash,
    extraction_stage_hash,
    latest_artifact_dir,
    make_input,
    observations_path,
    require_study,
    study_root,
)
from drl_similarity.tables import (
    ALL_SCENARIOS,
    CLIP_ALIGNED_STEPS,
    CROSS_CHECKPOINT_COLUMNS,
    EMBEDDING_COLUMNS,
    FRAME_PAIR_COLUMNS,
    METRICS_SCHEMA,
    NOT_APPLICABLE,
    RAW_COSINE_DIAGNOSTIC,
    SUMMARY_COLUMNS,
    TABLE_SCHEMA_VERSION,
    ordered_table,
)


PAIRWISE_METRICS: tuple[str, ...] = ("cosine", "correlation", "normalized_euclidean")
GEOMETRY_METRICS: tuple[str, ...] = (
    "linear_cka",
    "rsa_spearman",
    "svcca",
    "orthogonal_procrustes",
)
EMBEDDING_IDS: tuple[str, ...] = ("pca", "umap")
STANDARDIZE_CHOICES: tuple[str, ...] = ("train_clips", "all_clips")
SAMPLE_UNITS: tuple[str, ...] = ("frame", "clip_centroid")
#: Default number of evenly spaced steps per clip used as cross-checkpoint samples.
DEFAULT_MAX_STEPS_PER_CLIP = 4


# --------------------------------------------------------------------------- context


@dataclass(frozen=True)
class _SamplePlan:
    """Which samples a cross-checkpoint metric compares."""

    unit: str
    steps: tuple[int, ...]  # relative steps used per clip (empty for clip centroids)
    label: str

    @property
    def per_clip(self) -> int:
        return 1 if self.unit == "clip_centroid" else len(self.steps)

    def matrix(self, activations: np.ndarray) -> np.ndarray:
        if self.unit == "clip_centroid":
            return vector.clip_centroids(activations)
        selected = activations[:, list(self.steps), :]
        return selected.reshape(-1, activations.shape[2])

    def sample_ids(self, clip_ids: tuple[str, ...]) -> list[str]:
        if self.unit == "clip_centroid":
            return [f"{clip_id}:centroid" for clip_id in clip_ids]
        return [f"{clip_id}:{step}" for clip_id in clip_ids for step in self.steps]

    def sample_index(self, clip_indices: np.ndarray) -> np.ndarray:
        if self.per_clip == 1:
            return np.asarray(clip_indices, dtype=np.int64)
        offsets = np.arange(self.per_clip)
        return (np.asarray(clip_indices)[:, None] * self.per_clip + offsets[None, :]).ravel()


@dataclass
class _Context:
    study: StudyConfig
    config_hash: str
    labels: tuple[str, ...]
    clip_ids: tuple[str, ...]
    scenarios: np.ndarray
    splits: np.ndarray
    n_steps: int
    members: dict[str, np.ndarray]
    balanced: dict[str, np.ndarray]
    resamples: np.ndarray
    counts: np.ndarray
    split_resamples: np.ndarray
    replicates: int
    confidence: float
    seed: int
    pair_sets: tuple[str, ...]
    metrics: dict[str, dict[str, Any]]
    input_cosine: dict[int, np.ndarray]
    input_change: dict[int, np.ndarray]
    notes: list[str]

    @property
    def n_clips(self) -> int:
        return len(self.clip_ids)

    def enabled(self, metric_id: str) -> bool:
        return metric_id in self.metrics


def _even_steps(n_steps: int, count: int | None) -> tuple[int, ...]:
    if count is None or count >= n_steps:
        return tuple(range(n_steps))
    if count < 1:
        raise AnalysisError("max_steps_per_clip must be at least 1 or null")
    values = np.unique(np.rint(np.linspace(0, n_steps - 1, num=count)).astype(int))
    return tuple(int(value) for value in values)


def _sample_plan(n_steps: int, kwargs: dict[str, Any]) -> _SamplePlan:
    unit = kwargs.get("sample_unit", "frame")
    if unit not in SAMPLE_UNITS:
        raise AnalysisError(
            f"Unknown sample_unit {unit!r}; valid: {', '.join(SAMPLE_UNITS)}"
        )
    if unit == "clip_centroid":
        return _SamplePlan(unit=unit, steps=(), label="clip_centroid")
    steps = _even_steps(n_steps, kwargs.get("max_steps_per_clip", DEFAULT_MAX_STEPS_PER_CLIP))
    return _SamplePlan(unit=unit, steps=steps, label=f"frame[{len(steps)}_of_{n_steps}_steps_per_clip]")


def _enabled_metrics(study: StudyConfig) -> dict[str, dict[str, Any]]:
    enabled: dict[str, dict[str, Any]] = {}
    for spec in study.analysis.metrics:
        get_metric(spec.id)  # RegistryError lists valid IDs for unknown metrics
        validate_metric_kwargs(spec.id, dict(spec.kwargs))
        if spec.enabled:
            enabled[spec.id] = dict(spec.kwargs)
    return enabled


def _temporal_lags(metrics: dict[str, dict[str, Any]]) -> list[int]:
    if "temporal_cosine" not in metrics:
        return []
    lags = metrics["temporal_cosine"].get("lags", [1])
    if not isinstance(lags, list) or not all(isinstance(lag, int) and lag >= 1 for lag in lags):
        raise AnalysisError("temporal_cosine lags must be a list of positive integers", context={"lags": lags})
    return sorted(set(lags))


def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(values)
    count = valid.sum(axis=1)
    total = np.where(valid, values, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _input_controls(
    study: StudyConfig,
    bundle: ActivationBundle,
    lags: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Input-space controls for temporal metrics, computed from the frozen observations.

    Returns per-lag ``(clips, steps - lag)`` arrays: cosine between flattened
    observations, and mean absolute change of the newest frame (0-1 for uint8).
    """
    manifest = load_manifest(Path(study.scenarios.manifest))
    records = {clip.clip_id: clip for clip in manifest.clips}
    obs_file = observations_path(study)
    if not obs_file.exists():
        raise AnalysisError(f"Missing scenario observations: {obs_file}")
    n_steps = bundle.clip_steps
    usable = [lag for lag in lags if lag < n_steps]
    cosine = {lag: np.full((len(bundle.clips), n_steps - lag), np.nan) for lag in usable}
    change = {lag: np.full((len(bundle.clips), n_steps - lag), np.nan) for lag in usable}
    with np.load(obs_file, allow_pickle=False) as archive:
        for row in bundle.clips.itertuples(index=False):
            record = records.get(row.clip_id)
            if record is None or record.observation_sha256 != row.observation_sha256:
                raise AnalysisError(
                    f"Clip {row.clip_id} in the activation artifact no longer matches the "
                    "scenario manifest",
                    context={"regenerate": EXTRACT_COMMAND + " --force"},
                )
            if not usable:
                continue
            frames = np.asarray(archive[record.observation_key])[-n_steps:]
            flat = frames.reshape(n_steps, -1).astype(np.float64)
            newest = frames[:, -1] if frames.ndim >= 4 else frames
            newest = newest.reshape(n_steps, -1).astype(np.float64)
            scale = 255.0 if frames.dtype == np.uint8 else 1.0
            for lag in usable:
                cosine[lag][row.clip_index] = vector.temporal_cosine(flat[None, ...], lag)[0]
                change[lag][row.clip_index] = (
                    np.abs(newest[lag:] - newest[:-lag]).mean(axis=1) / scale
                )
    return cosine, change


def _build_context(
    resolved: ResolvedConfig, study: StudyConfig, bundle: ActivationBundle
) -> _Context:
    labels = tuple(study.scenarios.labels)
    clips = bundle.clips
    clip_ids = tuple(str(value) for value in clips["clip_id"])
    scenarios = clips["scenario"].astype(str).to_numpy()
    splits = clips["split"].astype(str).to_numpy()
    unknown = sorted(set(scenarios.tolist()) - set(labels))
    if unknown:
        raise AnalysisError(
            "Activation clips carry scenario labels that are not configured",
            context={"unknown": unknown, "labels": list(labels)},
        )
    pair_sets = tuple(validate_pair_set(pair_set) for pair_set in study.analysis.pair_sets)
    if len(set(pair_sets)) != len(pair_sets):
        raise AnalysisError("Duplicate analysis pair sets", context={"pair_sets": list(pair_sets)})
    metrics = _enabled_metrics(study)
    boot = study.analysis.bootstrap
    rng = np.random.default_rng(boot.seed)
    members = scenario_members(scenarios, labels)
    balanced = balanced_clip_indices(scenarios, labels, rng)
    notes: list[str] = []
    sizes = {label: int(len(indices)) for label, indices in members.items()}
    if len(set(sizes.values())) > 1:
        notes.append(f"unbalanced_clip_counts:{sizes}; between-scenario estimates use balanced subsets")
    resamples = stratified_clip_resample(scenarios, boot.replicates, rng)
    counts = resample_counts(resamples, len(clip_ids))
    split_strata = [f"{scenario}|{split}" for scenario, split in zip(scenarios, splits, strict=True)]
    split_resamples = stratified_clip_resample(split_strata, boot.replicates, rng)
    input_cosine, input_change = _input_controls(study, bundle, _temporal_lags(metrics))
    return _Context(
        study=study,
        config_hash=resolved.config_hash,
        labels=labels,
        clip_ids=clip_ids,
        scenarios=scenarios,
        splits=splits,
        n_steps=bundle.clip_steps,
        members=members,
        balanced=balanced,
        resamples=resamples,
        counts=counts,
        split_resamples=split_resamples,
        replicates=int(boot.replicates),
        confidence=float(boot.confidence_level),
        seed=int(boot.seed),
        pair_sets=pair_sets,
        metrics=metrics,
        input_cosine=input_cosine,
        input_change=input_change,
        notes=notes,
    )


# --------------------------------------------------------------------------- row builders


def _summary_row(
    ctx: _Context,
    entry: ActivationEntry,
    *,
    metric: str,
    scope: str,
    result: BootstrapResult,
    n_clips: int,
    pair_set: str = NOT_APPLICABLE,
    scenario: str = ALL_SCENARIOS,
    scenario_b: str | None = None,
    lag: int = 0,
    n_pairs: int = 0,
    n_nan: int = 0,
    input_control: float = float("nan"),
    preprocessing: str | None = None,
    warnings: tuple[str, ...] = (),
) -> dict[str, Any]:
    definition = get_metric(metric)
    notes = [*warnings, *result.warnings]
    if n_nan:
        notes.append(f"nan_values:{int(n_nan)}")
    if not np.isfinite(result.estimate):
        notes.append("estimate_undefined")
    return {
        "study_id": ctx.study.id,
        "config_hash": ctx.config_hash,
        "run_id": entry.run_id,
        "algorithm_id": entry.algorithm_id,
        "training_seed": int(entry.training_seed),
        "checkpoint": entry.capability,
        "checkpoint_id": entry.checkpoint_id,
        "training_step": int(entry.training_step),
        "layer": entry.layer,
        "metric": metric,
        "metric_kind": definition.kind,
        "preprocessing": preprocessing or definition.preprocessing,
        "scope": scope,
        "pair_set": pair_set,
        "scenario": scenario,
        "scenario_b": scenario if scenario_b is None else scenario_b,
        "lag": int(lag),
        "estimate": float(result.estimate),
        "ci_low": float(result.ci_low),
        "ci_high": float(result.ci_high),
        "n_clips": int(n_clips),
        "n_pairs": int(n_pairs),
        "n_nan": int(n_nan),
        "input_control": float(input_control),
        "bootstrap_replicates": int(result.replicates),
        "bootstrap_successful": int(result.successful),
        "bootstrap_seed": int(ctx.seed),
        "confidence_level": float(ctx.confidence),
        "warnings": ";".join(dict.fromkeys(notes)),
    }


def _pair_frame(
    ctx: _Context,
    entry: ActivationEntry,
    *,
    metric: str,
    pair_kind: str,
    pair_set: str,
    clip_a: np.ndarray,
    clip_b: np.ndarray,
    step_a: np.ndarray,
    step_b: np.ndarray,
    value: np.ndarray,
    lag: int = 0,
    input_change: np.ndarray | None = None,
    n_frame_pairs: np.ndarray | None = None,
    n_nan: np.ndarray | None = None,
) -> pd.DataFrame:
    ids = np.asarray(ctx.clip_ids, dtype=object)
    ids_a = ids[clip_a]
    ids_b = ids[clip_b]
    if pair_kind == "clip_pair":
        pair_ids = [clip_pair_id(a, b) for a, b in zip(ids_a, ids_b, strict=True)]
    else:
        pair_ids = [
            frame_pair_id(a, int(sa), b, int(sb))
            for a, sa, b, sb in zip(ids_a, step_a, ids_b, step_b, strict=True)
        ]
    size = len(pair_ids)
    return pd.DataFrame(
        {
            "study_id": ctx.study.id,
            "run_id": entry.run_id,
            "checkpoint": entry.capability,
            "layer": entry.layer,
            "metric": metric,
            "pair_kind": pair_kind,
            "pair_set": pair_set,
            "scenario_a": ctx.scenarios[clip_a],
            "scenario_b": ctx.scenarios[clip_b],
            "clip_id_a": ids_a,
            "clip_id_b": ids_b,
            "step_a": np.asarray(step_a, dtype=np.int64),
            "step_b": np.asarray(step_b, dtype=np.int64),
            "lag": np.full(size, int(lag), dtype=np.int64),
            "pair_id": pair_ids,
            "same_clip": np.asarray(clip_a) == np.asarray(clip_b),
            "value": np.asarray(value, dtype=np.float64),
            "input_change": (
                np.full(size, np.nan) if input_change is None else np.asarray(input_change, dtype=np.float64)
            ),
            "n_frame_pairs": (
                np.ones(size, dtype=np.int64) if n_frame_pairs is None else np.asarray(n_frame_pairs, dtype=np.int64)
            ),
            "n_nan": np.zeros(size, dtype=np.int64) if n_nan is None else np.asarray(n_nan, dtype=np.int64),
        }
    )


# --------------------------------------------------------------------------- within-checkpoint metrics


@dataclass
class _PairEstimate:
    estimate: float
    replicates: np.ndarray


def _clip_matrix_outputs(
    ctx: _Context,
    entry: ActivationEntry,
    *,
    metric: str,
    pair_set: str,
    matrix: np.ndarray,
    n_valid: np.ndarray,
    n_nan: np.ndarray,
    preprocessing: str,
    emit: bool,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], dict[tuple[str, str], _PairEstimate]]:
    """Within/between estimates, bootstrap intervals, and clip-pair diagnostics."""
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    estimates: dict[tuple[str, str], _PairEstimate] = {}
    labels = ctx.labels
    for label in labels:
        members = ctx.members[label]
        if members.size < 2:
            estimates[(label, label)] = _PairEstimate(float("nan"), np.full(ctx.replicates, np.nan))
            continue
        estimate = float(weighted_pair_mean(matrix, members, members)[0])
        replicates = weighted_pair_mean(matrix, members, members, ctx.counts)
        estimates[(label, label)] = _PairEstimate(estimate, replicates)
        if emit:
            upper = np.triu_indices(members.size, k=1)
            block = np.ix_(members, members)
            rows.append(
                _summary_row(
                    ctx,
                    entry,
                    metric=metric,
                    scope="within_scenario",
                    result=summarize(estimate, replicates, ctx.confidence),
                    n_clips=int(members.size),
                    pair_set=pair_set,
                    scenario=label,
                    n_pairs=int(n_valid[block][upper].sum()),
                    n_nan=int(n_nan[block][upper].sum()),
                    preprocessing=preprocessing,
                )
            )
    for position, label_a in enumerate(labels):
        for label_b in labels[position + 1 :]:
            rows_idx = ctx.balanced[label_a]
            cols_idx = ctx.balanced[label_b]
            if rows_idx.size == 0 or cols_idx.size == 0:
                estimates[(label_a, label_b)] = _PairEstimate(float("nan"), np.full(ctx.replicates, np.nan))
                continue
            estimate = float(weighted_pair_mean(matrix, rows_idx, cols_idx)[0])
            replicates = weighted_pair_mean(matrix, rows_idx, cols_idx, ctx.counts)
            estimates[(label_a, label_b)] = _PairEstimate(estimate, replicates)
            if emit:
                block = np.ix_(rows_idx, cols_idx)
                result = summarize(estimate, replicates, ctx.confidence)
                for first, second in ((label_a, label_b), (label_b, label_a)):
                    rows.append(
                        _summary_row(
                            ctx,
                            entry,
                            metric=metric,
                            scope="between_scenario",
                            result=result,
                            n_clips=int(rows_idx.size + cols_idx.size),
                            pair_set=pair_set,
                            scenario=first,
                            scenario_b=second,
                            n_pairs=int(n_valid[block].sum()),
                            n_nan=int(n_nan[block].sum()),
                            preprocessing=preprocessing,
                        )
                    )
    if emit:
        upper = np.triu_indices(ctx.n_clips, k=1)
        frames.append(
            _pair_frame(
                ctx,
                entry,
                metric=metric,
                pair_kind="clip_pair",
                pair_set=pair_set,
                clip_a=upper[0],
                clip_b=upper[1],
                step_a=np.full(upper[0].size, CENTROID_STEP),
                step_b=np.full(upper[0].size, CENTROID_STEP),
                value=matrix[upper],
                n_frame_pairs=n_valid[upper],
                n_nan=n_nan[upper],
            )
        )
    return rows, frames, estimates


def _gap_rows(
    ctx: _Context,
    entry: ActivationEntry,
    *,
    pair_set: str,
    base_metric: str,
    estimates: dict[tuple[str, str], _PairEstimate],
) -> list[dict[str, Any]]:
    labels = ctx.labels
    preprocessing = f"{base_metric} within minus between; matched pair set"
    if len(labels) < 2:
        return [
            _summary_row(
                ctx,
                entry,
                metric="within_between_gap",
                scope="separation",
                result=summarize(float("nan"), np.full(ctx.replicates, np.nan), ctx.confidence),
                n_clips=ctx.n_clips,
                pair_set=pair_set,
                preprocessing=preprocessing,
                warnings=("gap_undefined:single_label",),
            )
        ]

    def between(a: str, b: str) -> _PairEstimate:
        return estimates[(a, b)] if (a, b) in estimates else estimates[(b, a)]

    rows: list[dict[str, Any]] = []
    within_all = [estimates[(label, label)] for label in labels]
    between_all = [between(a, b) for i, a in enumerate(labels) for b in labels[i + 1 :]]
    estimate = vector.within_between_gap(
        float(np.mean([item.estimate for item in within_all])),
        float(np.mean([item.estimate for item in between_all])),
    )
    replicates = np.mean([item.replicates for item in within_all], axis=0) - np.mean(
        [item.replicates for item in between_all], axis=0
    )
    rows.append(
        _summary_row(
            ctx,
            entry,
            metric="within_between_gap",
            scope="separation",
            result=summarize(estimate, replicates, ctx.confidence),
            n_clips=ctx.n_clips,
            pair_set=pair_set,
            preprocessing=preprocessing,
        )
    )
    for label in labels:
        others = [between(label, other) for other in labels if other != label]
        estimate = vector.within_between_gap(
            estimates[(label, label)].estimate,
            float(np.mean([item.estimate for item in others])),
        )
        replicates = estimates[(label, label)].replicates - np.mean(
            [item.replicates for item in others], axis=0
        )
        rows.append(
            _summary_row(
                ctx,
                entry,
                metric="within_between_gap",
                scope="separation",
                result=summarize(estimate, replicates, ctx.confidence),
                n_clips=ctx.n_clips,
                pair_set=pair_set,
                scenario=label,
                preprocessing=preprocessing,
            )
        )
    return rows


def _dispersion_rows(ctx: _Context, entry: ActivationEntry, centroids: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in ctx.labels:
        members = ctx.members[label]
        if members.size == 0:
            continue
        estimate = vector.centroid_dispersion(centroids[members])

        def statistic(clip_indices: np.ndarray, positions: np.ndarray = members) -> float:
            return vector.centroid_dispersion(centroids[clip_indices[positions]])

        result = bootstrap_statistic(
            statistic, ctx.resamples, estimate=estimate, confidence_level=ctx.confidence
        )
        rows.append(
            _summary_row(
                ctx,
                entry,
                metric="centroid_dispersion",
                scope="within_scenario",
                result=result,
                n_clips=int(members.size),
                scenario=label,
            )
        )
    return rows


def _temporal_outputs(
    ctx: _Context, entry: ActivationEntry, activations: np.ndarray
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    everyone = np.arange(ctx.n_clips)
    groups = [(label, ctx.members[label]) for label in ctx.labels] + [(ALL_SCENARIOS, everyone)]
    for lag in _temporal_lags(ctx.metrics):
        if lag >= ctx.n_steps:
            rows.append(
                _summary_row(
                    ctx,
                    entry,
                    metric="temporal_cosine",
                    scope="temporal",
                    result=summarize(float("nan"), np.full(ctx.replicates, np.nan), ctx.confidence),
                    n_clips=ctx.n_clips,
                    lag=lag,
                    warnings=(f"lag_exceeds_clip_steps:{ctx.n_steps}",),
                )
            )
            continue
        values = vector.temporal_cosine(activations, lag)
        clip_means = _nanmean_rows(values)
        control_means = _nanmean_rows(ctx.input_cosine[lag])
        nan_per_clip = np.isnan(values).sum(axis=1)
        valid_per_clip = values.shape[1] - nan_per_clip
        for label, members in groups:
            if members.size == 0:
                continue
            estimate = float(weighted_clip_mean(clip_means, members)[0])
            replicates = weighted_clip_mean(clip_means, members, ctx.counts)
            rows.append(
                _summary_row(
                    ctx,
                    entry,
                    metric="temporal_cosine",
                    scope="temporal",
                    result=summarize(estimate, replicates, ctx.confidence),
                    n_clips=int(members.size),
                    scenario=label,
                    lag=lag,
                    n_pairs=int(valid_per_clip[members].sum()),
                    n_nan=int(nan_per_clip[members].sum()),
                    input_control=float(weighted_clip_mean(control_means, members)[0]),
                )
            )
        clip_index, step = np.meshgrid(everyone, np.arange(values.shape[1]), indexing="ij")
        frames.append(
            _pair_frame(
                ctx,
                entry,
                metric="temporal_cosine",
                pair_kind="temporal",
                pair_set=NOT_APPLICABLE,
                clip_a=clip_index.ravel(),
                clip_b=clip_index.ravel(),
                step_a=step.ravel(),
                step_b=step.ravel() + lag,
                value=values.ravel(),
                lag=lag,
                input_change=ctx.input_change[lag].ravel(),
            )
        )
    return rows, frames


def _silhouette_rows(ctx: _Context, entry: ActivationEntry, centroids: np.ndarray) -> list[dict[str, Any]]:
    result = vector.silhouette(centroids, ctx.scenarios)
    overall = np.full(ctx.replicates, np.nan)
    per_label = {label: np.full(ctx.replicates, np.nan) for label in ctx.labels}
    # Draw counts act as weights; a clip is never compared with its own duplicates.
    for row in range(ctx.replicates):
        replicate = vector.silhouette(centroids, ctx.scenarios, weights=ctx.counts[row])
        overall[row] = replicate.score
        for label in ctx.labels:
            per_label[label][row] = replicate.per_label.get(label, np.nan)
    rows = [
        _summary_row(
            ctx,
            entry,
            metric="silhouette",
            scope="separation",
            result=summarize(result.score, overall, ctx.confidence),
            n_clips=ctx.n_clips,
            warnings=result.warnings,
        )
    ]
    for label in ctx.labels:
        rows.append(
            _summary_row(
                ctx,
                entry,
                metric="silhouette",
                scope="separation",
                result=summarize(result.per_label.get(label, float("nan")), per_label[label], ctx.confidence),
                n_clips=int(ctx.members[label].size),
                scenario=label,
                warnings=result.warnings,
            )
        )
    return rows


def _nearest_centroid_rows(
    ctx: _Context, entry: ActivationEntry, centroids: np.ndarray
) -> list[dict[str, Any]]:
    train = np.flatnonzero(ctx.splits == "train")
    test = np.flatnonzero(ctx.splits == "test")
    result = vector.nearest_centroid_accuracy(
        centroids[train], ctx.scenarios[train], centroids[test], ctx.scenarios[test], ctx.labels
    )

    def statistic(clip_indices: np.ndarray) -> float:
        return vector.nearest_centroid_accuracy(
            centroids[clip_indices[train]],
            ctx.scenarios[train],
            centroids[clip_indices[test]],
            ctx.scenarios[test],
            ctx.labels,
        ).accuracy

    boot = bootstrap_statistic(
        statistic, ctx.split_resamples, estimate=result.accuracy, confidence_level=ctx.confidence
    )
    rows = [
        _summary_row(
            ctx,
            entry,
            metric="nearest_centroid_accuracy",
            scope="separation",
            result=boot,
            n_clips=int(test.size),
            n_pairs=int(test.size),
            warnings=result.warnings,
        )
    ]
    for i, true_label in enumerate(ctx.labels):
        total = int(result.confusion[i].sum())
        for j, predicted_label in enumerate(ctx.labels):
            count = int(result.confusion[i, j])
            rate = count / total if total else float("nan")
            rows.append(
                _summary_row(
                    ctx,
                    entry,
                    metric="nearest_centroid_accuracy",
                    scope="confusion",
                    result=_no_interval(rate, ctx.confidence),
                    n_clips=total,
                    n_pairs=count,
                    scenario=true_label,
                    scenario_b=predicted_label,
                    preprocessing="row-normalized by true label; n_pairs is the raw count",
                )
            )
    return rows


def _standardizers(
    ctx: _Context, frames: np.ndarray
) -> tuple[vector.Standardizer, vector.Standardizer, list[str]]:
    """(train-fitted standardizer, normalized_euclidean standardizer, warnings)."""
    notes: list[str] = []
    train_frames = np.repeat(ctx.splits == "train", ctx.n_steps)
    if train_frames.any():
        train_fit = vector.fit_standardizer(frames[train_frames], fitted_on="train_clips")
    else:
        train_fit = vector.fit_standardizer(frames, fitted_on="all_clips")
        notes.append("no_train_clips:standardized_on_all_clips")
    choice = ctx.metrics.get("normalized_euclidean", {}).get("standardize_on", "train_clips")
    if choice not in STANDARDIZE_CHOICES:
        raise AnalysisError(
            f"Unknown standardize_on {choice!r}; valid: {', '.join(STANDARDIZE_CHOICES)}"
        )
    distance_fit = train_fit if choice == "train_clips" else vector.fit_standardizer(frames, fitted_on="all_clips")
    if train_fit.n_constant:
        notes.append(f"constant_features:{train_fit.n_constant}")
    return train_fit, distance_fit, notes


def _within_checkpoint(
    ctx: _Context, entry: ActivationEntry, activations: np.ndarray
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    n_clips, n_steps, n_features = activations.shape
    frames_matrix = activations.reshape(n_clips * n_steps, n_features)
    train_fit, distance_fit, standard_notes = _standardizers(ctx, frames_matrix)
    centroids_raw = vector.clip_centroids(activations)
    centroids_std = train_fit.transform(frames_matrix).reshape(n_clips, n_steps, n_features).mean(axis=1)

    rows: list[dict[str, Any]] = []
    pair_frames: list[pd.DataFrame] = []

    gap_enabled = ctx.enabled("within_between_gap")
    base_metric = ctx.metrics.get("within_between_gap", {}).get("base_metric", "cosine")
    if gap_enabled and base_metric not in ("cosine", "correlation"):
        raise AnalysisError(
            f"within_between_gap base_metric must be cosine or correlation, got {base_metric!r}"
        )

    reps = representative_frames(ctx.clip_ids, ctx.scenarios.tolist(), ctx.labels, n_steps)
    rep_clip = np.asarray([frame.clip_index for frame in reps], dtype=np.int64)
    rep_step = np.asarray([frame.step for frame in reps], dtype=np.int64)
    rep_flat = rep_clip * n_steps + rep_step
    rep_upper = np.triu_indices(len(reps), k=0)

    for metric in PAIRWISE_METRICS:
        emit = ctx.enabled(metric)
        needed_for_gap = gap_enabled and metric == base_metric
        if not emit and not needed_for_gap:
            continue
        if metric == "cosine":
            frame_values = vector.cosine_similarity_matrix(frames_matrix)
            centroid_values = vector.cosine_similarity_matrix(centroids_raw)
            preprocessing = get_metric(metric).preprocessing
        elif metric == "correlation":
            frame_values = vector.correlation_similarity_matrix(frames_matrix)
            centroid_values = vector.correlation_similarity_matrix(centroids_raw)
            preprocessing = get_metric(metric).preprocessing
        else:
            standardized = distance_fit.transform(frames_matrix)
            frame_values = vector.normalized_euclidean_matrix(standardized)
            centroid_values = vector.normalized_euclidean_matrix(
                standardized.reshape(n_clips, n_steps, n_features).mean(axis=1)
            )
            preprocessing = f"zscore_fit_on_{distance_fit.fitted_on}; euclidean / sqrt(features)"
        for pair_set in ctx.pair_sets:
            if pair_set == "clip_centroids":
                matrix = centroid_values
                n_valid = (~np.isnan(matrix)).astype(np.int64)
                n_nan = np.isnan(matrix).astype(np.int64)
            else:
                matrix, n_valid, n_nan = vector.reduce_clip_blocks(frame_values, n_clips, n_steps, pair_set)
            metric_rows, frames, estimates = _clip_matrix_outputs(
                ctx,
                entry,
                metric=metric,
                pair_set=pair_set,
                matrix=matrix,
                n_valid=n_valid,
                n_nan=n_nan,
                preprocessing=preprocessing,
                emit=emit,
            )
            rows.extend(metric_rows)
            pair_frames.extend(frames)
            if needed_for_gap:
                rows.extend(
                    _gap_rows(ctx, entry, pair_set=pair_set, base_metric=base_metric, estimates=estimates)
                )
        if emit and len(reps) > 0:
            pair_frames.append(
                _pair_frame(
                    ctx,
                    entry,
                    metric=metric,
                    pair_kind="representative_frame",
                    pair_set=NOT_APPLICABLE,
                    clip_a=rep_clip[rep_upper[0]],
                    clip_b=rep_clip[rep_upper[1]],
                    step_a=rep_step[rep_upper[0]],
                    step_b=rep_step[rep_upper[1]],
                    value=frame_values[rep_flat[rep_upper[0]], rep_flat[rep_upper[1]]],
                )
            )
        del frame_values

    if ctx.enabled("linear_cka"):
        if n_steps < 3:
            ctx.notes.append("clip_pair_cka_skipped:needs_at_least_3_steps_per_clip")
        else:
            matrix = geometry.clip_pair_cka(activations)
            metric_rows, frames, _ = _clip_matrix_outputs(
                ctx,
                entry,
                metric="linear_cka",
                pair_set=CLIP_ALIGNED_STEPS,
                matrix=matrix,
                n_valid=(~np.isnan(matrix)).astype(np.int64),
                n_nan=np.isnan(matrix).astype(np.int64),
                preprocessing=(
                    f"clip-pair CKA over {n_steps} aligned relative steps; columns centered per clip"
                ),
                emit=True,
            )
            rows.extend(metric_rows)
            pair_frames.extend(frames)
            if len(reps) > 0:
                pair_frames.append(
                    _pair_frame(
                        ctx,
                        entry,
                        metric="linear_cka",
                        pair_kind="representative_frame",
                        pair_set=CLIP_ALIGNED_STEPS,
                        clip_a=rep_clip[rep_upper[0]],
                        clip_b=rep_clip[rep_upper[1]],
                        step_a=rep_step[rep_upper[0]],
                        step_b=rep_step[rep_upper[1]],
                        value=matrix[rep_clip[rep_upper[0]], rep_clip[rep_upper[1]]],
                    )
                )

    if ctx.enabled("centroid_dispersion"):
        rows.extend(_dispersion_rows(ctx, entry, centroids_std))
    if ctx.enabled("temporal_cosine"):
        temporal_rows, temporal_frames = _temporal_outputs(ctx, entry, activations)
        rows.extend(temporal_rows)
        pair_frames.extend(temporal_frames)
    if ctx.enabled("silhouette"):
        rows.extend(_silhouette_rows(ctx, entry, centroids_std))
    if ctx.enabled("nearest_centroid_accuracy"):
        rows.extend(_nearest_centroid_rows(ctx, entry, centroids_std))

    if standard_notes:
        suffix = ";".join(standard_notes)
        standardized_metrics = {
            "normalized_euclidean",
            "centroid_dispersion",
            "silhouette",
            "nearest_centroid_accuracy",
        }
        for row in rows:
            if row["metric"] in standardized_metrics:
                row["warnings"] = ";".join(filter(None, [row["warnings"], suffix]))
    return rows, pair_frames


# --------------------------------------------------------------------------- embeddings


def _embedding_frames(
    ctx: _Context, entry: ActivationEntry, activations: np.ndarray
) -> list[pd.DataFrame]:
    n_clips, n_steps, n_features = activations.shape
    samples = activations.reshape(n_clips * n_steps, n_features)
    frames: list[pd.DataFrame] = []
    for spec in ctx.study.analysis.embeddings:
        if spec.id not in EMBEDDING_IDS:
            raise AnalysisError(
                f"Unknown embedding {spec.id!r}; valid embeddings: {', '.join(EMBEDDING_IDS)}"
            )
        if not spec.enabled:
            continue
        kwargs = dict(spec.kwargs)
        requested = int(kwargs.get("components", 2))
        if requested < 1:
            raise AnalysisError("embedding components must be at least 1")
        ratio_x = ratio_y = float("nan")
        if spec.id == "pca":
            unknown = sorted(set(kwargs) - {"components"})
            if unknown:
                raise AnalysisError("Unknown kwargs for embedding 'pca'", context={"unknown": unknown})
            from sklearn.decomposition import PCA

            components = max(1, min(requested, samples.shape[0] - 1, n_features))
            model = PCA(n_components=components, svd_solver="full")
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                coordinates = model.fit_transform(samples)
            ratios = np.asarray(model.explained_variance_ratio_, dtype=np.float64)
            ratio_x = float(ratios[0])
            ratio_y = float(ratios[1]) if components > 1 else float("nan")
        else:
            unknown = sorted(set(kwargs) - {"components", "neighbors", "min_dist", "seed"})
            if unknown:
                raise AnalysisError("Unknown kwargs for embedding 'umap'", context={"unknown": unknown})
            try:
                import umap
            except ImportError as exc:
                raise AnalysisError(
                    "UMAP embedding is enabled but umap-learn is not installed",
                    context={"hint": "install the 'umap' extra or set enabled: false"},
                ) from exc
            components = requested
            model = umap.UMAP(
                n_components=components,
                n_neighbors=int(min(kwargs.get("neighbors", 15), samples.shape[0] - 1)),
                min_dist=float(kwargs.get("min_dist", 0.1)),
                random_state=int(kwargs.get("seed", 0)),
            )
            coordinates = np.asarray(model.fit_transform(samples), dtype=np.float64)
        x = coordinates[:, 0]
        y = coordinates[:, 1] if coordinates.shape[1] > 1 else np.zeros_like(x)
        parameters = json.dumps({"id": spec.id, **kwargs}, sort_keys=True)
        clip_ids = np.asarray(ctx.clip_ids, dtype=object)
        common = {
            "study_id": ctx.study.id,
            "run_id": entry.run_id,
            "checkpoint": entry.capability,
            "layer": entry.layer,
            "embedding": spec.id,
            "explained_variance_x": ratio_x,
            "explained_variance_y": ratio_y,
            "n_components": int(coordinates.shape[1]),
            "parameters": parameters,
        }
        frames.append(
            pd.DataFrame(
                {
                    **common,
                    "level": "frame",
                    "clip_id": np.repeat(clip_ids, n_steps),
                    "scenario": np.repeat(ctx.scenarios, n_steps),
                    "split": np.repeat(ctx.splits, n_steps),
                    "step": np.tile(np.arange(n_steps, dtype=np.int64), n_clips),
                    "x": x,
                    "y": y,
                }
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    **common,
                    "level": "clip_centroid",
                    "clip_id": clip_ids,
                    "scenario": ctx.scenarios,
                    "split": ctx.splits,
                    "step": np.full(n_clips, CENTROID_STEP, dtype=np.int64),
                    "x": x.reshape(n_clips, n_steps).mean(axis=1),
                    "y": y.reshape(n_clips, n_steps).mean(axis=1),
                }
            )
        )
    return frames


# --------------------------------------------------------------------------- cross-checkpoint geometry


def _ids_hash(sample_ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def _cross_row(
    ctx: _Context,
    a: ActivationEntry,
    b: ActivationEntry,
    *,
    metric: str,
    metric_kind: str,
    preprocessing: str,
    plan: _SamplePlan,
    result: BootstrapResult,
    n_samples: int,
    ids_hash: str,
    applicable: bool = True,
    reason: str = "",
    diagnostic: bool = False,
    warnings: tuple[str, ...] = (),
) -> dict[str, Any]:
    notes = [*warnings, *result.warnings]
    if applicable and not np.isfinite(result.estimate):
        notes.append("estimate_undefined")
    return {
        "study_id": ctx.study.id,
        "config_hash": ctx.config_hash,
        "run_id_a": a.run_id,
        "run_id_b": b.run_id,
        "algorithm_id_a": a.algorithm_id,
        "algorithm_id_b": b.algorithm_id,
        "checkpoint_a": a.capability,
        "checkpoint_b": b.capability,
        "checkpoint_id_a": a.checkpoint_id,
        "checkpoint_id_b": b.checkpoint_id,
        "training_step_a": int(a.training_step),
        "training_step_b": int(b.training_step),
        "layer_a": a.layer,
        "layer_b": b.layer,
        "metric": metric,
        "metric_kind": metric_kind,
        "preprocessing": preprocessing,
        "sample_unit": plan.label,
        "estimate": float(result.estimate),
        "ci_low": float(result.ci_low),
        "ci_high": float(result.ci_high),
        "n_clips": int(ctx.n_clips),
        "n_samples": int(n_samples),
        "n_features_a": int(a.feature_dim),
        "n_features_b": int(b.feature_dim),
        "sample_ids_sha256": ids_hash,
        "applicable": bool(applicable),
        "reason": reason,
        "diagnostic": bool(diagnostic),
        "bootstrap_replicates": int(result.replicates),
        "bootstrap_successful": int(result.successful),
        "bootstrap_seed": int(ctx.seed),
        "confidence_level": float(ctx.confidence),
        "warnings": ";".join(dict.fromkeys(notes)),
    }


def _mirror(row: dict[str, Any]) -> dict[str, Any]:
    mirrored = dict(row)
    for key in list(row):
        if key.endswith("_a"):
            other = key[:-2] + "_b"
            mirrored[key], mirrored[other] = row[other], row[key]
    return mirrored


def _no_interval(estimate: float, confidence: float) -> BootstrapResult:
    return BootstrapResult(
        estimate=estimate,
        ci_low=float("nan"),
        ci_high=float("nan"),
        replicates=0,
        successful=0,
        confidence_level=confidence,
    )


def _cross_checkpoint(
    ctx: _Context,
    entries: list[ActivationEntry],
    representations: dict[tuple[str, str], geometry.Representation],
    plans: dict[str, _SamplePlan],
    logger: Any,
    extra: dict[str, str],
) -> list[dict[str, Any]]:
    """Geometry between every pair of (run, capability) at every layer pair.

    Same-layer comparisons between different checkpoints carry clip-bootstrap
    intervals; the layer-by-layer grid and self-comparisons are point estimates.
    """
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], dict[str, ActivationEntry]] = {}
    for entry in entries:
        groups.setdefault((entry.run_id, entry.capability), {})[entry.layer] = entry
    group_keys = list(groups)
    layer_rank: dict[str, int] = {}
    for entry in entries:
        layer_rank.setdefault(entry.layer, len(layer_rank))

    for metric in GEOMETRY_METRICS:
        if not ctx.enabled(metric):
            continue
        definition = get_metric(metric)
        kwargs = ctx.metrics[metric]
        plan = plans[metric]
        options: dict[str, Any] = {}
        preprocessing = definition.preprocessing
        if metric == "rsa_spearman":
            options["first_order_distance"] = kwargs.get("first_order_distance", "correlation")
            preprocessing = f"RDM distance={options['first_order_distance']}; Spearman of upper triangles"
        if metric == "svcca":
            options["variance_retained"] = float(kwargs.get("variance_retained", 0.99))
            preprocessing = f"centered; SVD retaining {options['variance_retained']:g} variance; mean CCA"
        started = time.perf_counter()
        for ia, key_a in enumerate(group_keys):
            for ib in range(ia, len(group_keys)):
                key_b = group_keys[ib]
                for layer_a, entry_a in groups[key_a].items():
                    for layer_b, entry_b in groups[key_b].items():
                        if ia == ib and layer_rank[layer_a] > layer_rank[layer_b]:
                            continue
                        rep_a = representations[(entry_a.path, plan.label)]
                        rep_b = representations[(entry_b.path, plan.label)]
                        identical = ia == ib and layer_a == layer_b
                        notes: list[str] = []
                        if rep_a.n_samples < max(rep_a.n_features, rep_b.n_features):
                            notes.append("fewer_samples_than_features")
                        ids_hash = _ids_hash(rep_a.sample_ids)
                        try:
                            estimate = geometry.evaluate_geometry(metric, rep_a, rep_b, **options)
                        except geometry.InapplicableMetric as exc:
                            row = _cross_row(
                                ctx,
                                entry_a,
                                entry_b,
                                metric=metric,
                                metric_kind=definition.kind,
                                preprocessing=preprocessing,
                                plan=plan,
                                result=_no_interval(float("nan"), ctx.confidence),
                                n_samples=rep_a.n_samples,
                                ids_hash=ids_hash,
                                applicable=False,
                                reason=str(exc),
                            )
                            rows.append(row)
                            if not identical:
                                rows.append(_mirror(row))
                            continue
                        if ia != ib and layer_a == layer_b and ctx.replicates > 0:

                            def statistic(
                                clip_indices: np.ndarray,
                                left: geometry.Representation = rep_a,
                                right: geometry.Representation = rep_b,
                            ) -> float:
                                return geometry.evaluate_geometry(
                                    metric,
                                    left,
                                    right,
                                    index=plan.sample_index(clip_indices),
                                    **options,
                                )

                            result = bootstrap_statistic(
                                statistic,
                                ctx.resamples,
                                estimate=estimate,
                                confidence_level=ctx.confidence,
                            )
                        else:
                            result = _no_interval(estimate, ctx.confidence)
                        row = _cross_row(
                            ctx,
                            entry_a,
                            entry_b,
                            metric=metric,
                            metric_kind=definition.kind,
                            preprocessing=preprocessing,
                            plan=plan,
                            result=result,
                            n_samples=rep_a.n_samples,
                            ids_hash=ids_hash,
                            warnings=tuple(notes),
                        )
                        rows.append(row)
                        if not identical:
                            rows.append(_mirror(row))
        logger.info(
            "cross-checkpoint %s computed in %.1fs", metric, time.perf_counter() - started, extra=extra
        )

    # Raw-coordinate cosine: a labelled diagnostic, not a primary result. Latent bases
    # can rotate during training, so low values do not imply different geometry.
    diagnostic_metric = next((metric for metric in GEOMETRY_METRICS if ctx.enabled(metric)), None)
    if diagnostic_metric is not None:
        plan = plans[diagnostic_metric]
        for ia, key_a in enumerate(group_keys):
            for ib in range(ia + 1, len(group_keys)):
                key_b = group_keys[ib]
                for layer, entry_a in groups[key_a].items():
                    entry_b = groups[key_b].get(layer)
                    if entry_b is None or entry_a.feature_dim != entry_b.feature_dim:
                        continue
                    rep_a = representations[(entry_a.path, plan.label)]
                    rep_b = representations[(entry_b.path, plan.label)]
                    geometry.validate_aligned_samples(rep_a.sample_ids, rep_b.sample_ids)
                    numerator = np.einsum("ij,ij->i", rep_a.matrix, rep_b.matrix)
                    denominator = np.linalg.norm(rep_a.matrix, axis=1) * np.linalg.norm(rep_b.matrix, axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        sample_cosine = np.where(denominator < vector.EPS, np.nan, numerator / denominator)
                    clip_means = _nanmean_rows(sample_cosine.reshape(ctx.n_clips, plan.per_clip))
                    everyone = np.arange(ctx.n_clips)
                    estimate = float(weighted_clip_mean(clip_means, everyone)[0])
                    replicates = weighted_clip_mean(clip_means, everyone, ctx.counts)
                    row = _cross_row(
                        ctx,
                        entry_a,
                        entry_b,
                        metric=RAW_COSINE_DIAGNOSTIC,
                        metric_kind="similarity",
                        preprocessing="same-sample cosine in raw coordinates (basis-dependent diagnostic)",
                        plan=plan,
                        result=summarize(estimate, replicates, ctx.confidence),
                        n_samples=rep_a.n_samples,
                        ids_hash=_ids_hash(rep_a.sample_ids),
                        diagnostic=True,
                    )
                    rows.append(row)
                    rows.append(_mirror(row))
    return rows


# --------------------------------------------------------------------------- orchestration


def is_current_metrics_artifact(directory: Path, stage_hash: str, activations_sha: str) -> bool:
    """True when the metrics artifact verifies and matches config and activations."""
    try:
        manifest = verify_artifact_dir(directory)
    except ArtifactError:
        return False
    metadata = manifest.metadata
    return (
        metadata.get("stage_config_hash") == stage_hash
        and metadata.get("activations_manifest_sha256") == activations_sha
    )


def _frame(rows: list[dict[str, Any]] | list[pd.DataFrame]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    if isinstance(rows[0], pd.DataFrame):
        return pd.concat(rows, ignore_index=True)  # type: ignore[arg-type]
    return pd.DataFrame(rows)


def run_analysis(resolved: ResolvedConfig, force: bool = False) -> Path:
    """Compute all configured metrics and embeddings and publish the metrics artifact.

    An existing artifact is reused only when it verifies and matches both the
    analysis configuration and the activation artifact. ``force`` publishes a
    new version; nothing is overwritten in place.
    """
    study = require_study(resolved, stage="analyze")
    logger = get_logger("analyze")
    extra = log_extra(stage="analyze", config_hash=resolved.config_hash)
    root = study_root(study)

    activations_dir = latest_artifact_dir(root, ACTIVATIONS_ARTIFACT_ID)
    if activations_dir is None:
        raise AnalysisError(
            f"No activation artifact found under {root}",
            context={"regenerate": EXTRACT_COMMAND},
        )
    bundle = load_activation_bundle(activations_dir)
    if bundle.metadata.get("stage_config_hash") != extraction_stage_hash(resolved):
        raise AnalysisError(
            f"Activation artifact {activations_dir} was extracted with a different configuration",
            context={"regenerate": EXTRACT_COMMAND + " --force"},
        )
    if not bundle.entries:
        raise AnalysisError(f"Activation artifact {activations_dir} contains no arrays")

    stage_hash = analysis_stage_hash(resolved)
    activations_sha = sha256_file(activations_dir / "manifest.json")
    existing = latest_artifact_dir(root, METRICS_ARTIFACT_ID)
    if existing is not None and not force:
        if is_current_metrics_artifact(existing, stage_hash, activations_sha):
            logger.info("metrics up to date at %s", existing, extra=extra)
            return existing
        if study.artifacts.collision == "error":
            raise ArtifactError(
                f"Existing metrics artifact is stale or invalid: {existing}",
                context={"hint": "re-run with --force to publish a new version"},
            )

    ctx = _build_context(resolved, study, bundle)
    plans = {
        metric: _sample_plan(ctx.n_steps, ctx.metrics[metric])
        for metric in GEOMETRY_METRICS
        if ctx.enabled(metric)
    }
    distinct_plans = {plan.label: plan for plan in plans.values()}
    logger.info(
        "analyzing %d activation array(s): %d clips x %d steps, %d bootstrap replicates",
        len(bundle.entries),
        ctx.n_clips,
        ctx.n_steps,
        ctx.replicates,
        extra=extra,
    )

    summary_rows: list[dict[str, Any]] = []
    pair_frames: list[pd.DataFrame] = []
    embedding_frames: list[pd.DataFrame] = []
    representations: dict[tuple[str, str], geometry.Representation] = {}
    for entry in bundle.entries:
        started = time.perf_counter()
        activations = np.asarray(bundle.load(entry), dtype=np.float64)
        if not np.isfinite(activations).all():
            raise AnalysisError(
                f"Activation array {entry.path} contains non-finite values",
                context={"regenerate": EXTRACT_COMMAND + " --force"},
            )
        rows, frames = _within_checkpoint(ctx, entry, activations)
        summary_rows.extend(rows)
        pair_frames.extend(frames)
        embedding_frames.extend(_embedding_frames(ctx, entry, activations))
        for label, plan in distinct_plans.items():
            representations[(entry.path, label)] = geometry.Representation(
                plan.matrix(activations), plan.sample_ids(ctx.clip_ids)
            )
        logger.info(
            "analyzed %s/%s/%s in %.1fs",
            entry.run_id,
            entry.capability,
            entry.layer,
            time.perf_counter() - started,
            extra=extra,
        )
    cross_rows = _cross_checkpoint(ctx, list(bundle.entries), representations, plans, logger, extra)

    summaries = ordered_table(
        _frame(summary_rows),
        SUMMARY_COLUMNS,
        ["run_id", "training_step", "checkpoint", "layer", "metric", "scope", "pair_set", "scenario", "scenario_b", "lag"],
    )
    cross = ordered_table(
        _frame(cross_rows),
        CROSS_CHECKPOINT_COLUMNS,
        ["metric", "run_id_a", "training_step_a", "checkpoint_a", "layer_a", "run_id_b", "training_step_b", "checkpoint_b", "layer_b"],
    )
    pairs = ordered_table(
        _frame(pair_frames),
        FRAME_PAIR_COLUMNS,
        ["run_id", "checkpoint", "layer", "metric", "pair_kind", "pair_set", "lag", "pair_id"],
    )
    embeddings = ordered_table(
        _frame(embedding_frames),
        EMBEDDING_COLUMNS,
        ["run_id", "checkpoint", "layer", "embedding", "level", "clip_id", "step"],
    )

    nan_counts = {
        "summaries_estimate": int(summaries["estimate"].isna().sum()) if len(summaries) else 0,
        "cross_checkpoint_estimate": (
            int((cross["estimate"].isna() & cross["applicable"].astype(bool)).sum()) if len(cross) else 0
        ),
        "frame_pairs_value": int(pairs["value"].isna().sum()) if len(pairs) else 0,
    }
    for name, count in nan_counts.items():
        if count:
            logger.warning("%d NaN value(s) in %s (kept and reported)", count, name, extra=extra)
    biased_intervals = {
        name: int(table["warnings"].str.contains(ESTIMATE_OUTSIDE_INTERVAL, regex=False).sum())
        for name, table in (("summaries", summaries), ("cross_checkpoint", cross))
        if len(table)
    }
    for name, count in biased_intervals.items():
        if count:
            logger.warning(
                "%d %s row(s) have a point estimate outside the bootstrap percentile interval "
                "(resampling bias; rows are flagged)",
                count,
                name,
                extra=extra,
            )

    reps = representative_frames(ctx.clip_ids, ctx.scenarios.tolist(), ctx.labels, ctx.n_steps)
    store = ArtifactStore(
        root,
        artifact_id=METRICS_ARTIFACT_ID,
        config_hash=resolved.config_hash,
        stage="analyze",
        collision=study.artifacts.collision,
    )
    artifact_dir = store.target
    obs_file = observations_path(study)

    def writer(tmp: Path) -> ArtifactManifest:
        tables = {
            "summaries.parquet": (summaries, "summaries"),
            "cross_checkpoint.parquet": (cross, "cross_checkpoint"),
            "frame_pairs.parquet": (pairs, "frame_pairs"),
            "embeddings.parquet": (embeddings, "embeddings"),
        }
        outputs: list[ArtifactOutput] = []
        for filename, (table, role) in tables.items():
            table.to_parquet(tmp / filename, index=False)
            outputs.append(
                ArtifactOutput(path=filename, sha256="", role=role, schema_version=TABLE_SCHEMA_VERSION)
            )
        write_resolved_config(tmp / "resolved-config.yaml", resolved.canonical)
        outputs.append(ArtifactOutput(path="resolved-config.yaml", sha256="", role="resolved_config"))
        return store.build_manifest(
            inputs=[
                make_input(activations_dir / "manifest.json", role="activations_manifest", artifact_dir=artifact_dir),
                make_input(Path(study.scenarios.manifest), role="scenario_manifest", artifact_dir=artifact_dir),
                make_input(obs_file, role="observations", artifact_dir=artifact_dir),
            ],
            outputs=outputs,
            metadata={
                "schema": METRICS_SCHEMA,
                "study_id": study.id,
                "stage_config_hash": stage_hash,
                "activations_dir": activations_dir.name,
                "activations_manifest_sha256": activations_sha,
                "synthetic": bool(bundle.metadata.get("synthetic", False)),
                "labels": list(ctx.labels),
                "n_clips": ctx.n_clips,
                "clip_steps": ctx.n_steps,
                "clips_per_label": {label: int(ctx.members[label].size) for label in ctx.labels},
                "pair_sets": list(ctx.pair_sets),
                "metrics": [item for item in describe_metrics() if item["id"] in ctx.metrics],
                "metric_kwargs": ctx.metrics,
                "bootstrap": {
                    "unit": study.analysis.bootstrap.unit,
                    "stratified_by": "scenario",
                    "replicates": ctx.replicates,
                    "confidence_level": ctx.confidence,
                    "seed": ctx.seed,
                },
                "cross_checkpoint_samples": {metric: plan.label for metric, plan in plans.items()},
                "representative_frames": [
                    {
                        "scenario": frame.scenario,
                        "clip_id": frame.clip_id,
                        "step": frame.step,
                        "frame_rank": frame.frame_rank,
                        "offset_rank": frame.offset_rank,
                    }
                    for frame in reps
                ],
                "row_counts": {
                    "summaries": int(len(summaries)),
                    "cross_checkpoint": int(len(cross)),
                    "frame_pairs": int(len(pairs)),
                    "embeddings": int(len(embeddings)),
                },
                "nan_counts": nan_counts,
                "estimate_outside_interval": biased_intervals,
                "notes": list(dict.fromkeys(ctx.notes)),
            },
            git_commit=try_git_commit(REPO_ROOT),
            package_versions=capture_package_versions(["numpy", "pandas", "scipy", "sklearn", "pyarrow"]),
        )

    out = store.publish(writer, force=force)
    logger.info(
        "published metrics to %s (%d summary rows, %d cross-checkpoint rows)",
        out,
        len(summaries),
        len(cross),
        extra=extra,
    )
    return out
