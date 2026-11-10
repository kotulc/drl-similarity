"""Typed, read-only artifact readers for the dashboard.

This module does not import Streamlit, a training backend, or the analysis
code. It reads the tables written by ``drl-sim analyze`` plus run and scenario
metadata, validates schemas, and exposes small filter helpers. Pages never read
files directly.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from drl_similarity.artifacts.manifest import ArtifactManifest, sha256_file, verify_artifact_dir
from drl_similarity.config import REPO_ROOT, DashboardConfig, ResolvedConfig, StudyConfig
from drl_similarity.errors import ArtifactError, DrlSimilarityError
from drl_similarity.scenarios.schema import ScenarioManifest, load_manifest
from drl_similarity.study_io import (
    CAPABILITIES,
    METRICS_ARTIFACT_ID,
    RunInfo,
    analysis_stage_hash,
    artifact_versions,
    load_runs,
    observations_path,
    require_study,
    study_root,
)
from drl_similarity.tables import (
    ANALYZE_COMMAND,
    CLIP_ALIGNED_STEPS,
    CROSS_CHECKPOINT_COLUMNS,
    EMBEDDING_COLUMNS,
    EVALUATION_COLUMNS,
    FRAME_PAIR_COLUMNS,
    METRICS_SCHEMA,
    PAIR_KINDS,
    SUMMARY_COLUMNS,
    SUMMARY_SCOPES,
    TABLE_SCHEMA_VERSION,
    require_columns,
)


SEVERITIES: tuple[str, ...] = ("ok", "info", "warning", "error")

TRAIN_COMMAND = "drl-sim train --config <experiment.yaml>"
SELECT_COMMAND = "drl-sim select-checkpoints --config <experiment.yaml>"
VALIDATE_SCENARIOS_COMMAND = "drl-sim validate-scenarios --config <study.yaml>"
EXTRACT_COMMAND = "drl-sim extract --config <study.yaml>"
RUN_STUDY_COMMAND = "drl-sim run-study --config <study.yaml>"

#: Metrics shown in every case-matrix and frame-pair cell, in display order.
CELL_METRICS: tuple[str, ...] = ("cosine", "correlation", "linear_cka", "normalized_euclidean")

METRIC_TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "summaries": ("summaries.parquet", SUMMARY_COLUMNS),
    "cross_checkpoint": ("cross_checkpoint.parquet", CROSS_CHECKPOINT_COLUMNS),
    "frame_pairs": ("frame_pairs.parquet", FRAME_PAIR_COLUMNS),
    "embeddings": ("embeddings.parquet", EMBEDDING_COLUMNS),
}


@dataclass(frozen=True)
class QualityIssue:
    """One data-quality check result shown in the header badge and overview page."""

    severity: str  # one of SEVERITIES
    code: str
    message: str


@dataclass(frozen=True)
class StudySummary:
    study_id: str
    title: str
    config_hash: str
    config_path: str
    labels: tuple[str, ...]
    capabilities: tuple[str, ...]
    layers: tuple[str, ...]
    metrics: tuple[str, ...]
    pair_sets: tuple[str, ...]
    clip_steps: int
    burn_in_steps: int
    n_clips: int
    clips_per_label: tuple[tuple[str, int], ...]
    n_training_seeds: int
    synthetic: bool
    metrics_dir: str
    metrics_version: str
    metrics_manifest_sha256: str
    activations_dir: str | None
    observations_file: str
    scenario_manifest: str
    generated_at: str
    bootstrap: tuple[tuple[str, Any], ...]

    @property
    def bootstrap_info(self) -> dict[str, Any]:
        return dict(self.bootstrap)


@dataclass(frozen=True)
class DashboardData:
    """Immutable bundle of everything the pages render."""

    study: StudySummary
    dashboard: DashboardConfig
    runs: pd.DataFrame
    selected: pd.DataFrame
    evaluations: pd.DataFrame
    clips: pd.DataFrame
    summaries: pd.DataFrame
    cross_checkpoint: pd.DataFrame
    frame_pairs: pd.DataFrame
    embeddings: pd.DataFrame
    layers: pd.DataFrame
    representative_frames: pd.DataFrame
    metric_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    manifests: dict[str, dict[str, Any]] = field(default_factory=dict)
    selection: dict[str, dict[str, Any]] = field(default_factory=dict)
    resolved_yaml: str = ""
    source_configs: tuple[tuple[str, str], ...] = ()
    quality: tuple[QualityIssue, ...] = ()

    @property
    def status(self) -> str:
        """``error``, ``warning``, or ``ok`` — drives the header badge colour."""
        severities = {issue.severity for issue in self.quality}
        if "error" in severities:
            return "error"
        if "warning" in severities:
            return "warning"
        return "ok"


# --------------------------------------------------------------------------- discovery


def find_metrics_dir(resolved: ResolvedConfig) -> Path | None:
    """Latest published metrics artifact directory for the study, if any."""
    study = require_study(resolved, stage="dashboard")
    versions = artifact_versions(study_root(study), METRICS_ARTIFACT_ID)
    return versions[-1][1] if versions else None


def metrics_cache_key(resolved: ResolvedConfig) -> str:
    """Checksum of the metrics manifest; the explicit cache key for readers."""
    directory = find_metrics_dir(resolved)
    if directory is None:
        return "missing"
    return sha256_file(directory / "manifest.json")


@dataclass
class _Inspection:
    problems: list[str] = field(default_factory=list)
    runs: list[RunInfo] = field(default_factory=list)
    evaluations: dict[str, pd.DataFrame] = field(default_factory=dict)
    scenario_manifest: ScenarioManifest | None = None
    metrics_dir: Path | None = None
    metrics_manifest: ArtifactManifest | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def _describe(exc: DrlSimilarityError) -> str:
    text = str(exc.args[0]) if exc.args else str(exc)
    context = getattr(exc, "context", None) or {}
    hints = [f"{key}: {context[key]}" for key in ("hint", "regenerate", "missing_columns") if key in context]
    return text + (f" ({'; '.join(hints)})" if hints else "")


def _check_enum(frame: pd.DataFrame, column: str, valid: tuple[str, ...], *, name: str) -> str | None:
    if frame.empty:
        return None
    found = set(frame[column].dropna().astype(str).unique())
    unknown = sorted(found - set(valid))
    if unknown:
        return (
            f"{name}: column {column!r} contains unknown value(s) {unknown}; valid: {list(valid)} "
            f"(schema version {TABLE_SCHEMA_VERSION}; regenerate with: {ANALYZE_COMMAND} --force)"
        )
    return None


def _inspect(resolved: ResolvedConfig) -> _Inspection:
    """Read every artifact the dashboard needs, collecting all fatal problems."""
    study = require_study(resolved, stage="dashboard")
    result = _Inspection()

    try:
        result.runs = load_runs(study)
    except DrlSimilarityError as exc:
        result.problems.append(f"{_describe(exc)} [{TRAIN_COMMAND}; then {SELECT_COMMAND}]")
    for run in result.runs:
        path = run.run_dir / "evaluations.parquet"
        if not path.exists():
            result.problems.append(f"Missing evaluation table: {path} [{TRAIN_COMMAND}]")
            continue
        try:
            table = pd.read_parquet(path)
            require_columns(table, EVALUATION_COLUMNS, name=str(path), regenerate=TRAIN_COMMAND)
        except DrlSimilarityError as exc:
            result.problems.append(_describe(exc))
        except Exception as exc:  # noqa: BLE001 - corrupt parquet
            result.problems.append(f"Unreadable evaluation table {path}: {exc} [{TRAIN_COMMAND}]")
        else:
            result.evaluations[run.run_id] = table

    manifest_path = Path(study.scenarios.manifest)
    if not manifest_path.exists():
        result.problems.append(
            f"Missing scenario manifest: {manifest_path} [drl-sim collect-scenarios --config <study.yaml>]"
        )
    else:
        try:
            result.scenario_manifest = load_manifest(manifest_path)
        except DrlSimilarityError as exc:
            result.problems.append(f"{_describe(exc)} [{VALIDATE_SCENARIOS_COMMAND}]")
    obs_file = observations_path(study)
    if not obs_file.exists():
        result.problems.append(
            f"Missing scenario observations: {obs_file} [drl-sim collect-scenarios --config <study.yaml>]"
        )

    metrics_dir = find_metrics_dir(resolved)
    if metrics_dir is None:
        result.problems.append(
            f"No metrics artifact under {study_root(study)} [{ANALYZE_COMMAND}, or {RUN_STUDY_COMMAND}]"
        )
        return result
    result.metrics_dir = metrics_dir
    try:
        manifest = verify_artifact_dir(metrics_dir)
    except DrlSimilarityError as exc:
        result.problems.append(
            f"Metrics artifact {metrics_dir} failed verification: {_describe(exc)} "
            f"[{ANALYZE_COMMAND} --force]"
        )
        return result
    result.metrics_manifest = manifest
    metadata = manifest.metadata
    if metadata.get("schema") != METRICS_SCHEMA:
        result.problems.append(
            f"Metrics artifact {metrics_dir} has schema {metadata.get('schema')!r}; expected "
            f"{METRICS_SCHEMA!r} [{ANALYZE_COMMAND} --force]"
        )
    if metadata.get("stage_config_hash") != analysis_stage_hash(resolved):
        result.problems.append(
            f"Metrics artifact {metrics_dir} was computed with a different study configuration "
            f"[{ANALYZE_COMMAND} --force]"
        )
    for name, (filename, columns) in METRIC_TABLES.items():
        path = metrics_dir / filename
        if not path.exists():
            result.problems.append(
                f"Missing metric table: {path} (schema version {TABLE_SCHEMA_VERSION}) "
                f"[{ANALYZE_COMMAND} --force]"
            )
            continue
        try:
            table = pd.read_parquet(path)
            require_columns(table, columns, name=str(path), regenerate=f"{ANALYZE_COMMAND} --force")
        except DrlSimilarityError as exc:
            result.problems.append(_describe(exc))
            continue
        except Exception as exc:  # noqa: BLE001 - corrupt parquet
            result.problems.append(
                f"Unreadable metric table {path} (schema version {TABLE_SCHEMA_VERSION}): {exc} "
                f"[{ANALYZE_COMMAND} --force]"
            )
            continue
        result.tables[name] = table
    if "summaries" in result.tables:
        problem = _check_enum(result.tables["summaries"], "scope", SUMMARY_SCOPES, name="summaries.parquet")
        if problem:
            result.problems.append(problem)
        if result.tables["summaries"].empty:
            result.problems.append(f"summaries.parquet contains no rows [{ANALYZE_COMMAND} --force]")
    if "frame_pairs" in result.tables:
        problem = _check_enum(result.tables["frame_pairs"], "pair_kind", PAIR_KINDS, name="frame_pairs.parquet")
        if problem:
            result.problems.append(problem)
    return result


def _raise_for(problems: list[str]) -> None:
    if problems:
        lines = "\n".join(f"  - {problem}" for problem in problems)
        raise ArtifactError(
            f"Dashboard artifacts are incomplete or invalid ({len(problems)} issue(s)):\n{lines}",
            context={"regenerate": RUN_STUDY_COMMAND},
        )


def validate_dashboard_artifacts(resolved: ResolvedConfig) -> None:
    """Raise ``ArtifactError`` naming every missing or corrupt artifact and how to regenerate it."""
    _raise_for(_inspect(resolved).problems)


# --------------------------------------------------------------------------- loading


def redact_text(text: str) -> str:
    """Replace absolute user paths so provenance can be shared."""
    replacements = [
        (str(REPO_ROOT), "<repo>"),
        (REPO_ROOT.as_posix(), "<repo>"),
        (str(Path.home()), "<home>"),
        (Path.home().as_posix(), "<home>"),
    ]
    for needle, replacement in replacements:
        if needle and needle not in ("/", "\\"):
            text = text.replace(needle, replacement)
            text = text.replace(needle.replace("\\", "\\\\"), replacement)
    return text


def _display_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return redact_text(str(path))


def _runs_table(runs: list[RunInfo]) -> pd.DataFrame:
    rows = []
    for run in runs:
        selection = run.experiment.checkpoint_selection
        rows.append(
            {
                "run_id": run.run_id,
                "run_dir": _display_path(run.run_dir),
                "algorithm_id": run.algorithm.id,
                "adapter": run.algorithm.adapter,
                "policy": run.algorithm.policy,
                "environment_id": run.environment.id,
                "env_id": run.environment.env_id,
                "training_seed": int(run.experiment.seed),
                "total_timesteps": int(run.experiment.training.total_timesteps),
                "selection_seeds": ", ".join(str(s) for s in run.experiment.evaluation.selection.seeds),
                "final_seeds": ", ".join(str(s) for s in run.experiment.evaluation.final.seeds),
                "weak_rule": selection.weak.rule,
                "medium_rule": selection.medium.rule,
                "strong_rule": selection.strong.rule,
                "minimum_strong_improvement": float(selection.minimum_strong_improvement),
                "observation_fingerprint": run.observation_fingerprint or "",
                "runtime_fingerprint": run.runtime_fingerprint or "",
            }
        )
    return pd.DataFrame(rows)


def _selected_table(runs: list[RunInfo]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for capability in CAPABILITIES:
            item = run.selected.get(capability)
            if item is None:
                continue
            rows.append(
                {
                    "run_id": run.run_id,
                    "capability": capability,
                    "checkpoint_id": item.checkpoint_id,
                    "training_step": int(item.training_step),
                    "score": np.nan if item.score is None else float(item.score),
                    "uncertainty": np.nan if item.uncertainty is None else float(item.uncertainty),
                }
            )
    return pd.DataFrame(
        rows, columns=["run_id", "capability", "checkpoint_id", "training_step", "score", "uncertainty"]
    )


def _clips_table(manifest: ScenarioManifest, analysed: list[str]) -> pd.DataFrame:
    order = {clip_id: index for index, clip_id in enumerate(analysed)}
    rows = []
    for clip in manifest.clips:
        row = clip.model_dump(mode="json")
        row["clip_index"] = order.get(clip.clip_id, -1)
        row["analysed"] = clip.clip_id in order
        row["curator"] = clip.curator or ""
        rows.append(row)
    return pd.DataFrame(rows)


def _layers_table(entries: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "run_id", "capability", "layer", "source_layer", "pooling", "original_shape",
        "feature_dim", "checkpoint_id", "training_step", "checkpoint_sha256", "path",
    ]  # fmt: skip
    rows = []
    for entry in entries:
        row = {column: entry.get(column) for column in columns}
        row["original_shape"] = "×".join(str(dim) for dim in entry.get("original_shape", []))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _flatten(payload: Any, prefix: str = "") -> dict[str, str]:
    """Flatten nested version metadata into ``dotted.key -> text``."""
    if isinstance(payload, dict):
        out: dict[str, str] = {}
        for key, value in payload.items():
            out.update(_flatten(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip(".") or "value": redact_text(str(payload))}


def _ordered(found: list[str], preferred: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Preferred order first, then any remaining values in first-seen order."""
    seen = list(dict.fromkeys(found))
    head = [item for item in preferred if item in seen]
    return tuple(head + [item for item in seen if item not in head])


def _quality(
    study: StudyConfig,
    dashboard: DashboardConfig,
    inspection: _Inspection,
    summary: StudySummary,
    clips: pd.DataFrame,
    summaries: pd.DataFrame,
    activations_manifest: dict[str, Any] | None,
) -> tuple[QualityIssue, ...]:
    issues: list[QualityIssue] = []
    metadata = inspection.metrics_manifest.metadata if inspection.metrics_manifest else {}

    if summary.synthetic:
        issues.append(
            QualityIssue(
                "warning",
                "synthetic_data",
                "This study contains SYNTHETIC data produced by scripts/generate_fixture_study.py. "
                "Values demonstrate the pipeline and are not experimental results.",
            )
        )
    if summary.n_training_seeds < 2:
        issues.append(
            QualityIssue(
                "warning",
                "single_seed",
                "One training seed: results are exploratory. Intervals reflect clip resampling only, "
                "not variation between training runs.",
            )
        )
    else:
        issues.append(QualityIssue("ok", "seeds", f"{summary.n_training_seeds} training seeds."))

    issues.append(
        QualityIssue("ok", "checksums", "Metrics manifest verified: every input and output checksum matches.")
    )

    counts = dict(summary.clips_per_label)
    if len(set(counts.values())) > 1:
        issues.append(QualityIssue("warning", "class_balance", f"Clips per scenario are unbalanced: {counts}."))
    else:
        issues.append(QualityIssue("ok", "class_balance", f"Clips per scenario are balanced: {counts}."))
    minimum = study.scenarios.minimum_clips_per_label
    short = {label: n for label, n in counts.items() if n < minimum}
    if short:
        issues.append(
            QualityIssue("warning", "clip_count", f"Fewer than {minimum} clips for: {short}.")
        )

    analysed = clips[clips["analysed"]] if len(clips) else clips
    if len(analysed):
        # Splits are per clip; scenario validation already rejects shared source frames
        episodes = analysed.groupby("source_episode_id")["split"].nunique()
        shared = int((episodes > 1).sum())
        issues.append(
            QualityIssue(
                "info" if shared else "ok",
                "split_isolation",
                f"{shared} source episode(s) contribute non-overlapping clips to both train and test."
                if shared
                else "No source episode contributes to both train and test.",
            )
        )
        duplicated = int(analysed["observation_sha256"].duplicated().sum())
        if duplicated:
            issues.append(
                QualityIssue("error", "clip_overlap", f"{duplicated} analysed clip(s) share identical observations.")
            )
        else:
            issues.append(QualityIssue("ok", "clip_overlap", "No analysed clips share identical observations."))
    pending = int((clips["review_status"] != "approved").sum()) if len(clips) else 0
    if pending:
        issues.append(
            QualityIssue("info", "review", f"{pending} clip(s) are pending or rejected and were not analysed.")
        )

    collection = set(study.scenarios.collection.seeds)
    overlap: set[int] = set()
    for run in inspection.runs:
        overlap |= collection & run.evaluation_seeds
    if overlap:
        issues.append(
            QualityIssue(
                "error", "seed_separation", f"Scenario collection seeds overlap evaluation seeds: {sorted(overlap)}."
            )
        )
    else:
        issues.append(
            QualityIssue("ok", "seed_separation", "Scenario collection seeds are disjoint from evaluation seeds.")
        )

    nan_counts = {key: int(value) for key, value in (metadata.get("nan_counts") or {}).items() if value}
    if nan_counts:
        issues.append(QualityIssue("warning", "missing_values", f"Undefined (NaN) metric values: {nan_counts}."))
    else:
        issues.append(QualityIssue("ok", "missing_values", "No undefined metric values."))
    biased = {
        key: int(value) for key, value in (metadata.get("estimate_outside_interval") or {}).items() if value
    }
    if biased:
        issues.append(
            QualityIssue(
                "warning",
                "interval_bias",
                f"Point estimate lies outside its bootstrap percentile interval in {biased} row(s). "
                "Resampling clips with replacement biases some statistics (notably CKA, SVCCA and "
                "dispersion with few clips); those intervals are flagged and should not be read as "
                "coverage intervals.",
            )
        )
    flagged = int((summaries["warnings"].fillna("") != "").sum()) if len(summaries) else 0
    if flagged:
        issues.append(
            QualityIssue("info", "metric_warnings", f"{flagged} summary row(s) carry metric warnings (see the warnings column).")
        )
    for note in metadata.get("notes") or []:
        issues.append(QualityIssue("info", "analysis_note", str(note)))
    if activations_manifest is None:
        issues.append(
            QualityIssue(
                "info",
                "activations_unavailable",
                "Activation arrays are not available; latent signatures are hidden.",
            )
        )

    defaults = dashboard.default_filters
    checks = [
        ("checkpoint", defaults.checkpoint, summary.capabilities),
        ("layer", defaults.layer, summary.layers),
        ("row_scenario", defaults.row_scenario, summary.labels),
        ("column_scenario", defaults.column_scenario, summary.labels),
        ("primary_metric", defaults.primary_metric, CELL_METRICS),
        ("pair_set", defaults.pair_set, summary.pair_sets),
    ]
    for name, value, options in checks:
        if options and value not in options:
            issues.append(
                QualityIssue(
                    "warning",
                    "invalid_default",
                    f"Dashboard default {name}={value!r} is not available; using {options[0]!r}.",
                )
            )
    return tuple(issues)


def load_dashboard_data(resolved: ResolvedConfig) -> DashboardData:
    """Read and validate every artifact; raise one ``ArtifactError`` listing all failures."""
    study = require_study(resolved, stage="dashboard")
    inspection = _inspect(resolved)
    _raise_for(inspection.problems)
    assert inspection.metrics_dir is not None and inspection.metrics_manifest is not None
    assert inspection.scenario_manifest is not None
    dashboard = study.dashboard
    if not isinstance(dashboard, DashboardConfig):
        raise ArtifactError("Study dashboard configuration was not resolved")
    metrics_dir = inspection.metrics_dir
    manifest = inspection.metrics_manifest
    metadata = manifest.metadata
    summaries = inspection.tables["summaries"]

    activations_dir: Path | None = None
    activations_manifest: dict[str, Any] | None = None
    name = metadata.get("activations_dir")
    if name:
        candidate = metrics_dir.parent / str(name)
        if (candidate / "manifest.json").exists():
            try:
                activations_manifest = ArtifactManifest.read(candidate / "manifest.json").model_dump(mode="json")
                activations_dir = candidate
            except DrlSimilarityError:
                activations_manifest = None
    entries = (activations_manifest or {}).get("metadata", {}).get("entries", [])
    analysed_ids = [str(v) for v in (activations_manifest or {}).get("metadata", {}).get("clip_ids", [])]
    if not analysed_ids:
        analysed_ids = [
            clip.clip_id
            for clip in inspection.scenario_manifest.active_clips(
                require_approved=study.scenarios.review.require_approved
            )
        ]

    clips = _clips_table(inspection.scenario_manifest, analysed_ids)
    labels = tuple(study.scenarios.labels)
    capabilities = _ordered(summaries["checkpoint"].astype(str).tolist(), dashboard.theme.capability_order)
    configured_layers = [layer.name for layer in study.extraction.layers]
    layers = _ordered(summaries["layer"].astype(str).tolist(), configured_layers)
    metric_order = [spec.id for spec in study.analysis.metrics]
    metrics = _ordered(summaries["metric"].astype(str).tolist(), metric_order)
    clips_per_label = metadata.get("clips_per_label") or {}
    evaluations = pd.concat(list(inspection.evaluations.values()), ignore_index=True)
    runs = _runs_table(inspection.runs)

    summary = StudySummary(
        study_id=study.id,
        title=dashboard.title,
        config_hash=resolved.config_hash,
        config_path=_display_path(resolved.source_paths[0]) if resolved.source_paths else "",
        labels=labels,
        capabilities=capabilities,
        layers=layers,
        metrics=metrics,
        pair_sets=tuple(study.analysis.pair_sets),
        clip_steps=int(study.scenarios.clip_steps),
        burn_in_steps=int(study.scenarios.burn_in_steps),
        n_clips=int(metadata.get("n_clips", len(analysed_ids))),
        clips_per_label=tuple((label, int(clips_per_label.get(label, 0))) for label in labels),
        n_training_seeds=int(runs["training_seed"].nunique()) if len(runs) else 0,
        synthetic=bool(metadata.get("synthetic", False) or inspection.scenario_manifest.synthetic),
        metrics_dir=str(metrics_dir),
        metrics_version=metrics_dir.name,
        metrics_manifest_sha256=sha256_file(metrics_dir / "manifest.json"),
        activations_dir=None if activations_dir is None else str(activations_dir),
        observations_file=str(observations_path(study)),
        scenario_manifest=str(study.scenarios.manifest),
        generated_at=manifest.created_at,
        bootstrap=tuple(sorted((metadata.get("bootstrap") or {}).items())),
    )

    representative = pd.DataFrame(
        metadata.get("representative_frames") or [],
        columns=["scenario", "clip_id", "step", "frame_rank", "offset_rank"],
    )
    manifests: dict[str, dict[str, Any]] = {"metrics": manifest.model_dump(mode="json")}
    if activations_manifest is not None:
        manifests["activations"] = activations_manifest
    manifests["scenarios"] = inspection.scenario_manifest.model_dump(mode="json", exclude={"clips"})
    for run in inspection.runs:
        versions_path = run.run_dir / "versions.json"
        if not versions_path.exists():
            continue
        try:
            flat = _flatten(json.loads(versions_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
        commit = next((value for key, value in flat.items() if "commit" in key), None)
        manifests[f"run {run.run_id}"] = {"package_versions": flat, "git_commit": commit, "created_at": None}
    sources: list[tuple[str, str]] = []
    for path in resolved.source_paths:
        try:
            sources.append((_display_path(path), redact_text(Path(path).read_text(encoding="utf-8"))))
        except OSError:
            continue
    selection = {run.run_id: run.selection_payload for run in inspection.runs}
    quality = _quality(study, dashboard, inspection, summary, clips, summaries, activations_manifest)
    return DashboardData(
        study=summary,
        dashboard=dashboard,
        runs=runs,
        selected=_selected_table(inspection.runs),
        evaluations=evaluations,
        clips=clips,
        summaries=summaries,
        cross_checkpoint=inspection.tables["cross_checkpoint"],
        frame_pairs=inspection.tables["frame_pairs"],
        embeddings=inspection.tables["embeddings"],
        layers=_layers_table(entries),
        representative_frames=representative,
        metric_info={str(item["id"]): item for item in metadata.get("metrics") or []},
        manifests=manifests,
        selection=selection,
        resolved_yaml=redact_text(yaml.safe_dump(resolved.canonical, sort_keys=True)),
        source_configs=tuple(sources),
        quality=quality,
    )


# --------------------------------------------------------------------------- filters and joins


def select_rows(frame: pd.DataFrame, **equals: Any) -> pd.DataFrame:
    """Rows where every given column equals the value (or is in the list/tuple/set)."""
    if frame.empty:
        return frame
    mask = np.ones(len(frame), dtype=bool)
    for column, value in equals.items():
        if value is None:
            continue
        if isinstance(value, list | tuple | set | frozenset):
            mask &= frame[column].isin(list(value)).to_numpy()
        else:
            mask &= (frame[column] == value).to_numpy()
    return frame.loc[mask]


def metric_pair_set(metric: str, pair_set: str) -> str:
    """Pair set under which a cell metric is stored (clip-pair CKA has its own)."""
    return CLIP_ALIGNED_STEPS if metric == "linear_cka" else pair_set


def metric_direction(data: DashboardData, metric: str) -> str:
    info = data.metric_info.get(metric, {})
    kind = info.get("kind", "similarity")
    if kind == "distance":
        return "lower = more similar"
    if kind == "similarity":
        return "higher = more similar"
    return str(info.get("direction", "higher = stronger"))


def case_cell_rows(
    data: DashboardData,
    *,
    run_id: str,
    checkpoint: str,
    layer: str,
    pair_set: str,
    row: str,
    column: str,
    metrics: tuple[str, ...] = CELL_METRICS,
) -> dict[str, pd.Series]:
    """Summary rows (one per metric) for one case-to-case cell."""
    scope = "within_scenario" if row == column else "between_scenario"
    out: dict[str, pd.Series] = {}
    base = select_rows(
        data.summaries, run_id=run_id, checkpoint=checkpoint, layer=layer, scope=scope,
        scenario=row, scenario_b=column,
    )  # fmt: skip
    for metric in metrics:
        rows = select_rows(base, metric=metric, pair_set=metric_pair_set(metric, pair_set))
        if len(rows):
            out[metric] = rows.iloc[0]
    return out


def case_matrix_table(
    data: DashboardData, *, run_id: str, checkpoint: str, layer: str, pair_set: str
) -> pd.DataFrame:
    """Tidy table behind the case matrix (also the accessible tabular fallback)."""
    rows = select_rows(
        data.summaries, run_id=run_id, checkpoint=checkpoint, layer=layer,
        scope=["within_scenario", "between_scenario"], metric=list(CELL_METRICS),
    )  # fmt: skip
    if rows.empty:
        return rows
    keep = [metric_pair_set(m, pair_set) == p for m, p in zip(rows["metric"], rows["pair_set"], strict=True)]
    return rows.loc[keep].rename(columns={"scenario": "row_scenario", "scenario_b": "column_scenario"})


def frames_for_case(data: DashboardData, scenario: str, frames_per_case: int) -> pd.DataFrame:
    """Representative frames for a scenario: one frame per clip, spread across offsets.

    Frame *k* is taken from the *k*-th representative clip at the *k*-th
    representative offset, so the frames differ in both clip and time.
    """
    frames = data.representative_frames
    rows = frames[frames["scenario"] == scenario]
    picks = []
    for rank in range(int(frames_per_case)):
        candidates = rows[rows["frame_rank"] == rank]
        if candidates.empty:
            continue
        offsets = sorted(candidates["offset_rank"].unique())
        chosen = candidates[candidates["offset_rank"] == offsets[min(rank, len(offsets) - 1)]]
        picks.append(chosen.iloc[0])
    return pd.DataFrame(picks, columns=frames.columns).reset_index(drop=True)


def frame_pair_values(
    data: DashboardData,
    *,
    run_id: str,
    checkpoint: str,
    layer: str,
    clip_a: str,
    step_a: int,
    clip_b: str,
    step_b: int,
    metrics: tuple[str, ...] = CELL_METRICS,
) -> dict[str, float]:
    """Stored representative-frame values for one unordered frame pair."""
    pairs = select_rows(
        data.frame_pairs, run_id=run_id, checkpoint=checkpoint, layer=layer, pair_kind="representative_frame"
    )
    if pairs.empty:
        return {}
    forward = (
        (pairs["clip_id_a"] == clip_a) & (pairs["step_a"] == step_a)
        & (pairs["clip_id_b"] == clip_b) & (pairs["step_b"] == step_b)
    )  # fmt: skip
    backward = (
        (pairs["clip_id_a"] == clip_b) & (pairs["step_a"] == step_b)
        & (pairs["clip_id_b"] == clip_a) & (pairs["step_b"] == step_a)
    )  # fmt: skip
    matched = pairs[forward | backward]
    out: dict[str, float] = {}
    for metric in metrics:
        rows = matched[matched["metric"] == metric]
        if len(rows):
            out[metric] = float(rows["value"].iloc[0])
    return out


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def manifest_json(manifest: dict[str, Any]) -> str:
    return redact_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))


# --------------------------------------------------------------------------- lazy binary readers


def read_clip_observations(observations_file: str, observation_key: str) -> np.ndarray:
    """Frozen observations of one clip, including burn-in frames: ``(frames, *obs_shape)``."""
    path = Path(observations_file)
    if not path.exists():
        raise ArtifactError(f"Missing scenario observations: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if observation_key not in archive.files:
            raise ArtifactError(f"Observation key {observation_key!r} missing from {path}")
        return np.asarray(archive[observation_key])


def newest_frame(observation: np.ndarray) -> np.ndarray:
    """Newest 2-D frame of one observation (last channel of a channel-first frame stack)."""
    array = np.asarray(observation)
    if array.ndim == 3:
        if array.shape[-1] in (1, 3) and array.shape[0] not in (1, 3, 4):
            return array if array.shape[-1] == 3 else array[..., 0]
        return array[-1]
    if array.ndim == 2:
        return array
    raise ArtifactError(f"Cannot render an observation with shape {array.shape}")


def stack_strip(observation: np.ndarray, *, gap: int = 2) -> np.ndarray:
    """All frames of a channel-first stack side by side, oldest to newest."""
    array = np.asarray(observation)
    if array.ndim != 3:
        return newest_frame(array)
    pad = np.full((array.shape[1], gap), 255, dtype=array.dtype)
    parts: list[np.ndarray] = []
    for index in range(array.shape[0]):
        if index:
            parts.append(pad)
        parts.append(array[index])
    return np.concatenate(parts, axis=1)


def contact_sheet(frames: list[np.ndarray], *, columns: int = 6, gap: int = 2) -> np.ndarray:
    """Grid of equally sized 2-D frames."""
    if not frames:
        return np.zeros((1, 1), dtype=np.uint8)
    height, width = frames[0].shape[:2]
    columns = max(1, min(columns, len(frames)))
    rows = int(np.ceil(len(frames) / columns))
    sheet = np.full((rows * height + (rows - 1) * gap, columns * width + (columns - 1) * gap), 255, dtype=np.uint8)
    for index, frame in enumerate(frames):
        top = (index // columns) * (height + gap)
        left = (index % columns) * (width + gap)
        sheet[top : top + height, left : left + width] = frame
    return sheet


def clip_gif(frames: np.ndarray, *, fps: int, scale: int = 3) -> bytes:
    """Animated GIF of 2-D frames ``(n, H, W)`` for the clip player."""
    from PIL import Image

    images = [
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize(
            (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST
        )
        for frame in frames
    ]
    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=max(int(1000 / max(fps, 1)), 20),
        loop=0,
    )
    return buffer.getvalue()


def read_latent_vectors(
    activations_dir: str, relpath: str, picks: tuple[tuple[int, int], ...]
) -> np.ndarray:
    """Stored activation vectors for ``(clip_index, step)`` picks: ``(len(picks), features)``."""
    path = Path(activations_dir) / relpath
    if not path.exists():
        raise ArtifactError(f"Missing activation array: {path}", context={"regenerate": EXTRACT_COMMAND})
    with np.load(path, allow_pickle=False) as archive:
        array = archive["activations"]
        return np.stack([np.asarray(array[clip, step], dtype=np.float64) for clip, step in picks])


def latent_signature(vector: np.ndarray, *, bins: int = 48) -> np.ndarray:
    """Fixed-length display signature: block means of a stored activation vector."""
    values = np.asarray(vector, dtype=np.float64).ravel()
    if values.size <= bins:
        return values
    edges = np.linspace(0, values.size, bins + 1).astype(int)
    return np.asarray([values[start:stop].mean() for start, stop in zip(edges[:-1], edges[1:], strict=True)])


def activation_path(data: DashboardData, run_id: str, checkpoint: str, layer: str) -> str | None:
    if data.study.activations_dir is None or data.layers.empty:
        return None
    rows = select_rows(data.layers, run_id=run_id, capability=checkpoint, layer=layer)
    return None if rows.empty else str(rows["path"].iloc[0])
