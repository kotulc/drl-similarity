"""Column contracts for the tidy tables shared by analysis and the dashboard."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from drl_similarity.errors import ArtifactError


TABLE_SCHEMA_VERSION = 1
METRICS_SCHEMA = "metrics/v1"
ANALYZE_COMMAND = "drl-sim analyze --config <study.yaml>"
#: Diagnostic cross-checkpoint metric that is basis-dependent (never a primary result).
RAW_COSINE_DIAGNOSTIC = "raw_coordinate_cosine"

#: Value used in string key columns when a dimension does not apply.
NOT_APPLICABLE = "none"
ALL_SCENARIOS = "all"

SUMMARY_SCOPES: tuple[str, ...] = (
    "within_scenario",
    "between_scenario",
    "temporal",
    "separation",
    "confusion",
)

PAIR_KINDS: tuple[str, ...] = ("clip_pair", "representative_frame", "temporal")

#: Pair set recorded for clip-pair linear CKA (samples are aligned relative steps).
CLIP_ALIGNED_STEPS = "clip_aligned_steps"

SUMMARY_COLUMNS: tuple[str, ...] = (
    "study_id",
    "config_hash",
    "run_id",
    "algorithm_id",
    "training_seed",
    "checkpoint",
    "checkpoint_id",
    "training_step",
    "layer",
    "metric",
    "metric_kind",
    "preprocessing",
    "scope",
    "pair_set",
    "scenario",
    "scenario_b",
    "lag",
    "estimate",
    "ci_low",
    "ci_high",
    "n_clips",
    "n_pairs",
    "n_nan",
    "input_control",
    "bootstrap_replicates",
    "bootstrap_successful",
    "bootstrap_seed",
    "confidence_level",
    "warnings",
)

CROSS_CHECKPOINT_COLUMNS: tuple[str, ...] = (
    "study_id",
    "config_hash",
    "run_id_a",
    "run_id_b",
    "algorithm_id_a",
    "algorithm_id_b",
    "checkpoint_a",
    "checkpoint_b",
    "checkpoint_id_a",
    "checkpoint_id_b",
    "training_step_a",
    "training_step_b",
    "layer_a",
    "layer_b",
    "metric",
    "metric_kind",
    "preprocessing",
    "sample_unit",
    "estimate",
    "ci_low",
    "ci_high",
    "n_clips",
    "n_samples",
    "n_features_a",
    "n_features_b",
    "sample_ids_sha256",
    "applicable",
    "reason",
    "diagnostic",
    "bootstrap_replicates",
    "bootstrap_successful",
    "bootstrap_seed",
    "confidence_level",
    "warnings",
)

FRAME_PAIR_COLUMNS: tuple[str, ...] = (
    "study_id",
    "run_id",
    "checkpoint",
    "layer",
    "metric",
    "pair_kind",
    "pair_set",
    "scenario_a",
    "scenario_b",
    "clip_id_a",
    "clip_id_b",
    "step_a",
    "step_b",
    "lag",
    "pair_id",
    "same_clip",
    "value",
    "input_change",
    "n_frame_pairs",
    "n_nan",
)

EMBEDDING_COLUMNS: tuple[str, ...] = (
    "study_id",
    "run_id",
    "checkpoint",
    "layer",
    "embedding",
    "level",
    "clip_id",
    "scenario",
    "split",
    "step",
    "x",
    "y",
    "explained_variance_x",
    "explained_variance_y",
    "n_components",
    "parameters",
)

EVALUATION_COLUMNS: tuple[str, ...] = (
    "run_id",
    "training_seed",
    "checkpoint_id",
    "training_step",
    "suite",
    "eval_seed",
    "episode_index",
    "return",
    "length",
)

CLIP_COLUMNS: tuple[str, ...] = (
    "clip_index",
    "clip_id",
    "scenario",
    "split",
    "source_episode_id",
    "collection_seed",
    "observation_sha256",
)


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    name: str,
    regenerate: str | None = None,
) -> None:
    """Raise an actionable error when a table is missing required columns."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        context: dict[str, object] = {
            "missing_columns": missing,
            "expected_schema_version": TABLE_SCHEMA_VERSION,
        }
        if regenerate:
            context["regenerate"] = regenerate
        raise ArtifactError(f"Table {name} does not match the expected schema", context=context)


def empty_table(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def ordered_table(frame: pd.DataFrame, columns: Sequence[str], sort_by: Sequence[str]) -> pd.DataFrame:
    """Select the contract columns and sort rows by stable identifier columns."""
    if frame.empty:
        return empty_table(columns)
    require_columns(frame, columns, name="output")
    keys = [column for column in sort_by if column in frame.columns]
    out = frame.loc[:, list(columns)]
    if keys:
        out = out.sort_values(keys, kind="mergesort")
    return out.reset_index(drop=True)
