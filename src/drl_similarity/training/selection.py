"""Weak/medium/strong checkpoint selection from the selection evaluation suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from drl_similarity.config import AlgorithmConfig, ExperimentConfig, ResolvedConfig
from drl_similarity.errors import CheckpointError, ConfigError
from drl_similarity.registry import Registry


RULE_VERSION = "v1"

# A rule maps (per-checkpoint table sorted by step, context) -> the chosen row
SelectionRule = Callable[[pd.DataFrame, dict[str, Any]], pd.Series]
SELECTION_RULES: Registry[SelectionRule] = Registry("selection rule")


def aggregate_selection(evaluations: pd.DataFrame, *, smoothing_window: int = 1) -> pd.DataFrame:
    """One row per checkpoint: mean/std/count of selection returns plus a smoothed mean."""
    selection = evaluations[evaluations["suite"] == "selection"]
    if selection.empty:
        raise CheckpointError("No selection-suite evaluation rows found")

    table = (
        selection.groupby(["checkpoint_id", "training_step"], as_index=False)
        .agg(mean_return=("return", "mean"), std_return=("return", "std"), n_episodes=("return", "count"))
        .sort_values("training_step")
        .reset_index(drop=True)
    )
    table["smoothed_return"] = (
        table["mean_return"].rolling(smoothing_window, min_periods=1, center=True).mean()
    )

    return table


def earliest_complete_after_learning_starts(table: pd.DataFrame, context: dict[str, Any]) -> pd.Series:
    """Weak: first fully evaluated checkpoint at or after ``learning_starts``."""
    eligible = table[
        (table["training_step"] >= context["learning_starts"])
        & (table["n_episodes"] >= context["expected_episodes"])
    ]
    if eligible.empty:
        raise CheckpointError(
            "No complete evaluation at or after learning_starts",
            context={"learning_starts": context["learning_starts"]},
        )

    return eligible.iloc[0]


def highest_mean_selection_return(table: pd.DataFrame, context: dict[str, Any]) -> pd.Series:
    """Strong: highest mean return; ties prefer the earlier checkpoint."""
    return table.loc[table["mean_return"].idxmax()]


def nearest_performance_midpoint(table: pd.DataFrame, context: dict[str, Any]) -> pd.Series:
    """Medium: smoothed return closest to the weak/strong midpoint, strictly between them."""
    weak, strong = context["weak"], context["strong"]
    between = table[
        (table["training_step"] > weak["training_step"]) & (table["training_step"] < strong["training_step"])
    ]
    if between.empty:
        raise CheckpointError(
            "No checkpoint lies between weak and strong for medium selection",
            context={"weak_step": int(weak["training_step"]), "strong_step": int(strong["training_step"])},
        )

    midpoint = 0.5 * (weak["mean_return"] + strong["mean_return"])
    distance = (between["smoothed_return"] - midpoint).abs()

    return between.loc[distance.idxmin()]


SELECTION_RULES.register("earliest_complete_after_learning_starts", earliest_complete_after_learning_starts)
SELECTION_RULES.register("highest_mean_selection_return", highest_mean_selection_return)
SELECTION_RULES.register("nearest_performance_midpoint", nearest_performance_midpoint)


def select_checkpoints(resolved: ResolvedConfig) -> Path:
    """Apply the configured rules and write ``selected-checkpoints.yaml`` with evidence."""
    experiment = resolved.value
    if not isinstance(experiment, ExperimentConfig):
        raise ConfigError("select-checkpoints requires an experiment config")
    assert isinstance(experiment.algorithm, AlgorithmConfig)

    run_dir = Path(experiment.artifacts.root) / experiment.id
    eval_path = run_dir / "evaluations.parquet"
    if not eval_path.exists():
        raise CheckpointError(
            f"Missing evaluations: {eval_path}", context={"hint": "run drl-sim train first"}
        )

    rules = experiment.checkpoint_selection
    suite = experiment.evaluation.selection
    table = aggregate_selection(pd.read_parquet(eval_path), smoothing_window=rules.medium.smoothing_window)
    context: dict[str, Any] = {
        "learning_starts": int(experiment.algorithm.model_kwargs.get("learning_starts", 0)),
        "expected_episodes": len(suite.seeds) * suite.episodes_per_seed,
    }

    weak = SELECTION_RULES.get(rules.weak.rule)(table, context)
    strong = SELECTION_RULES.get(rules.strong.rule)(table, context)
    improvement = float(strong["mean_return"] - weak["mean_return"])
    if improvement < rules.minimum_strong_improvement or strong["training_step"] <= weak["training_step"]:
        raise CheckpointError(
            "Insufficient learning: strong does not materially outperform an earlier weak checkpoint",
            context={
                "weak": f"step {int(weak['training_step'])} return {weak['mean_return']:.3f}",
                "strong": f"step {int(strong['training_step'])} return {strong['mean_return']:.3f}",
                "minimum_strong_improvement": rules.minimum_strong_improvement,
            },
        )
    medium = SELECTION_RULES.get(rules.medium.rule)(table, {**context, "weak": weak, "strong": strong})

    payload = {
        "schema_version": 1,
        "rule_version": RULE_VERSION,
        "config_hash": resolved.config_hash,
        "selected": {"weak": _entry(weak), "medium": _entry(medium), "strong": _entry(strong)},
        "evidence": {
            "rules": {name: getattr(rules, name).rule for name in ("weak", "medium", "strong")},
            "n_checkpoints": len(table),
            "improvement": improvement,
            "minimum_strong_improvement": rules.minimum_strong_improvement,
            "target_midpoint": float(0.5 * (weak["mean_return"] + strong["mean_return"])),
            "smoothing_window": rules.medium.smoothing_window,
        },
    }
    out = run_dir / "selected-checkpoints.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    return out


def _entry(row: pd.Series) -> dict[str, Any]:
    """Selection record: step, score, and uncertainty (std of episode returns)."""
    return {
        "checkpoint_id": str(row["checkpoint_id"]),
        "training_step": int(row["training_step"]),
        "score": float(row["mean_return"]),
        "uncertainty": None if pd.isna(row["std_return"]) else float(row["std_return"]),
        "smoothed_score": float(row["smoothed_return"]),
        "n_episodes": int(row["n_episodes"]),
    }
