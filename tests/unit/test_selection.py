"""Weak/medium/strong selection rules on synthetic evaluation curves."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from drl_similarity.config import load_config
from drl_similarity.errors import CheckpointError
from drl_similarity.training.selection import aggregate_selection, select_checkpoints
from tests.conftest import algorithm_document, environment_document, write_yaml


def _experiment(tmp_path: Path, returns: list[float], *, improvement: float = 1.0):
    """Resolved experiment whose run directory holds one selection episode per checkpoint."""
    write_yaml(tmp_path / "env.yaml", environment_document())
    write_yaml(tmp_path / "algo.yaml", algorithm_document("sb3.dqn"))
    config = write_yaml(
        tmp_path / "experiment.yaml",
        {
            "schema_version": 1,
            "kind": "experiment",
            "id": "run",
            "environment": {"ref": "env.yaml"},
            "algorithm": {"ref": "algo.yaml"},
            "seed": 1,
            "training": {"total_timesteps": 100 * len(returns)},
            "evaluation": {"selection": {"seeds": [1]}, "final": {"seeds": [2]}},
            "checkpoint_selection": {
                "weak": {"rule": "earliest_complete_after_learning_starts"},
                "medium": {"rule": "nearest_performance_midpoint"},
                "strong": {"rule": "highest_mean_selection_return"},
                "minimum_strong_improvement": improvement,
            },
            "artifacts": {"root": "runs"},
        },
    )
    rows = [
        {"checkpoint_id": f"checkpoint_{100 * (i + 1)}", "training_step": 100 * (i + 1), "suite": "selection", "return": value}
        for i, value in enumerate(returns)
    ]
    (tmp_path / "runs" / "run").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(tmp_path / "runs" / "run" / "evaluations.parquet")

    return load_config(config, "experiment")


def _selected_steps(tmp_path: Path, returns: list[float]) -> dict[str, int]:
    out = select_checkpoints(_experiment(tmp_path, returns))
    selected = yaml.safe_load(out.read_text(encoding="utf-8"))["selected"]

    return {name: item["training_step"] for name, item in selected.items()}


@pytest.mark.parametrize(
    ("returns", "expected"),
    [
        ([0, 2, 5, 8, 10], {"weak": 100, "medium": 300, "strong": 500}),  # improving
        ([0, 5, 10, 10, 10], {"weak": 100, "medium": 200, "strong": 300}),  # tie prefers earlier
        ([0, 9, 1, 6, 10, 3], {"weak": 100, "medium": 400, "strong": 500}),  # noisy
    ],
)
def test_select_checkpoints_orders_capabilities(tmp_path: Path, returns: list[float], expected: dict[str, int]) -> None:
    assert _selected_steps(tmp_path, returns) == expected


@pytest.mark.parametrize(
    "returns",
    [
        [5, 5, 5, 5],  # flat
        [5, 4, 3, 2],  # best is the first checkpoint
        [0, 0.5, 0.5, 0.5],  # improvement below the configured minimum
        [0, 10],  # nothing between weak and strong
    ],
)
def test_select_checkpoints_reports_insufficient_learning(tmp_path: Path, returns: list[float]) -> None:
    with pytest.raises(CheckpointError):
        select_checkpoints(_experiment(tmp_path, returns))


def test_aggregate_selection_ignores_final_suite() -> None:
    evaluations = pd.DataFrame(
        {
            "checkpoint_id": ["a", "a", "a"],
            "training_step": [1, 1, 1],
            "suite": ["selection", "selection", "final"],
            "return": [1.0, 3.0, 100.0],
        }
    )

    table = aggregate_selection(evaluations)

    assert table["mean_return"].tolist() == [2.0]
    assert table["n_episodes"].tolist() == [2]
