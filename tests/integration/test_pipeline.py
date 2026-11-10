"""End-to-end pipeline on the fake pixel environment, identical for DQN and PPO (no Atari)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from drl_similarity.activations.store import load_activation_bundle
from drl_similarity.config import REPO_ROOT, AlgorithmConfig, load_config
from drl_similarity.dashboard.data import load_dashboard_data
from drl_similarity.errors import CheckpointError, ConfigError
from drl_similarity.pipeline import run_study_pipeline
from drl_similarity.scenarios.collect import collect_scenarios
from drl_similarity.training.evaluate import run_final_evaluation
from drl_similarity.training.runner import run_training
from drl_similarity.training.selection import select_checkpoints
from tests.conftest import algorithm_document, environment_document, write_yaml


def _write_configs(root: Path, algorithm: AlgorithmConfig) -> tuple[Path, Path]:
    """Experiment and study YAML; only the algorithm reference differs between adapters."""
    write_yaml(root / "env.yaml", environment_document())
    write_yaml(root / "algo.yaml", algorithm_document(algorithm.adapter))
    write_yaml(root / "dashboard.yaml", {"schema_version": 1, "kind": "dashboard", "id": "d", "title": "Tiny"})
    experiment = write_yaml(
        root / "experiment.yaml",
        {
            "schema_version": 1,
            "kind": "experiment",
            "id": "tiny-run",
            "environment": {"ref": "env.yaml"},
            "algorithm": {"ref": "algo.yaml"},
            "seed": 1,
            "device": "cpu",
            "training": {"total_timesteps": 120, "progress_bar": False},
            "evaluation": {
                "max_agent_steps": 50,
                "selection": {"seeds": [1001, 1002], "interval_steps": 40},
                "final": {"seeds": [2001]},
            },
            "checkpoint_selection": {
                "weak": {"rule": "earliest_complete_after_learning_starts"},
                "medium": {"rule": "nearest_performance_midpoint"},
                "strong": {"rule": "highest_mean_selection_return"},
                "minimum_strong_improvement": 1.0,
            },
            "artifacts": {"root": "runs"},
        },
    )
    study = write_yaml(
        root / "study.yaml",
        {
            "schema_version": 1,
            "kind": "study",
            "id": "tiny-study",
            "experiments": [{"run_dir": "runs/tiny-run"}],
            "scenarios": {
                "manifest": "scenarios/tiny.yaml",
                "labels": ["baseline", "danger", "goal"],
                "minimum_clips_per_label": 2,
                "clip_steps": 4,
                "burn_in_steps": 1,
                "collection": {
                    "seeds": [3001, 3002],
                    "episodes_per_seed": 1,
                    "max_agent_steps": 50,
                    "write_video_previews": False,
                    "candidate_windows": {
                        "baseline": {"anchor": "life_start"},
                        "danger": {"anchor": "life_lost", "offset_steps": -3},
                        "goal": {"anchor": "positive_goal_event", "offset_steps": -3},
                    },
                },
                "split": {"train_fraction": 0.5},
            },
            "extraction": {
                "device": "cpu",
                "layers": [
                    {"name": "conv3", "source": "conv3", "pooling": "global_average"},
                    {"name": "policy_features", "source": "policy_features"},
                    {
                        "name": "action_output",
                        "sources_by_adapter": {"sb3.dqn": "q_values", "sb3.ppo": "action_logits"},
                    },
                ],
            },
            "analysis": {
                "pair_sets": ["cross_clip_all", "cross_clip_aligned_step", "clip_centroids"],
                "metrics": [
                    {"id": "cosine"},
                    {"id": "correlation"},
                    {"id": "normalized_euclidean"},
                    {"id": "temporal_cosine", "kwargs": {"lags": [1]}},
                    {"id": "within_between_gap"},
                    {"id": "linear_cka"},
                ],
                "bootstrap": {"replicates": 10},
                "embeddings": [{"id": "pca", "kwargs": {"components": 2}}],
            },
            "dashboard": {"ref": "dashboard.yaml"},
            "artifacts": {"root": "artifacts"},
        },
    )

    return experiment, study


def _approve(study: Path) -> None:
    """Run the demo approval script in place of human review."""
    spec = importlib.util.spec_from_file_location("approve", REPO_ROOT / "scripts" / "approve_demo_clips.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = ["approve", "--config", str(study)]
    module.main()


@pytest.mark.slow
def test_pipeline_runs_end_to_end(tmp_path: Path, algorithm_config: AlgorithmConfig) -> None:
    experiment_path, study_path = _write_configs(tmp_path, algorithm_config)
    experiment = load_config(experiment_path, "experiment")
    run_dir = tmp_path / "runs" / "tiny-run"
    eval_path = run_dir / "evaluations.parquet"

    # Train: a checkpoint and a complete selection evaluation every 40 steps
    assert run_training(experiment) == run_dir
    evaluations = pd.read_parquet(eval_path)
    # (on-policy rollouts may overshoot the budget; the final step is always checkpointed)
    assert sorted(evaluations["training_step"].unique())[:3] == [40, 80, 120]
    assert (evaluations["length"] == 40).all() and (evaluations["return"] == 8).all()
    with pytest.raises(ConfigError, match="already exists"):
        run_training(experiment)

    # Resume from the middle checkpoint keeps earlier rows and rewrites later ones
    run_training(experiment, resume=run_dir / "checkpoints" / "checkpoint_80.zip")
    assert len(pd.read_parquet(eval_path)) == len(evaluations)

    # The fake task has a constant return, so selection must refuse it ...
    with pytest.raises(CheckpointError, match="Insufficient learning"):
        select_checkpoints(experiment)
    # ... and a rising curve is planted to exercise the remaining stages
    evaluations["return"] = evaluations["training_step"] / 10.0
    evaluations.to_parquet(eval_path, index=False)
    select_checkpoints(experiment)
    run_final_evaluation(experiment)
    final = pd.read_parquet(eval_path).query("suite == 'final'")
    assert final["checkpoint_id"].nunique() == 3

    # Scenarios: collection never approves; an unreviewed manifest blocks the study
    collect_scenarios(load_config(study_path, "study", allow_missing_scenario_manifest=True))
    study = load_config(study_path, "study")
    assert run_study_pipeline(study).blocked
    _approve(study_path)

    # Study: extract + analyze once, then everything is skipped
    summary = run_study_pipeline(study)
    assert not summary.blocked and sum("[executed]" in line for line in summary.lines) == 2
    assert all("[skipped]" in line for line in run_study_pipeline(study).lines)

    bundle = load_activation_bundle(tmp_path / "artifacts" / "tiny-study" / "activations")
    assert {entry.layer for entry in bundle.entries} == {"conv3", "policy_features", "action_output"}
    assert bundle.load(bundle.find("tiny-run", "strong", "policy_features")).shape == (6, 4, 512)

    data = load_dashboard_data(study)
    assert data.status != "error"
    assert set(data.summaries["checkpoint"]) == {"weak", "medium", "strong"}
    assert not data.cross_checkpoint.empty
