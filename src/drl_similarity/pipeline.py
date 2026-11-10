"""Idempotent study orchestration: runs -> scenarios -> extract -> analyze.

A stage is complete only if its manifest exists and verifies. Training is never
started implicitly; a missing run blocks the pipeline with the command to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from drl_similarity.artifacts.manifest import sha256_file, verify_artifact_dir
from drl_similarity.config import ResolvedConfig
from drl_similarity.errors import DrlSimilarityError
from drl_similarity.study_io import (
    ACTIVATIONS_ARTIFACT_ID,
    METRICS_ARTIFACT_ID,
    analysis_stage_hash,
    extraction_stage_hash,
    latest_artifact_dir,
    load_run,
    require_study,
    study_root,
)


@dataclass
class PipelineSummary:
    """Printable stage outcomes: skipped | planned | executed | blocked."""

    lines: list[str] = field(default_factory=list)
    blocked: bool = False

    def add(self, name: str, status: str, detail: str) -> None:
        self.lines.append(f"[{status}] {name}: {detail}")
        self.blocked = self.blocked or status == "blocked"


def run_study_pipeline(resolved: ResolvedConfig, *, dry_run: bool = False) -> PipelineSummary:
    """Verify runs and scenarios, then extract and analyze whatever is missing or stale."""
    # Stage modules load lazily so a dry run stays light
    from drl_similarity.activations.extract import (
        extract_activations,
        is_current_activation_artifact,
    )
    from drl_similarity.analysis import is_current_metrics_artifact, run_analysis
    from drl_similarity.scenarios.validate import collect_scenario_issues

    study = require_study(resolved, stage="run-study")
    root = study_root(study)
    summary = PipelineSummary()

    # Stage: completed experiment runs
    for index, ref in enumerate(study.experiments):
        try:
            run = load_run(Path(ref.run_dir), index)
            verify_artifact_dir(run.run_dir)
            summary.add(f"experiment[{index}]", "skipped", f"verified run {run.run_dir}")
        except DrlSimilarityError as exc:
            hint = "drl-sim train, then select-checkpoints and evaluate, --config <experiment.yaml>"
            summary.add(f"experiment[{index}]", "blocked", f"{exc}. Run: {hint}")
            return summary

    # Stage: curated scenarios
    issues = collect_scenario_issues(resolved)
    if issues:
        summary.add("scenarios", "blocked", "; ".join(issues))
        return summary
    summary.add("scenarios", "skipped", f"valid {study.scenarios.manifest}")

    # Stage: activations
    activations = latest_artifact_dir(root, ACTIVATIONS_ARTIFACT_ID)
    current = activations is not None and is_current_activation_artifact(
        activations, extraction_stage_hash(resolved)
    )
    if current:
        summary.add("extract", "skipped", str(activations))
    elif dry_run:
        summary.add("extract", "planned", str(root / ACTIVATIONS_ARTIFACT_ID))
    else:
        # A stale artifact is superseded by a new version, never overwritten
        activations = extract_activations(resolved, force=activations is not None)
        summary.add("extract", "executed", str(activations))

    # Stage: metrics (stale whenever activations were just re-extracted)
    metrics = latest_artifact_dir(root, METRICS_ARTIFACT_ID)
    current = (
        current
        and metrics is not None
        and is_current_metrics_artifact(
            metrics, analysis_stage_hash(resolved), sha256_file(activations / "manifest.json")
        )
    )
    if current:
        summary.add("analyze", "skipped", str(metrics))
    elif dry_run:
        summary.add("analyze", "planned", str(root / METRICS_ARTIFACT_ID))
    else:
        metrics = run_analysis(resolved, force=metrics is not None)
        summary.add("analyze", "executed", str(metrics))

    return summary
