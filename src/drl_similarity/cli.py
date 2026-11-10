"""Typer CLI: thin commands that load a YAML config and call one application service.

Flags only change execution (``--resume``, ``--force``, ``--dry-run``, ``--log-level``);
every scientific choice lives in YAML. Backends are imported inside commands.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer

from drl_similarity.config import DocumentKind, ResolvedConfig, load_config
from drl_similarity.errors import DrlSimilarityError
from drl_similarity.logging import configure_logging


app = typer.Typer(
    name="drl-sim",
    help="YAML-driven DRL representation similarity study framework.",
    no_args_is_help=True,
    add_completion=False,
)

CONFIG = typer.Option(..., "--config", exists=True, dir_okay=False, readable=True, help="YAML config")
DRY_RUN = typer.Option(False, "--dry-run", help="Validate and print the plan; write nothing")
FORCE = typer.Option(False, "--force", help="Publish a new artifact version (never overwrites)")


@contextmanager
def _domain_errors() -> Iterator[None]:
    """Turn expected domain errors into a concise message and exit code 1."""
    try:
        yield
    except DrlSimilarityError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _load(config: Path, kind: DocumentKind | None, *, allow_missing_manifest: bool = False) -> ResolvedConfig:
    """Resolve a config and print its identity."""
    resolved = load_config(config, kind, allow_missing_scenario_manifest=allow_missing_manifest)
    typer.echo(f"kind={resolved.value.kind} id={resolved.value.id}")
    typer.echo(f"config_hash={resolved.config_hash}")
    for path in resolved.source_paths:
        typer.echo(f"source={path.as_posix()}")

    return resolved


def _ok(message: str) -> None:
    typer.secho(f"ok: {message}", fg=typer.colors.GREEN)


def _planned(dry_run: bool, output: Path | str) -> bool:
    """Print the planned output for a dry run; return True when nothing should be written."""
    if dry_run:
        typer.echo(f"dry-run: would write {Path(output).as_posix()}; nothing written")

    return dry_run


def _run_dir(resolved: ResolvedConfig) -> Path:
    return Path(resolved.value.artifacts.root) / resolved.value.id  # type: ignore[union-attr]


@app.callback()
def main(log_level: str = typer.Option("INFO", "--log-level", help="Logging level")) -> None:
    configure_logging(log_level)


@app.command("validate-config")
def validate_config(config: Path = CONFIG) -> None:
    """Resolve and validate any YAML configuration document."""
    with _domain_errors():
        _load(config, None, allow_missing_manifest=True)
        _ok("configuration is valid")


@app.command("doctor")
def doctor(config: Path = CONFIG) -> None:
    """Check packages, registries, CUDA, and (for environment/experiment configs) the environment."""
    from drl_similarity.doctor import run_doctor

    with _domain_errors():
        checks = run_doctor(_load(config, None, allow_missing_manifest=True))
        for check in checks:
            color = typer.colors.GREEN if check.ok else typer.colors.RED
            typer.secho(f"[{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}", fg=color)

        passed = sum(check.ok for check in checks)
        typer.echo(f"doctor: {passed}/{len(checks)} checks passed")
        if passed < len(checks):
            raise typer.Exit(code=1)


@app.command("train")
def train(
    config: Path = CONFIG,
    resume: Path | None = typer.Option(None, "--resume", exists=True, dir_okay=False, help="Checkpoint .zip"),
    dry_run: bool = DRY_RUN,
) -> None:
    """Train an agent from an experiment YAML, checkpointing and evaluating as it goes."""
    with _domain_errors():
        resolved = _load(config, "experiment")
        if _planned(dry_run, _run_dir(resolved)):
            return
        from drl_similarity.training.runner import run_training

        _ok(f"run directory {run_training(resolved, resume=resume).as_posix()}")


@app.command("select-checkpoints")
def select_checkpoints(config: Path = CONFIG, dry_run: bool = DRY_RUN) -> None:
    """Select weak/medium/strong checkpoints from the selection suite."""
    with _domain_errors():
        resolved = _load(config, "experiment")
        if _planned(dry_run, _run_dir(resolved) / "selected-checkpoints.yaml"):
            return
        from drl_similarity.training.selection import select_checkpoints as select

        _ok(f"selection written to {select(resolved).as_posix()}")


@app.command("evaluate")
def evaluate(config: Path = CONFIG, dry_run: bool = DRY_RUN) -> None:
    """Measure the selected checkpoints on the disjoint final evaluation suite."""
    with _domain_errors():
        resolved = _load(config, "experiment")
        if _planned(dry_run, _run_dir(resolved) / "evaluations.parquet"):
            return
        from drl_similarity.training.evaluate import run_final_evaluation

        _ok(f"final evaluations written to {run_final_evaluation(resolved).as_posix()}")


@app.command("collect-scenarios")
def collect_scenarios(config: Path = CONFIG, dry_run: bool = DRY_RUN) -> None:
    """Collect candidate clips from the collector checkpoint and write a manifest template."""
    with _domain_errors():
        resolved = _load(config, "study", allow_missing_manifest=True)
        if _planned(dry_run, resolved.value.scenarios.manifest):  # type: ignore[union-attr]
            return
        from drl_similarity.scenarios.collect import collect_scenarios as collect

        _ok(f"candidates at {collect(resolved).as_posix()}; review the manifest before validating")


@app.command("validate-scenarios")
def validate_scenarios(config: Path = CONFIG) -> None:
    """Validate the curated scenario manifest and its frozen observations."""
    from drl_similarity.scenarios.validate import validate_scenarios as validate

    with _domain_errors():
        validate(_load(config, "study"))
        _ok("scenarios valid")


@app.command("extract")
def extract(config: Path = CONFIG, force: bool = FORCE, dry_run: bool = DRY_RUN) -> None:
    """Extract activations for the selected checkpoints on the curated clips."""
    from drl_similarity.study_io import ACTIVATIONS_ARTIFACT_ID, study_root

    with _domain_errors():
        resolved = _load(config, "study")
        if _planned(dry_run, study_root(resolved.value) / ACTIVATIONS_ARTIFACT_ID):  # type: ignore[arg-type]
            return
        from drl_similarity.activations.extract import extract_activations

        _ok(f"activations at {extract_activations(resolved, force=force).as_posix()}")


@app.command("analyze")
def analyze(config: Path = CONFIG, force: bool = FORCE, dry_run: bool = DRY_RUN) -> None:
    """Compute metrics, bootstrap intervals, and embeddings from activations."""
    from drl_similarity.study_io import METRICS_ARTIFACT_ID, study_root

    with _domain_errors():
        resolved = _load(config, "study")
        if _planned(dry_run, study_root(resolved.value) / METRICS_ARTIFACT_ID):  # type: ignore[arg-type]
            return
        from drl_similarity.analysis import run_analysis

        _ok(f"metrics at {run_analysis(resolved, force=force).as_posix()}")


@app.command("run-study")
def run_study(config: Path = CONFIG, dry_run: bool = DRY_RUN) -> None:
    """Idempotently verify runs and scenarios, then extract and analyze. Never trains."""
    from drl_similarity.pipeline import run_study_pipeline

    with _domain_errors():
        summary = run_study_pipeline(_load(config, "study"), dry_run=dry_run)
        for line in summary.lines:
            typer.echo(line)
        if summary.blocked:
            raise typer.Exit(code=1)
        _ok(f"study ready. Next: drl-sim dashboard --config {config.as_posix()}")


@app.command("dashboard")
def dashboard(config: Path = CONFIG) -> None:
    """Launch the read-only Streamlit dashboard for a completed study."""
    from drl_similarity.dashboard.data import validate_dashboard_artifacts

    with _domain_errors():
        validate_dashboard_artifacts(_load(config, "study"))

    # The Streamlit process finds the study through the environment
    os.environ["DRL_SIM_STUDY_CONFIG"] = str(config.resolve())
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    # Headless skips Streamlit's first-run prompt; open the printed URL in a browser
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"]
    raise typer.Exit(subprocess.call(command))
