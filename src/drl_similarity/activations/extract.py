"""Extract aligned activations from the selected checkpoints on curated clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drl_similarity.activations.pooling import apply_pooling
from drl_similarity.activations.store import (
    ActivationEntry,
    activation_relpath,
    publish_activations,
)
from drl_similarity.agents.registry import get_algorithm_adapter
from drl_similarity.artifacts.manifest import sha256_file, verify_artifact_dir
from drl_similarity.config import ResolvedConfig, StudyConfig
from drl_similarity.errors import ArtifactError, CheckpointError, ExtractionError
from drl_similarity.logging import get_logger, log_extra
from drl_similarity.scenarios.schema import ClipRecord, load_clip_arrays
from drl_similarity.scenarios.validate import load_validated_clips
from drl_similarity.study_io import (
    ACTIVATIONS_ARTIFACT_ID,
    CAPABILITIES,
    RunInfo,
    extraction_stage_hash,
    latest_artifact_dir,
    load_runs,
    observations_path,
    require_study,
    resolve_device,
    study_root,
)


_PROBE_SIZE = 8


@dataclass(frozen=True)
class _LayerSpec:
    name: str
    source: str
    pooling: str


def clips_table(clips: list[ClipRecord]) -> pd.DataFrame:
    """Clip order shared by every activation array."""
    return pd.DataFrame(
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


def is_current_activation_artifact(directory: Path, stage_hash: str) -> bool:
    """True when the artifact verifies and was produced by the same extraction config."""
    try:
        manifest = verify_artifact_dir(directory)
    except ArtifactError:
        return False
    return manifest.metadata.get("stage_config_hash") == stage_hash


def _resolve_layers(study: StudyConfig, run: RunInfo, available: tuple[str, ...]) -> list[_LayerSpec]:
    adapter_id = run.algorithm.adapter
    specs: list[_LayerSpec] = []
    seen: set[str] = set()
    for layer in study.extraction.layers:
        if layer.name in seen:
            raise ExtractionError(f"Duplicate extraction layer name {layer.name!r}")
        seen.add(layer.name)
        source = (layer.sources_by_adapter or {}).get(adapter_id, layer.source)
        if source is None or source not in available:
            if not layer.required:
                continue
            raise ExtractionError(
                f"Extraction layer {layer.name!r} is not available for adapter {adapter_id!r}",
                context={
                    "requested_source": source,
                    "valid_layers": list(available),
                    "run_id": run.run_id,
                    "hint": "set required: false or add a sources_by_adapter entry",
                },
            )
        specs.append(_LayerSpec(name=layer.name, source=source, pooling=layer.pooling))
    if not specs:
        raise ExtractionError(
            f"No extraction layers resolved for run {run.run_id}",
            context={"valid_layers": list(available)},
        )
    return specs


def _extract_checkpoint(
    agent: Any,
    observations: np.ndarray,
    episode_starts: np.ndarray,
    specs: list[_LayerSpec],
    *,
    batch_size: int,
    context: str,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
    """Run batched extraction and return pooled arrays plus pre-pooling shapes."""
    n_samples = observations.shape[0]
    sources = tuple(dict.fromkeys(spec.source for spec in specs))
    probe = observations[: min(_PROBE_SIZE, n_samples)]
    before = np.asarray(agent.predict(probe, deterministic=True))

    chunks: dict[str, list[np.ndarray]] = {spec.name: [] for spec in specs}
    original_shapes: dict[str, tuple[int, ...]] = {}
    for start in range(0, n_samples, batch_size):
        batch = observations[start : start + batch_size]
        outputs = agent.extract(
            batch,
            layers=sources,
            episode_starts=episode_starts[start : start + batch_size],
        )
        for spec in specs:
            raw = np.asarray(outputs[spec.source])
            if raw.shape[0] != batch.shape[0]:
                raise ExtractionError(
                    f"Layer {spec.source!r} returned {raw.shape[0]} samples for a batch of "
                    f"{batch.shape[0]} [{context}]"
                )
            original_shapes[spec.name] = tuple(int(dim) for dim in raw.shape[1:])
            pooled = apply_pooling(raw, spec.pooling, context=f"{context} layer={spec.name}")
            chunks[spec.name].append(np.asarray(pooled, dtype=np.float32))

    after = np.asarray(agent.predict(probe, deterministic=True))
    if not np.array_equal(before, after):
        raise ExtractionError(
            f"Predictions changed after extraction; activation hooks are not passive [{context}]"
        )
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ExtractionError(
                f"Non-finite activations in layer {name!r} [{context}]",
                context={"non_finite": int((~np.isfinite(array)).sum())},
            )
    return arrays, original_shapes


def extract_activations(resolved: ResolvedConfig, force: bool = False) -> Path:
    """Extract every configured layer for weak/medium/strong on the curated clips.

    An existing artifact is reused only when it verifies and matches the
    extraction configuration. ``force`` publishes a new version; nothing is
    overwritten in place.
    """
    study = require_study(resolved, stage="extract")
    logger = get_logger("extract")
    extra = log_extra(stage="extract", config_hash=resolved.config_hash)
    root = study_root(study)
    stage_hash = extraction_stage_hash(resolved)

    existing = latest_artifact_dir(root, ACTIVATIONS_ARTIFACT_ID)
    if existing is not None and not force:
        if is_current_activation_artifact(existing, stage_hash):
            logger.info("activations up to date at %s", existing, extra=extra)
            return existing
        if study.artifacts.collision == "error":
            raise ArtifactError(
                f"Existing activation artifact is stale or invalid: {existing}",
                context={"hint": "re-run with --force to publish a new version"},
            )

    manifest, clips = load_validated_clips(resolved)
    runs = load_runs(study)
    cfg = study.scenarios
    obs_file = observations_path(study)
    observations = load_clip_arrays(obs_file, clips, clip_steps=cfg.clip_steps)
    n_clips, n_steps = observations.shape[:2]
    flat = observations.reshape(n_clips * n_steps, *observations.shape[2:])
    episode_starts = np.zeros(n_clips * n_steps, dtype=bool)
    episode_starts[::n_steps] = True
    device = resolve_device(study.extraction.device)

    entries: list[ActivationEntry] = []
    arrays: dict[str, np.ndarray] = {}
    input_files: list[tuple[Path, str]] = [
        (Path(cfg.manifest), "scenario_manifest"),
        (obs_file, "observations"),
    ]
    for run in runs:
        adapter = get_algorithm_adapter(run.algorithm.adapter)
        if adapter.capabilities(run.algorithm).recurrent:
            raise ExtractionError(
                "Recurrent extraction (ordered clips with burn-in and reset masks) is not "
                "implemented yet",
                context={"run_id": run.run_id, "adapter": run.algorithm.adapter},
            )
        input_files.append((run.run_dir / "resolved-config.yaml", "run_config"))
        input_files.append((run.run_dir / "selected-checkpoints.yaml", "checkpoint_selection"))
        for capability in CAPABILITIES:
            checkpoint = run.selected[capability]
            if not checkpoint.path.exists():
                raise CheckpointError(
                    f"Selected {capability} checkpoint file is missing: {checkpoint.path}",
                    context={"run_id": run.run_id},
                )
            context = f"run={run.run_id} checkpoint={capability}"
            logger.info("extracting %s on %d samples", context, flat.shape[0], extra=extra)
            agent = adapter.load(checkpoint.path, run.algorithm, None, device=device)
            specs = _resolve_layers(study, run, tuple(agent.layer_names()))
            layer_arrays, original_shapes = _extract_checkpoint(
                agent,
                flat,
                episode_starts,
                specs,
                batch_size=study.extraction.batch_size,
                context=context,
            )
            checkpoint_sha = sha256_file(checkpoint.path)
            input_files.append((checkpoint.path, f"checkpoint:{run.run_id}:{capability}"))
            for spec in specs:
                array = layer_arrays[spec.name].reshape(n_clips, n_steps, -1)
                path = activation_relpath(run.run_id, capability, spec.name)
                arrays[path] = array
                entries.append(
                    ActivationEntry(
                        run_id=run.run_id,
                        algorithm_id=run.algorithm.id,
                        adapter_id=run.algorithm.adapter,
                        training_seed=int(run.experiment.seed),
                        capability=capability,
                        checkpoint_id=checkpoint.checkpoint_id,
                        training_step=int(checkpoint.training_step),
                        checkpoint_sha256=checkpoint_sha,
                        layer=spec.name,
                        source_layer=spec.source,
                        pooling=spec.pooling,
                        original_shape=original_shapes[spec.name],
                        feature_dim=int(array.shape[-1]),
                        dtype=str(array.dtype),
                        path=path,
                    )
                )
            del agent

    out = publish_activations(
        resolved=resolved,
        study=study,
        clips=clips_table(clips),
        clip_steps=n_steps,
        entries=entries,
        arrays=arrays,
        input_files=input_files,
        metadata={
            "device": device,
            "environment_fingerprint": manifest.environment_fingerprint,
            "observations_sha256": sha256_file(obs_file),
            "recurrent": False,
            "valid_step_mask": "all steps valid; burn-in frames are excluded for feed-forward models",
        },
        packages=["numpy", "pandas", "torch", "stable_baselines3", "gymnasium"],
        force=force,
    )
    logger.info("published %d activation array(s) to %s", len(entries), out, extra=extra)
    return out
