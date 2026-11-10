"""Activation artifact layout, writer, and reader.

Layout of ``artifacts/<study_id>/activations/``::

    manifest.json                       checksums, lineage, and entry metadata
    resolved-config.yaml                study configuration used for extraction
    clips.parquet                       clip order shared by every array
    <run_id>/<capability>/<layer>.npz   ``activations`` (clips, steps, features)
                                        float32 and ``valid_mask`` (clips, steps)

This module has no training-backend imports so analysis and the dashboard can
read activations without loading a model.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drl_similarity.artifacts.manifest import (
    ArtifactManifest,
    ArtifactOutput,
    capture_package_versions,
    try_git_commit,
    verify_artifact_dir,
)
from drl_similarity.artifacts.store import ArtifactStore
from drl_similarity.config import REPO_ROOT, ResolvedConfig, StudyConfig, write_resolved_config
from drl_similarity.errors import ArtifactError, ExtractionError
from drl_similarity.study_io import (
    ACTIVATIONS_ARTIFACT_ID,
    extraction_stage_hash,
    make_input,
    study_root,
)
from drl_similarity.tables import CLIP_COLUMNS, require_columns


ACTIVATION_SCHEMA = "activations/v1"
EXTRACT_COMMAND = "drl-sim extract --config <study.yaml>"


@dataclass(frozen=True)
class ActivationEntry:
    """Metadata for one (run, capability, layer) activation array."""

    run_id: str
    algorithm_id: str
    adapter_id: str
    training_seed: int
    capability: str
    checkpoint_id: str
    training_step: int
    checkpoint_sha256: str
    layer: str
    source_layer: str
    pooling: str
    original_shape: tuple[int, ...]
    feature_dim: int
    dtype: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["original_shape"] = list(self.original_shape)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActivationEntry:
        data = dict(payload)
        data["original_shape"] = tuple(int(dim) for dim in data.get("original_shape", ()))
        return cls(**data)


@dataclass(frozen=True)
class ActivationBundle:
    directory: Path
    manifest: ArtifactManifest
    clips: pd.DataFrame
    entries: tuple[ActivationEntry, ...]
    clip_steps: int

    @property
    def metadata(self) -> dict[str, Any]:
        return self.manifest.metadata

    def find(self, run_id: str, capability: str, layer: str) -> ActivationEntry | None:
        for entry in self.entries:
            if (entry.run_id, entry.capability, entry.layer) == (run_id, capability, layer):
                return entry
        return None

    def load(self, entry: ActivationEntry) -> np.ndarray:
        """Return the ``(clips, steps, features)`` array for an entry."""
        path = self.directory / entry.path
        if not path.exists():
            raise ArtifactError(
                f"Missing activation array: {path}", context={"regenerate": EXTRACT_COMMAND}
            )
        with np.load(path, allow_pickle=False) as archive:
            array = np.asarray(archive["activations"])
            mask = np.asarray(archive["valid_mask"]) if "valid_mask" in archive.files else None
        expected = (len(self.clips), self.clip_steps, entry.feature_dim)
        if array.shape != expected:
            raise ArtifactError(
                f"Activation array {entry.path} has shape {array.shape}; expected {expected}",
                context={"regenerate": EXTRACT_COMMAND},
            )
        if mask is not None and not bool(mask.all()):
            raise ArtifactError(
                f"Activation array {entry.path} contains masked (invalid) steps, which "
                "feed-forward analysis does not support",
            )
        return array


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def activation_relpath(run_id: str, capability: str, layer: str) -> str:
    return f"{_safe(run_id)}/{_safe(capability)}/{_safe(layer)}.npz"


def publish_activations(
    *,
    resolved: ResolvedConfig,
    study: StudyConfig,
    clips: pd.DataFrame,
    clip_steps: int,
    entries: list[ActivationEntry],
    arrays: dict[str, np.ndarray],
    input_files: list[tuple[Path, str]],
    metadata: dict[str, Any] | None = None,
    packages: list[str] | None = None,
    force: bool = False,
) -> Path:
    """Atomically publish activation arrays; ``arrays`` is keyed by ``entry.path``."""
    require_columns(clips, CLIP_COLUMNS, name="clips")
    n_clips = len(clips)
    store = ArtifactStore(
        study_root(study),
        artifact_id=ACTIVATIONS_ARTIFACT_ID,
        config_hash=resolved.config_hash,
        stage="extract",
        collision=study.artifacts.collision,
    )
    artifact_dir = store.target

    def writer(tmp: Path) -> ArtifactManifest:
        outputs: list[ArtifactOutput] = []
        clips.to_parquet(tmp / "clips.parquet", index=False)
        outputs.append(ArtifactOutput(path="clips.parquet", sha256="", role="clips"))
        write_resolved_config(tmp / "resolved-config.yaml", resolved.canonical)
        outputs.append(ArtifactOutput(path="resolved-config.yaml", sha256="", role="resolved_config"))
        for entry in entries:
            array = np.asarray(arrays[entry.path])
            expected = (n_clips, clip_steps, entry.feature_dim)
            if array.shape != expected:
                raise ExtractionError(
                    f"Activation for {entry.capability}/{entry.layer} has shape {array.shape}; "
                    f"expected {expected}"
                )
            if not np.isfinite(array).all():
                raise ExtractionError(
                    f"Activation for {entry.capability}/{entry.layer} contains non-finite values",
                    context={"non_finite": int((~np.isfinite(array)).sum())},
                )
            target = tmp / entry.path
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                activations=array.astype(np.float32, copy=False),
                valid_mask=np.ones((n_clips, clip_steps), dtype=bool),
            )
            outputs.append(ArtifactOutput(path=entry.path, sha256="", role="activations"))
        payload: dict[str, Any] = {
            "schema": ACTIVATION_SCHEMA,
            "study_id": study.id,
            "stage_config_hash": extraction_stage_hash(resolved),
            "clip_steps": int(clip_steps),
            "n_clips": int(n_clips),
            "clip_ids": [str(value) for value in clips["clip_id"]],
            "dtype": study.extraction.dtype,
            "batch_size": int(study.extraction.batch_size),
            "entries": [entry.to_dict() for entry in entries],
            "synthetic": False,
        }
        payload.update(metadata or {})
        return store.build_manifest(
            inputs=[make_input(path, role=role, artifact_dir=artifact_dir) for path, role in input_files],
            outputs=outputs,
            metadata=payload,
            git_commit=try_git_commit(REPO_ROOT),
            package_versions=capture_package_versions(packages or ["numpy", "pandas"]),
        )

    return store.publish(writer, force=force)


def load_activation_bundle(directory: Path, *, verify: bool = True) -> ActivationBundle:
    """Read (and by default checksum-verify) an activation artifact directory."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise ArtifactError(
            f"Missing activation manifest: {manifest_path}",
            context={"regenerate": EXTRACT_COMMAND},
        )
    manifest = verify_artifact_dir(directory) if verify else ArtifactManifest.read(manifest_path)
    metadata = manifest.metadata
    if metadata.get("schema") != ACTIVATION_SCHEMA:
        raise ArtifactError(
            f"Unsupported activation schema in {manifest_path}",
            context={"found": metadata.get("schema"), "expected": ACTIVATION_SCHEMA},
        )
    clips_path = directory / "clips.parquet"
    if not clips_path.exists():
        raise ArtifactError(f"Missing {clips_path}", context={"regenerate": EXTRACT_COMMAND})
    clips = pd.read_parquet(clips_path)
    require_columns(clips, CLIP_COLUMNS, name=str(clips_path), regenerate=EXTRACT_COMMAND)
    try:
        entries = tuple(ActivationEntry.from_dict(item) for item in metadata["entries"])
        clip_steps = int(metadata["clip_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(
            f"Activation manifest metadata is incomplete: {manifest_path}",
            context={"error": repr(exc), "regenerate": EXTRACT_COMMAND},
        ) from exc
    return ActivationBundle(
        directory=directory,
        manifest=manifest,
        clips=clips.sort_values("clip_index").reset_index(drop=True),
        entries=entries,
        clip_steps=clip_steps,
    )
