"""Scenario manifest models and observation checksum helpers."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from drl_similarity.errors import ScenarioError


ReviewStatus = Literal["pending", "approved", "rejected"]
Split = Literal["train", "test"]

REVIEW_STATUSES: tuple[str, ...] = ("pending", "approved", "rejected")
SPLITS: tuple[str, ...] = ("train", "test")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClipRecord(_Strict):
    """One fixed-length clip.

    Step indices are agent steps within the source episode. ``start_step`` and
    ``end_step`` are both inclusive, so a clip covers
    ``end_step - start_step + 1`` model steps. The stored observation array holds
    ``burn_in_steps`` extra leading frames followed by the clip frames.
    """

    clip_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    split: Split
    source_episode_id: str = Field(min_length=1)
    collection_seed: int
    start_step: int = Field(ge=0)
    anchor_step: int = Field(ge=0)
    end_step: int = Field(ge=0)
    burn_in_steps: int = Field(ge=0)
    anchor_event: str = Field(min_length=1)
    observation_key: str = Field(min_length=1)
    observation_sha256: str = Field(min_length=64, max_length=64)
    review_status: ReviewStatus = "pending"
    curator: str | None = None
    notes: str = ""
    created_at: str

    @model_validator(mode="after")
    def _ordered_window(self) -> ClipRecord:
        if self.end_step < self.start_step:
            raise ValueError("end_step must be >= start_step")
        return self

    @property
    def n_steps(self) -> int:
        return self.end_step - self.start_step + 1

    @property
    def reviewers(self) -> tuple[str, ...]:
        if not self.curator:
            return ()
        return tuple(name.strip() for name in self.curator.split(",") if name.strip())


class CollectorInfo(_Strict):
    """Provenance of the policy that generated candidate episodes."""

    policy: str = "checkpoint"
    run_id: str | None = None
    capability: str | None = None
    checkpoint_id: str | None = None
    checkpoint_sha256: str | None = None


class ScenarioManifest(_Strict):
    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    environment_fingerprint: str = Field(min_length=1)
    collector: CollectorInfo | None = None
    created_at: str | None = None
    synthetic: bool = False
    notes: str = ""
    clips: list[ClipRecord] = Field(default_factory=list)

    def active_clips(self, *, require_approved: bool = True) -> list[ClipRecord]:
        """Clips used downstream, in manifest order.

        Rejected clips are always excluded. Pending clips are excluded when
        approval is required (validation reports them as failures).
        """
        if require_approved:
            return [clip for clip in self.clips if clip.review_status == "approved"]
        return [clip for clip in self.clips if clip.review_status != "rejected"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def observation_sha256(array: np.ndarray) -> str:
    """Checksum of an observation array including its dtype and shape."""
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f"{arr.dtype.str}|{','.join(str(dim) for dim in arr.shape)}|".encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def load_manifest(path: Path) -> ScenarioManifest:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScenarioError(f"Unable to read scenario manifest: {path}") from exc
    except yaml.YAMLError as exc:
        raise ScenarioError(f"Invalid YAML in scenario manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioError(f"Scenario manifest root must be a mapping: {path}")
    try:
        return ScenarioManifest.model_validate(raw)
    except ValidationError as exc:
        lines = [f"Scenario manifest schema validation failed for {path}:"]
        for err in exc.errors():
            loc = ".".join(str(item) for item in err.get("loc", ()))
            lines.append(f"  - {loc or '<root>'}: {err.get('msg')} (input={err.get('input')!r})")
        raise ScenarioError("\n".join(lines)) from exc


def dump_manifest(manifest: ScenarioManifest, path: Path, *, header: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = manifest.model_dump(mode="json")
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, allow_unicode=True)
    text = ""
    if header:
        text = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines())
    path.write_text(text + body, encoding="utf-8")


def assign_splits(
    labels_by_clip: dict[str, str], *, train_fraction: float, seed: int
) -> dict[str, str]:
    """Deterministic clip-level train/test split, stratified by label.

    Splits are assigned per clip, never per frame. Every label with at least two
    clips receives at least one clip in each split.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[str]] = {}
    for clip_id, label in labels_by_clip.items():
        by_label.setdefault(label, []).append(clip_id)
    splits: dict[str, str] = {}
    for label in sorted(by_label):
        clip_ids = sorted(by_label[label])
        order = rng.permutation(len(clip_ids))
        n_train = int(round(len(clip_ids) * train_fraction))
        if len(clip_ids) >= 2:
            n_train = min(max(n_train, 1), len(clip_ids) - 1)
        for rank, position in enumerate(order):
            splits[clip_ids[int(position)]] = "train" if rank < n_train else "test"
    return splits


def load_clip_arrays(
    observations_file: Path,
    clips: list[ClipRecord],
    *,
    clip_steps: int,
) -> np.ndarray:
    """Load clip observations in the given order with burn-in frames removed.

    Returns an array shaped ``(clips, clip_steps, *observation_shape)``.
    """
    observations_file = Path(observations_file)
    if not observations_file.exists():
        raise ScenarioError(f"Missing scenario observations: {observations_file}")
    frames: list[np.ndarray] = []
    with np.load(observations_file, allow_pickle=False) as archive:
        for clip in clips:
            if clip.observation_key not in archive.files:
                raise ScenarioError(
                    f"Observation key {clip.observation_key!r} missing from {observations_file}",
                    context={"clip_id": clip.clip_id},
                )
            array = archive[clip.observation_key]
            expected = clip.burn_in_steps + clip_steps
            if array.shape[0] != expected:
                raise ScenarioError(
                    f"Clip {clip.clip_id} has {array.shape[0]} frames; expected {expected}",
                    context={"burn_in_steps": clip.burn_in_steps, "clip_steps": clip_steps},
                )
            frames.append(array[clip.burn_in_steps :])
    if not frames:
        raise ScenarioError("No clips available to load")
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ScenarioError("Clips have inconsistent observation shapes", context={"shapes": sorted(shapes)})
    return np.stack(frames, axis=0)
