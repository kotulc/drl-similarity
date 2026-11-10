"""Artifact store, manifest verification, pooling, and scenario schema helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drl_similarity.activations.pooling import apply_pooling
from drl_similarity.artifacts.manifest import ArtifactManifest, ArtifactOutput, verify_artifact_dir
from drl_similarity.artifacts.store import ArtifactStore
from drl_similarity.errors import ArtifactError, ExtractionError
from drl_similarity.scenarios.schema import assign_splits, observation_sha256


def _publish(root: Path, text: str = "payload", *, force: bool = False) -> Path:
    store = ArtifactStore(root, artifact_id="metrics", config_hash="abc", stage="analyze")

    def writer(tmp: Path) -> ArtifactManifest:
        (tmp / "table.txt").write_text(text, encoding="utf-8")
        return store.build_manifest(outputs=[ArtifactOutput(path="table.txt", sha256="", role="table")])

    return store.publish(writer, force=force)


def test_publish_writes_verifiable_artifact(tmp_path: Path) -> None:
    out = _publish(tmp_path)

    assert verify_artifact_dir(out).outputs[0].bytes == len("payload")
    assert not list(tmp_path.glob(".metrics.tmp.*"))


def test_publish_collision_errors_without_force(tmp_path: Path) -> None:
    _publish(tmp_path)

    with pytest.raises(ArtifactError, match="already exists"):
        _publish(tmp_path)


def test_publish_force_creates_new_version(tmp_path: Path) -> None:
    first = _publish(tmp_path, "one")
    second = _publish(tmp_path, "two", force=True)

    assert second.name == "metrics-v2"
    assert (first / "table.txt").read_text(encoding="utf-8") == "one"


def test_verify_detects_changed_file(tmp_path: Path) -> None:
    out = _publish(tmp_path)
    (out / "table.txt").write_text("tampered", encoding="utf-8")

    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        verify_artifact_dir(out)


@pytest.mark.parametrize(
    ("method", "shape", "expected"),
    [
        ("none", (5, 7), (5, 7)),
        ("flatten", (5, 3, 4, 4), (5, 48)),
        ("global_average", (5, 3, 4, 4), (5, 3)),
    ],
)
def test_apply_pooling_shapes(method: str, shape: tuple[int, ...], expected: tuple[int, ...]) -> None:
    assert apply_pooling(np.ones(shape), method).shape == expected


@pytest.mark.parametrize(("method", "shape"), [("none", (5, 3, 4, 4)), ("global_average", (5, 7)), ("max", (5, 7))])
def test_apply_pooling_rejects_ambiguous_rank(method: str, shape: tuple[int, ...]) -> None:
    with pytest.raises(ExtractionError):
        apply_pooling(np.ones(shape), method)


def test_assign_splits_is_deterministic_and_stratified() -> None:
    labels = {f"{label}-{i}": label for label in ("a", "b") for i in range(10)}

    splits = assign_splits(labels, train_fraction=0.7, seed=1)

    assert splits == assign_splits(labels, train_fraction=0.7, seed=1)
    for label in ("a", "b"):
        assert sum(splits[f"{label}-{i}"] == "train" for i in range(10)) == 7


def test_observation_sha256_covers_dtype_and_shape() -> None:
    array = np.zeros((2, 4), np.uint8)

    assert observation_sha256(array) == observation_sha256(array.copy())
    assert observation_sha256(array) != observation_sha256(array.reshape(4, 2))
    assert observation_sha256(array) != observation_sha256(array.astype(np.int8))
