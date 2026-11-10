"""Metric invariances, sample alignment, pair construction, and clip bootstrap."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from scipy.stats import ortho_group

from drl_similarity.errors import AnalysisError
from drl_similarity.metrics import geometry
from drl_similarity.metrics.bootstrap import stratified_clip_resample
from drl_similarity.metrics.pairs import (
    between_scenario_clip_pairs,
    build_frame_pairs,
    within_scenario_clip_pairs,
)
from drl_similarity.metrics.registry import METRICS
from drl_similarity.metrics.vector import cosine_similarity_matrix


RNG = np.random.default_rng(0)
X = RNG.normal(size=(60, 12))
ROTATION = ortho_group.rvs(12, random_state=0)
NOISE = RNG.normal(size=X.shape)

# RSA is rotation invariant only for a rotation-invariant first-order distance
RSA_EUCLIDEAN = partial(geometry.rsa_spearman, first_order_distance="euclidean")
SIMILARITIES = [geometry.linear_cka, RSA_EUCLIDEAN, geometry.svcca]
GEOMETRY = [*SIMILARITIES, geometry.orthogonal_procrustes]

CLIP_IDS = ["a1", "a2", "a3", "b1", "b2"]
SCENARIOS = ["a", "a", "a", "b", "b"]


def test_registry_exposes_exactly_the_planned_metrics() -> None:
    assert METRICS.ids() == sorted(
        [
            "centroid_dispersion", "correlation", "cosine", "linear_cka", "nearest_centroid_accuracy",
            "normalized_euclidean", "orthogonal_procrustes", "rsa_spearman", "silhouette", "svcca",
            "temporal_cosine", "within_between_gap",
        ]
    )  # fmt: skip


@pytest.mark.parametrize("metric", SIMILARITIES)
@pytest.mark.parametrize("transform", [X, X @ ROTATION, 3.0 * X], ids=["identical", "rotated", "scaled"])
def test_geometry_similarity_invariances(metric, transform: np.ndarray) -> None:
    """Identity, orthogonal rotation, and isotropic scaling leave the similarity near one."""
    assert metric(X, transform) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("transform", [X, X @ ROTATION, 3.0 * X], ids=["identical", "rotated", "scaled"])
def test_orthogonal_procrustes_invariances(transform: np.ndarray) -> None:
    assert geometry.orthogonal_procrustes(X, transform) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("metric", SIMILARITIES)
def test_geometry_similarity_drops_with_noise(metric) -> None:
    assert metric(X, X + 2.0 * NOISE) < metric(X, X + 0.1 * NOISE) < 1.0


def test_raw_cosine_is_not_rotation_invariant() -> None:
    """Coordinate-wise cosine depends on the basis, unlike the geometry metrics."""
    paired = np.diag(cosine_similarity_matrix(X, X @ ROTATION))

    assert paired.mean() < 0.9


@pytest.mark.parametrize("metric", GEOMETRY)
def test_geometry_rejects_shuffled_sample_ids(metric) -> None:
    ids = [f"clip{i}" for i in range(len(X))]

    with pytest.raises(AnalysisError, match="not aligned"):
        metric(X, X, ids_x=ids, ids_y=ids[::-1])


def test_linear_cka_accepts_different_feature_counts() -> None:
    assert 0.0 <= geometry.linear_cka(X, X[:, :5]) <= 1.0


def test_within_scenario_pairs_exclude_same_clip() -> None:
    pairs = within_scenario_clip_pairs(CLIP_IDS, SCENARIOS, "a")

    assert {(p.clip_id_a, p.clip_id_b) for p in pairs} == {("a1", "a2"), ("a1", "a3"), ("a2", "a3")}


def test_between_scenario_pairs_are_balanced() -> None:
    pairs = between_scenario_clip_pairs(CLIP_IDS, SCENARIOS, "a", "b", rng=np.random.default_rng(0))

    assert len({p.clip_id_a for p in pairs}) == len({p.clip_id_b for p in pairs}) == 2
    assert all(p.scenario_a == "a" and p.scenario_b == "b" for p in pairs)


@pytest.mark.parametrize(("pair_set", "per_clip_pair"), [("cross_clip_all", 16), ("cross_clip_aligned_step", 4)])
def test_build_frame_pairs_never_pairs_a_clip_with_itself(pair_set: str, per_clip_pair: int) -> None:
    clip_pairs = within_scenario_clip_pairs(CLIP_IDS, SCENARIOS, "a")

    frame_pairs = build_frame_pairs(clip_pairs, pair_set, n_steps=4)

    assert len(frame_pairs) == per_clip_pair * len(clip_pairs)
    assert (frame_pairs["clip_id_a"] != frame_pairs["clip_id_b"]).all()


def test_stratified_clip_resample_draws_whole_clips_within_scenario() -> None:
    """Each replicate holds clip indices (not frames), drawn from the clip's own scenario."""
    labels = np.asarray(SCENARIOS)

    indices = stratified_clip_resample(SCENARIOS, 200, np.random.default_rng(0))

    assert indices.shape == (200, len(SCENARIOS))
    assert (labels[indices] == labels[None, :]).all()
    assert len(np.unique(indices[:, 0])) > 1
