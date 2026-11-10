"""Representation-geometry metrics for aligned sample sets.

All functions compare two sample-by-feature matrices whose rows are the *same
ordered stimuli*. Pass ``ids_x``/``ids_y`` to have that alignment verified;
mismatched or reordered IDs raise :class:`AnalysisError`.

Invariances (asserted in ``tests/unit/test_metrics_invariance.py``):

==========================  ========  =================  ===========
metric                      rotation  isotropic scaling  translation
==========================  ========  =================  ===========
``linear_cka``              yes       yes                yes
``rsa_spearman`` euclidean  yes       yes                yes
``rsa_spearman`` correlat.  no        yes                no (per-sample mean only)
``svcca``                   yes       yes                yes
``orthogonal_procrustes``   yes       yes                yes
==========================  ========  =================  ===========
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from drl_similarity.errors import AnalysisError
from drl_similarity.metrics.vector import as_matrix


EPS = 1e-12
RDM_DISTANCES: tuple[str, ...] = ("correlation", "euclidean", "cosine")


class InapplicableMetric(AnalysisError):
    """A metric cannot be computed for this pair (for example mismatched widths)."""


def validate_aligned_samples(ids_x: Sequence[object] | None, ids_y: Sequence[object] | None) -> None:
    """Require identical ordered sample IDs for a cross-representation comparison."""
    if ids_x is None and ids_y is None:
        return
    if ids_x is None or ids_y is None:
        raise AnalysisError("sample IDs must be provided for both representations or neither")
    if len(ids_x) != len(ids_y):
        raise AnalysisError(
            "sample ID lists differ in length",
            context={"n_x": len(ids_x), "n_y": len(ids_y)},
        )
    for position, (left, right) in enumerate(zip(ids_x, ids_y, strict=True)):
        if left != right:
            same_set = set(ids_x) == set(ids_y)
            raise AnalysisError(
                "sample IDs are not aligned"
                + (" (same IDs in a different order)" if same_set else ""),
                context={"first_mismatch": position, "x": left, "y": right},
            )


def _pair(
    x: np.ndarray,
    y: np.ndarray,
    ids_x: Sequence[object] | None,
    ids_y: Sequence[object] | None,
) -> tuple[np.ndarray, np.ndarray]:
    a = as_matrix(x, name="x")
    b = as_matrix(y, name="y")
    if a.shape[0] != b.shape[0]:
        raise AnalysisError(
            "representations must have the same number of samples",
            context={"n_x": a.shape[0], "n_y": b.shape[0]},
        )
    if ids_x is not None and len(ids_x) != a.shape[0]:
        raise AnalysisError("ids_x length does not match the number of samples")
    validate_aligned_samples(ids_x, ids_y)
    return a, b


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


# --------------------------------------------------------------------------- linear CKA


def linear_cka(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ids_x: Sequence[object] | None = None,
    ids_y: Sequence[object] | None = None,
    eps: float = EPS,
) -> float:
    """Linear centered kernel alignment (Kornblith et al., 2019).

    ``CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)`` with columns
    centered over samples. The value is in ``[0, 1]`` and is invariant to
    orthogonal transformations and isotropic scaling of either representation.
    Feature counts may differ. Returns NaN when either representation has no
    variance across samples (denominator below ``eps``).
    """
    a, b = _pair(x, y, ids_x, ids_y)
    a = _center(a)
    b = _center(b)
    n = a.shape[0]
    if a.shape[1] > n or b.shape[1] > n:
        gram_a = a @ a.T
        gram_b = b @ b.T
        cross = float(np.sum(gram_a * gram_b))
        norm_a = float(np.linalg.norm(gram_a))
        norm_b = float(np.linalg.norm(gram_b))
    else:
        cross = float(np.linalg.norm(b.T @ a) ** 2)
        norm_a = float(np.linalg.norm(a.T @ a))
        norm_b = float(np.linalg.norm(b.T @ b))
    denominator = norm_a * norm_b
    if denominator < eps:
        return float("nan")
    return float(np.clip(cross / denominator, 0.0, 1.0))


def clip_pair_cka(activations: np.ndarray, *, eps: float = EPS) -> np.ndarray:
    """Linear CKA between every pair of clips, using aligned relative steps as samples.

    ``activations`` is ``(clips, steps, features)``. Entry ``(i, j)`` is the
    CKA between clip ``i`` and clip ``j`` where sample ``t`` of each is the
    activation at relative step ``t``. Clips with no variation across steps
    give NaN rows and columns.
    """
    array = np.asarray(activations, dtype=np.float64)
    if array.ndim != 3:
        raise AnalysisError(
            "activations must be shaped (clips, steps, features)",
            context={"shape": tuple(array.shape)},
        )
    n_clips = array.shape[0]
    centered = array - array.mean(axis=1, keepdims=True)
    grams = np.einsum("ctf,cuf->ctu", centered, centered).reshape(n_clips, -1)
    norms = np.linalg.norm(grams, axis=1)
    degenerate = norms < eps
    unit = grams / np.where(degenerate, 1.0, norms)[:, None]
    matrix = np.clip(unit @ unit.T, 0.0, 1.0)
    matrix[degenerate, :] = np.nan
    matrix[:, degenerate] = np.nan
    return matrix


# --------------------------------------------------------------------------- RSA


def compute_rdm(x: np.ndarray, distance: str = "correlation") -> np.ndarray:
    """Square representational dissimilarity matrix between samples.

    ``correlation`` and ``euclidean`` use ``rsatoolbox`` (falling back to the
    equivalent SciPy computation if it cannot be imported); ``cosine`` uses
    SciPy. rsatoolbox's Euclidean RDM is squared and divided by the feature
    count, a monotone transform that leaves Spearman RSA unchanged.
    """
    array = as_matrix(x, name="x")
    if distance not in RDM_DISTANCES:
        raise AnalysisError(
            f"Unknown first_order_distance {distance!r}; valid: {', '.join(RDM_DISTANCES)}"
        )
    if array.shape[0] < 2:
        raise AnalysisError("an RDM needs at least two samples")
    if distance in ("correlation", "euclidean"):
        try:
            import rsatoolbox

            dataset = rsatoolbox.data.Dataset(array)
            rdms = rsatoolbox.rdm.calc_rdm(dataset, method=distance)
            return np.asarray(rdms.get_matrices()[0], dtype=np.float64)
        except ImportError:
            pass
    with np.errstate(invalid="ignore", divide="ignore"):
        return squareform(pdist(array, metric=distance), checks=False)


def rsa_from_rdms(rdm_x: np.ndarray, rdm_y: np.ndarray, *, mask: np.ndarray | None = None) -> float:
    """Spearman correlation between the upper triangles of two square RDMs.

    ``mask`` optionally selects which sample pairs to include (used to drop
    duplicated samples during a bootstrap). Pairs that are NaN in either RDM are
    excluded. Returns NaN when fewer than three pairs remain or either RDM is
    constant.
    """
    if rdm_x.shape != rdm_y.shape or rdm_x.ndim != 2 or rdm_x.shape[0] != rdm_x.shape[1]:
        raise AnalysisError(
            "RDMs must be square with identical shapes",
            context={"x": rdm_x.shape, "y": rdm_y.shape},
        )
    upper = np.triu_indices(rdm_x.shape[0], k=1)
    left = rdm_x[upper]
    right = rdm_y[upper]
    keep = ~(np.isnan(left) | np.isnan(right))
    if mask is not None:
        keep &= mask[upper]
    left = left[keep]
    right = right[keep]
    if left.size < 3:
        return float("nan")
    rank_left = rankdata(left)
    rank_right = rankdata(right)
    rank_left -= rank_left.mean()
    rank_right -= rank_right.mean()
    denominator = float(np.linalg.norm(rank_left) * np.linalg.norm(rank_right))
    if denominator < EPS:
        return float("nan")
    return float(np.clip(rank_left @ rank_right / denominator, -1.0, 1.0))


def rsa_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    first_order_distance: str = "correlation",
    ids_x: Sequence[object] | None = None,
    ids_y: Sequence[object] | None = None,
) -> float:
    """Representational similarity analysis: Spearman correlation of two RDMs."""
    a, b = _pair(x, y, ids_x, ids_y)
    return rsa_from_rdms(compute_rdm(a, first_order_distance), compute_rdm(b, first_order_distance))


# --------------------------------------------------------------------------- SVCCA


def _principal_subspace(x: np.ndarray, variance_retained: float) -> np.ndarray:
    """Orthonormal sample-space basis of the top singular directions of centered ``x``."""
    centered = _center(x)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    if singular.size == 0 or singular[0] < EPS:
        return left[:, :0]
    significant = singular > singular[0] * 1e-10
    left = left[:, significant]
    energy = singular[significant] ** 2
    cumulative = np.cumsum(energy) / energy.sum()
    keep = int(np.searchsorted(cumulative, variance_retained - 1e-12) + 1)
    return left[:, : min(keep, left.shape[1])]


def svcca(
    x: np.ndarray,
    y: np.ndarray,
    *,
    variance_retained: float = 0.99,
    ids_x: Sequence[object] | None = None,
    ids_y: Sequence[object] | None = None,
) -> float:
    """Singular vector CCA (Raghu et al., 2017): mean canonical correlation.

    Each representation is centered and reduced to the singular directions
    explaining ``variance_retained`` of its variance; canonical correlations
    between the two subspaces are then averaged. The value is in ``[0, 1]``.
    With few samples relative to the retained dimensions the canonical
    correlations are inflated toward one; callers should report the sample
    count next to the estimate.
    """
    if not 0.0 < variance_retained <= 1.0:
        raise AnalysisError(
            "variance_retained must be in (0, 1]", context={"variance_retained": variance_retained}
        )
    a, b = _pair(x, y, ids_x, ids_y)
    basis_a = _principal_subspace(a, variance_retained)
    basis_b = _principal_subspace(b, variance_retained)
    if basis_a.shape[1] == 0 or basis_b.shape[1] == 0:
        return float("nan")
    correlations = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    count = min(basis_a.shape[1], basis_b.shape[1])
    return float(np.clip(correlations[:count], 0.0, 1.0).mean())


# --------------------------------------------------------------------------- Procrustes


def _procrustes_disparity(a: np.ndarray, b: np.ndarray) -> float:
    a = _center(a)
    b = _center(b)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < EPS or norm_b < EPS:
        return float("nan")
    singular = np.linalg.svd((a / norm_a).T @ (b / norm_b), compute_uv=False)
    return float(np.clip(1.0 - singular.sum() ** 2, 0.0, 1.0))


def orthogonal_procrustes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ids_x: Sequence[object] | None = None,
    ids_y: Sequence[object] | None = None,
) -> float:
    """Normalized residual after the best orthogonal alignment (a distance).

    Both matrices are centered and scaled to unit Frobenius norm; the result is
    ``1 - (sum of singular values of Xn^T Yn)^2``, the standard Procrustes
    disparity in ``[0, 1]`` (0 means identical up to rotation, reflection,
    isotropic scaling, and translation). Requires equal feature counts;
    otherwise :class:`InapplicableMetric` is raised.
    """
    a, b = _pair(x, y, ids_x, ids_y)
    if a.shape[1] != b.shape[1]:
        raise InapplicableMetric(
            "orthogonal_procrustes requires identical feature counts",
            context={"features_x": a.shape[1], "features_y": b.shape[1]},
        )
    return _procrustes_disparity(a, b)


# --------------------------------------------------------------------------- reusable representation


def reduce_to_sample_rank(x: np.ndarray) -> np.ndarray:
    """Project wide matrices onto their row space without changing sample geometry.

    For ``n`` samples and ``d > n`` features the returned ``(n, r)`` matrix has
    exactly the same Gram matrix ``X X^T``, so every Gram-based metric (linear
    CKA, SVCCA, Procrustes, Euclidean/cosine distances) is unchanged while
    repeated evaluation becomes much cheaper.
    """
    array = as_matrix(x, name="x")
    n, d = array.shape
    if d <= n:
        return array
    left, singular, _ = np.linalg.svd(array, full_matrices=False)
    keep = singular > (singular[0] * 1e-12 if singular.size and singular[0] > 0 else 0.0)
    if not keep.any():
        return np.zeros((n, 1), dtype=np.float64)
    return left[:, keep] * singular[keep]


class Representation:
    """One aligned sample matrix with cached reductions for repeated comparisons."""

    def __init__(self, matrix: np.ndarray, sample_ids: Sequence[str]) -> None:
        self.matrix = as_matrix(matrix, name="representation")
        self.sample_ids = tuple(sample_ids)
        if len(self.sample_ids) != self.matrix.shape[0]:
            raise AnalysisError(
                "sample_ids length does not match the number of samples",
                context={"ids": len(self.sample_ids), "samples": self.matrix.shape[0]},
            )
        self._rdms: dict[str, np.ndarray] = {}

    @property
    def n_samples(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1])

    @cached_property
    def reduced(self) -> np.ndarray:
        return reduce_to_sample_rank(self.matrix)

    def rdm(self, distance: str) -> np.ndarray:
        if distance not in self._rdms:
            self._rdms[distance] = compute_rdm(self.matrix, distance)
        return self._rdms[distance]


def evaluate_geometry(
    metric_id: str,
    rep_x: Representation,
    rep_y: Representation,
    *,
    index: np.ndarray | None = None,
    first_order_distance: str = "correlation",
    variance_retained: float = 0.99,
) -> float:
    """Evaluate a geometry metric on two aligned representations.

    ``index`` selects (possibly repeated) samples, as produced by a clip
    bootstrap. Results equal the plain functions above applied to the indexed
    full matrices; reductions and RDMs are computed once and reused.
    """
    validate_aligned_samples(rep_x.sample_ids, rep_y.sample_ids)
    if metric_id == "orthogonal_procrustes" and rep_x.n_features != rep_y.n_features:
        raise InapplicableMetric(
            "orthogonal_procrustes requires identical feature counts",
            context={"features_x": rep_x.n_features, "features_y": rep_y.n_features},
        )
    if metric_id == "rsa_spearman":
        rdm_x = rep_x.rdm(first_order_distance)
        rdm_y = rep_y.rdm(first_order_distance)
        if index is None:
            return rsa_from_rdms(rdm_x, rdm_y)
        grid = np.ix_(index, index)
        distinct = index[:, None] != index[None, :]
        return rsa_from_rdms(rdm_x[grid], rdm_y[grid], mask=distinct)
    left = rep_x.reduced if index is None else rep_x.reduced[index]
    right = rep_y.reduced if index is None else rep_y.reduced[index]
    if metric_id == "linear_cka":
        return linear_cka(left, right)
    if metric_id == "svcca":
        return svcca(left, right, variance_retained=variance_retained)
    if metric_id == "orthogonal_procrustes":
        return _procrustes_disparity(left, right)
    raise AnalysisError(f"Unknown geometry metric {metric_id!r}")
