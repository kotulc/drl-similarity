"""Clip-level bootstrap.

Frames from one clip are correlated, so the resampling unit is always a whole
clip. Clips are drawn with replacement *within each stratum* (scenario), which
keeps every replicate balanced exactly like the original dataset.

Pair-based statistics are rebuilt inside each replicate: a replicate that draws
clip ``i`` ``w_i`` times and clip ``j`` ``w_j`` times contains ``w_i * w_j``
copies of the pair ``(i, j)``, and pairs of a clip with itself (including a
clip drawn twice) are excluded. :func:`weighted_pair_mean` evaluates exactly
that rebuilt pair set for all replicates at once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from drl_similarity.errors import AnalysisError


#: Warning recorded when a point estimate lies outside its own percentile interval.
ESTIMATE_OUTSIDE_INTERVAL = "estimate_outside_interval:bootstrap_bias"


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    replicates: int
    successful: int
    confidence_level: float
    warnings: tuple[str, ...] = ()


def stratified_clip_resample(
    strata: Sequence[str], replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw clip indices with replacement within each stratum.

    Returns an integer array shaped ``(replicates, n_clips)``. Column ``p`` of
    every replicate holds a clip drawn from the stratum of original clip ``p``,
    so stratum sizes (and any per-position labels) are preserved.
    """
    if replicates < 0:
        raise AnalysisError("bootstrap replicates must be non-negative")
    labels = np.asarray(list(strata)).astype(str)
    n_clips = labels.shape[0]
    indices = np.empty((replicates, n_clips), dtype=np.int64)
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        indices[:, members] = rng.choice(members, size=(replicates, members.size), replace=True)
    return indices


def resample_counts(indices: np.ndarray, n_clips: int) -> np.ndarray:
    """How many times each clip was drawn in each replicate: ``(replicates, n_clips)``."""
    counts = np.zeros((indices.shape[0], n_clips), dtype=np.float64)
    rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    np.add.at(counts, (rows, indices.ravel()), 1.0)
    return counts


def percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float, int]:
    """Percentile interval over finite replicate values; returns (low, high, n_finite)."""
    if not 0.0 < confidence_level < 1.0:
        raise AnalysisError(
            "confidence_level must be in (0, 1)", context={"confidence_level": confidence_level}
        )
    array = np.asarray(values, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high), int(finite.size)


def summarize(
    estimate: float, replicate_values: np.ndarray, confidence_level: float
) -> BootstrapResult:
    values = np.asarray(replicate_values, dtype=np.float64).ravel()
    low, high, successful = percentile_interval(values, confidence_level)
    warnings: list[str] = []
    if values.size == 0:
        warnings.append("bootstrap_disabled")
    elif successful < values.size:
        warnings.append(f"bootstrap_failed_replicates:{values.size - successful}")
    if successful and np.isfinite(estimate):
        tolerance = 1e-9 * max(1.0, abs(float(estimate)))
        if estimate < low - tolerance or estimate > high + tolerance:
            # The resampled statistic is biased relative to the point estimate (typical for
            # small-sample CKA/SVCCA and variance-like statistics); a percentile interval
            # does not correct for that, so the row is flagged instead of silently reported.
            warnings.append(ESTIMATE_OUTSIDE_INTERVAL)
    return BootstrapResult(
        estimate=float(estimate),
        ci_low=low,
        ci_high=high,
        replicates=int(values.size),
        successful=successful,
        confidence_level=float(confidence_level),
        warnings=tuple(warnings),
    )


def bootstrap_statistic(
    statistic: Callable[[np.ndarray], float],
    indices: np.ndarray,
    *,
    estimate: float,
    confidence_level: float,
) -> BootstrapResult:
    """Evaluate ``statistic(clip_indices)`` on every resampled row.

    ``statistic`` receives one row of :func:`stratified_clip_resample` and must
    rebuild whatever it needs (pairs, centroids, classifiers) from those clips.
    A replicate that raises :class:`AnalysisError` or :class:`ValueError`, or
    returns a non-finite value, is counted as unsuccessful.
    """
    values = np.full(indices.shape[0], np.nan, dtype=np.float64)
    for row, clip_indices in enumerate(indices):
        try:
            values[row] = float(statistic(clip_indices))
        except (AnalysisError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            values[row] = np.nan
    return summarize(estimate, values, confidence_level)


def weighted_pair_mean(
    matrix: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    counts: np.ndarray | None = None,
    *,
    exclude_same_clip: bool = True,
) -> np.ndarray:
    """Mean of ``matrix[i, j]`` over clip pairs ``i in rows, j in cols``.

    ``matrix`` is a clip-by-clip matrix of per-pair values (NaN where undefined).
    With ``counts`` (``(replicates, n_clips)`` draw counts) the result has one
    value per replicate, weighting pair ``(i, j)`` by ``counts[i] * counts[j]``;
    without it a single-element array holds the point estimate. Same-clip pairs
    are excluded so a clip drawn twice is never compared with itself. When
    ``rows`` and ``cols`` are the same set every unordered pair is counted twice
    in both numerator and denominator, which leaves the mean unchanged.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    block = np.asarray(matrix, dtype=np.float64)[np.ix_(rows, cols)]
    valid = ~np.isnan(block)
    if exclude_same_clip:
        valid &= rows[:, None] != cols[None, :]
    values = np.where(valid, block, 0.0)
    weight = valid.astype(np.float64)
    if counts is None:
        row_counts = np.ones((1, rows.size))
        col_counts = np.ones((1, cols.size))
    else:
        row_counts = counts[:, rows]
        col_counts = counts[:, cols]
    numerator = np.einsum("ra,ab,rb->r", row_counts, values, col_counts)
    denominator = np.einsum("ra,ab,rb->r", row_counts, weight, col_counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def weighted_clip_mean(
    values: np.ndarray, members: np.ndarray, counts: np.ndarray | None = None
) -> np.ndarray:
    """Mean of per-clip ``values`` over ``members``, optionally per bootstrap replicate."""
    members = np.asarray(members, dtype=np.int64)
    selected = np.asarray(values, dtype=np.float64)[members]
    valid = ~np.isnan(selected)
    filled = np.where(valid, selected, 0.0)
    weights = np.ones((1, members.size)) if counts is None else counts[:, members]
    numerator = weights @ filled
    denominator = weights @ valid.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)
