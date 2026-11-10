"""Vector, dispersion, temporal, and classification metrics.

Every function is pure and operates on ordinary NumPy arrays. Undefined values
(for example the cosine of a zero vector) are returned as NaN so callers can
count and report them instead of silently dropping them.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestCentroid

from drl_similarity.errors import AnalysisError


EPS = 1e-12


def as_matrix(values: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    """Validate a sample-by-feature matrix and return it as float64."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise AnalysisError(
            f"{name} must be a sample-by-feature matrix (rank 2)",
            context={"shape": tuple(array.shape)},
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise AnalysisError(f"{name} must not be empty", context={"shape": tuple(array.shape)})
    return array


# --------------------------------------------------------------------------- pairwise matrices


def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Pairwise cosine similarity between rows; zero vectors give NaN rows/columns."""
    a = as_matrix(x, name="x")
    b = a if y is None else as_matrix(y, name="y")
    if a.shape[1] != b.shape[1]:
        raise AnalysisError(
            "cosine requires matching feature counts",
            context={"x": a.shape, "y": b.shape},
        )
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    zero_a = norm_a < EPS
    zero_b = norm_b < EPS
    safe_a = np.where(zero_a, 1.0, norm_a)
    safe_b = np.where(zero_b, 1.0, norm_b)
    similarity = (a / safe_a[:, None]) @ (b / safe_b[:, None]).T
    np.clip(similarity, -1.0, 1.0, out=similarity)
    similarity[zero_a, :] = np.nan
    similarity[:, zero_b] = np.nan
    return similarity


def correlation_similarity_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Pairwise Pearson correlation between rows (each vector centered on its own mean)."""
    a = as_matrix(x, name="x")
    b = a if y is None else as_matrix(y, name="y")
    centered_a = a - a.mean(axis=1, keepdims=True)
    centered_b = centered_a if y is None else b - b.mean(axis=1, keepdims=True)
    return cosine_similarity_matrix(centered_a, None if y is None else centered_b)


def count_zero_vectors(x: np.ndarray) -> int:
    return int((np.linalg.norm(as_matrix(x), axis=1) < EPS).sum())


@dataclass(frozen=True)
class Standardizer:
    """Per-feature z-score parameters fitted on a declared subset of samples."""

    mean: np.ndarray
    scale: np.ndarray
    n_constant: int
    fitted_on: str

    def transform(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float64)
        return (array - self.mean) / self.scale


def fit_standardizer(x: np.ndarray, *, fitted_on: str = "train_clips") -> Standardizer:
    """Fit feature means and standard deviations; constant features keep scale 1."""
    array = as_matrix(x, name="standardizer input")
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    constant = std < EPS
    scale = np.where(constant, 1.0, std)
    return Standardizer(mean=mean, scale=scale, n_constant=int(constant.sum()), fitted_on=fitted_on)


def normalized_euclidean_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Pairwise Euclidean distance divided by ``sqrt(n_features)``.

    Inputs are expected to be standardized already (see :func:`fit_standardizer`),
    which makes the value comparable across layers of different width.
    """
    a = as_matrix(x, name="x")
    b = a if y is None else as_matrix(y, name="y")
    if a.shape[1] != b.shape[1]:
        raise AnalysisError(
            "normalized_euclidean requires matching feature counts",
            context={"x": a.shape, "y": b.shape},
        )
    sq_a = np.einsum("ij,ij->i", a, a)
    sq_b = np.einsum("ij,ij->i", b, b)
    squared = sq_a[:, None] + sq_b[None, :] - 2.0 * (a @ b.T)
    np.maximum(squared, 0.0, out=squared)
    if y is None:
        np.fill_diagonal(squared, 0.0)
    return np.sqrt(squared) / np.sqrt(a.shape[1])


# --------------------------------------------------------------------------- clip-level reductions


def clip_centroids(activations: np.ndarray) -> np.ndarray:
    """Mean activation per clip for a ``(clips, steps, features)`` array."""
    array = np.asarray(activations, dtype=np.float64)
    if array.ndim != 3:
        raise AnalysisError(
            "activations must be shaped (clips, steps, features)",
            context={"shape": tuple(array.shape)},
        )
    return array.mean(axis=1)


def reduce_clip_blocks(
    frame_matrix: np.ndarray, n_clips: int, n_steps: int, pair_set: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a frame-by-frame matrix into a clip-by-clip matrix for one pair set.

    ``frame_matrix`` is ordered clip-major (all steps of clip 0, then clip 1, ...).
    Returns ``(mean, n_valid, n_nan)`` where each is ``(clips, clips)``.
    ``cross_clip_all`` averages every step pair; ``cross_clip_aligned_step``
    averages only pairs at the same relative step.
    """
    expected = (n_clips * n_steps, n_clips * n_steps)
    if frame_matrix.shape != expected:
        raise AnalysisError(
            "frame matrix shape does not match clips x steps",
            context={"shape": frame_matrix.shape, "expected": expected},
        )
    blocks = frame_matrix.reshape(n_clips, n_steps, n_clips, n_steps)
    if pair_set == "cross_clip_all":
        values = blocks.transpose(0, 2, 1, 3).reshape(n_clips, n_clips, n_steps * n_steps)
    elif pair_set == "cross_clip_aligned_step":
        steps = np.arange(n_steps)
        values = np.moveaxis(blocks[:, steps, :, steps], 0, -1)
    else:
        raise AnalysisError(
            f"pair set {pair_set!r} is not a frame-level pair set",
            context={"valid": ["cross_clip_all", "cross_clip_aligned_step"]},
        )
    valid = ~np.isnan(values)
    n_valid = valid.sum(axis=2)
    n_nan = values.shape[2] - n_valid
    total = np.where(valid, values, 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_valid > 0, total / n_valid, np.nan)
    return mean, n_valid.astype(np.int64), n_nan.astype(np.int64)


def centroid_dispersion(centroids: np.ndarray) -> float:
    """Mean distance of clip centroids from their common centroid, per ``sqrt(features)``."""
    array = as_matrix(centroids, name="centroids")
    center = array.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(array - center, axis=1) / np.sqrt(array.shape[1])
    return float(distances.mean())


def temporal_cosine(activations: np.ndarray, lag: int) -> np.ndarray:
    """Cosine between each step and the step ``lag`` later, within every clip.

    Returns ``(clips, steps - lag)``; zero vectors give NaN.
    """
    array = np.asarray(activations, dtype=np.float64)
    if array.ndim != 3:
        raise AnalysisError(
            "activations must be shaped (clips, steps, features)",
            context={"shape": tuple(array.shape)},
        )
    if lag < 1 or lag >= array.shape[1]:
        raise AnalysisError(
            "temporal lag must be between 1 and steps - 1",
            context={"lag": lag, "steps": array.shape[1]},
        )
    early = array[:, :-lag, :]
    late = array[:, lag:, :]
    numerator = np.einsum("ctf,ctf->ct", early, late)
    denominator = np.linalg.norm(early, axis=2) * np.linalg.norm(late, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(denominator < EPS, np.nan, numerator / np.where(denominator < EPS, 1.0, denominator))
    return np.clip(values, -1.0, 1.0)


def within_between_gap(within: float, between: float) -> float:
    """Mean same-scenario similarity minus mean different-scenario similarity."""
    return float(within - between)


# --------------------------------------------------------------------------- separation scores


@dataclass(frozen=True)
class SilhouetteResult:
    score: float
    per_label: dict[str, float]
    warnings: tuple[str, ...] = ()


def silhouette(
    samples: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None
) -> SilhouetteResult:
    """Silhouette score (Euclidean) of labelled samples, overall and per label.

    ``weights`` are per-sample multiplicities (bootstrap draw counts). A sample
    drawn ``w`` times counts ``w`` times as a neighbour of *other* samples and
    in the averages, but is never compared with its own copies: zero distances
    between duplicates of one clip are a resampling artefact, not evidence of a
    tight cluster. With unit weights this equals scikit-learn's
    ``silhouette_samples``.
    """
    array = as_matrix(samples, name="samples")
    label_array = np.asarray(labels).astype(str)
    unique = sorted(set(label_array.tolist()))
    undefined = SilhouetteResult(
        score=float("nan"),
        per_label={label: float("nan") for label in unique},
        warnings=("silhouette_undefined:needs_2_to_n_minus_1_labels",),
    )
    if len(unique) < 2 or len(unique) > array.shape[0] - 1:
        return undefined
    if weights is None:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            values = silhouette_samples(array, label_array, metric="euclidean")
        per_label = {label: float(values[label_array == label].mean()) for label in unique}
        return SilhouetteResult(score=float(values.mean()), per_label=per_label)

    weight = np.asarray(weights, dtype=np.float64).ravel()
    if weight.shape[0] != array.shape[0] or (weight < 0).any():
        raise AnalysisError(
            "silhouette weights must be non-negative with one entry per sample",
            context={"weights": int(weight.shape[0]), "samples": int(array.shape[0])},
        )
    present = [label for label in unique if weight[label_array == label].sum() > 0]
    if len(present) < 2:
        return undefined
    squared = (array**2).sum(axis=1)
    distances = np.sqrt(np.maximum(squared[:, None] + squared[None, :] - 2.0 * array @ array.T, 0.0))
    np.fill_diagonal(distances, 0.0)
    n_samples = array.shape[0]
    intra = np.full(n_samples, np.nan)
    nearest = np.full(n_samples, np.inf)
    for label in present:
        members = label_array == label
        member_weight = np.where(members, weight, 0.0)
        totals = distances @ member_weight  # weighted distance sums to this label (self term is 0)
        own_total = member_weight.sum() - np.where(members, weight, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            intra = np.where(members, np.where(own_total > 0, totals / own_total, np.nan), intra)
            other = totals / member_weight.sum()
        nearest = np.where(members, nearest, np.minimum(nearest, other))
    with np.errstate(invalid="ignore", divide="ignore"):
        values = (nearest - intra) / np.maximum(intra, nearest)
    # scikit-learn convention: a sample alone in its cluster scores 0.
    values = np.where(np.isnan(intra), 0.0, np.nan_to_num(values, nan=0.0))
    total = weight.sum()
    if total <= 0:
        return undefined
    per_label: dict[str, float] = {}
    for label in unique:
        members = label_array == label
        mass = weight[members].sum()
        per_label[label] = float((weight[members] * values[members]).sum() / mass) if mass > 0 else float("nan")
    return SilhouetteResult(score=float((weight * values).sum() / total), per_label=per_label)


@dataclass(frozen=True)
class NearestCentroidResult:
    accuracy: float
    labels: tuple[str, ...]
    confusion: np.ndarray  # rows: true label, columns: predicted label (counts)
    predicted: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)


def nearest_centroid_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    labels: tuple[str, ...] | list[str],
) -> NearestCentroidResult:
    """Fit a nearest-centroid classifier on train samples and score held-out samples."""
    label_order = tuple(str(label) for label in labels)
    confusion = np.zeros((len(label_order), len(label_order)), dtype=np.int64)
    train_labels = np.asarray(train_y).astype(str)
    test_labels = np.asarray(test_y).astype(str)
    notes: list[str] = []
    if train_labels.size == 0 or test_labels.size == 0:
        return NearestCentroidResult(
            accuracy=float("nan"),
            labels=label_order,
            confusion=confusion,
            warnings=("nearest_centroid_undefined:empty_train_or_test",),
        )
    if len(set(train_labels.tolist())) < 2:
        return NearestCentroidResult(
            accuracy=float("nan"),
            labels=label_order,
            confusion=confusion,
            warnings=("nearest_centroid_undefined:single_train_label",),
        )
    missing = sorted(set(label_order) - set(train_labels.tolist()))
    if missing:
        notes.append(f"labels_missing_from_train:{','.join(missing)}")
    model = NearestCentroid()
    model.fit(as_matrix(train_x, name="train_x"), train_labels)
    predicted = model.predict(as_matrix(test_x, name="test_x")).astype(str)
    index = {label: position for position, label in enumerate(label_order)}
    for true, pred in zip(test_labels.tolist(), predicted.tolist(), strict=True):
        if true in index and pred in index:
            confusion[index[true], index[pred]] += 1
    return NearestCentroidResult(
        accuracy=float((predicted == test_labels).mean()),
        labels=label_order,
        confusion=confusion,
        predicted=tuple(predicted.tolist()),
        warnings=tuple(notes),
    )
