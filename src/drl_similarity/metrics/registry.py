"""Metric registry: the twelve metric IDs a study may request, with their contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from drl_similarity.errors import AnalysisError
from drl_similarity.metrics import geometry, vector
from drl_similarity.registry import Registry


@dataclass(frozen=True)
class MetricDefinition:
    """Declared contract of one metric.

    ``kind``
        ``similarity`` (higher means more alike), ``distance`` (lower means more
        alike), or ``score`` (higher means better separation).
    ``value_range``
        Theoretical ``(low, high)`` bounds; ``None`` means unbounded.
    ``scopes``
        Analysis scopes the metric is reported in.
    ``requires_alignment``
        True when the metric compares two representations of identical ordered
        samples (cross-checkpoint geometry).
    ``supported_rank``
        Tensor rank of the array the function consumes.
    """

    id: str
    kind: str
    value_range: tuple[float | None, float | None]
    scopes: tuple[str, ...]
    requires_alignment: bool
    preprocessing: str
    supported_rank: int
    allowed_kwargs: tuple[str, ...]
    description: str
    function: Callable[..., Any]

    @property
    def direction(self) -> str:
        return {
            "similarity": "higher = more similar",
            "distance": "lower = more similar",
            "score": "higher = better separated",
        }[self.kind]

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "direction": self.direction,
            "range": list(self.value_range),
            "scopes": list(self.scopes),
            "requires_alignment": self.requires_alignment,
            "preprocessing": self.preprocessing,
            "supported_rank": self.supported_rank,
            "description": self.description,
        }


METRICS: Registry[MetricDefinition] = Registry("metric")

_PAIR_SCOPES = ("within_scenario", "between_scenario")
_GEOMETRY_KWARGS = ("sample_unit", "max_steps_per_clip")


def _register(definition: MetricDefinition) -> None:
    METRICS.register(definition.id, definition)


_register(
    MetricDefinition(
        id="cosine",
        kind="similarity",
        value_range=(-1.0, 1.0),
        scopes=_PAIR_SCOPES,
        requires_alignment=False,
        preprocessing="raw vectors; zero vectors reported as NaN",
        supported_rank=2,
        allowed_kwargs=(),
        description="Directional agreement between latent vectors of frames from different clips.",
        function=vector.cosine_similarity_matrix,
    )
)
_register(
    MetricDefinition(
        id="correlation",
        kind="similarity",
        value_range=(-1.0, 1.0),
        scopes=_PAIR_SCOPES,
        requires_alignment=False,
        preprocessing="each vector centered on its own mean (Pearson)",
        supported_rank=2,
        allowed_kwargs=(),
        description="Pearson correlation between latent vectors after removing each vector's mean.",
        function=vector.correlation_similarity_matrix,
    )
)
_register(
    MetricDefinition(
        id="normalized_euclidean",
        kind="distance",
        value_range=(0.0, None),
        scopes=_PAIR_SCOPES,
        requires_alignment=False,
        preprocessing="features z-scored on the declared clips; distance / sqrt(features)",
        supported_rank=2,
        allowed_kwargs=("standardize_on",),
        description="Magnitude-sensitive distance after feature standardization.",
        function=vector.normalized_euclidean_matrix,
    )
)
_register(
    MetricDefinition(
        id="centroid_dispersion",
        kind="distance",
        value_range=(0.0, None),
        scopes=("within_scenario",),
        requires_alignment=False,
        preprocessing="features z-scored on train clips; clip centroids; distance / sqrt(features)",
        supported_rank=2,
        allowed_kwargs=(),
        description="Mean distance of clip centroids from their scenario centroid.",
        function=vector.centroid_dispersion,
    )
)
_register(
    MetricDefinition(
        id="temporal_cosine",
        kind="similarity",
        value_range=(-1.0, 1.0),
        scopes=("temporal",),
        requires_alignment=False,
        preprocessing="raw vectors at fixed lags within a clip; reported with input-pixel control",
        supported_rank=3,
        allowed_kwargs=("lags",),
        description="Cosine similarity between steps a fixed lag apart inside one clip.",
        function=vector.temporal_cosine,
    )
)
_register(
    MetricDefinition(
        id="within_between_gap",
        kind="score",
        value_range=(-2.0, 2.0),
        scopes=("separation",),
        requires_alignment=False,
        preprocessing="base similarity (default cosine) over matched pair definitions",
        supported_rank=2,
        allowed_kwargs=("base_metric",),
        description="Mean same-scenario similarity minus mean different-scenario similarity.",
        function=vector.within_between_gap,
    )
)
_register(
    MetricDefinition(
        id="silhouette",
        kind="score",
        value_range=(-1.0, 1.0),
        scopes=("separation",),
        requires_alignment=False,
        preprocessing="clip centroids of train-z-scored features; Euclidean",
        supported_rank=2,
        allowed_kwargs=(),
        description="Scenario compactness relative to separation, computed on clip centroids.",
        function=vector.silhouette,
    )
)
_register(
    MetricDefinition(
        id="nearest_centroid_accuracy",
        kind="score",
        value_range=(0.0, 1.0),
        scopes=("separation", "confusion"),
        requires_alignment=False,
        preprocessing="normalization and class centroids fitted on train clips only",
        supported_rank=2,
        allowed_kwargs=(),
        description="Held-out scenario prediction from clip centroids with a nearest-centroid rule.",
        function=vector.nearest_centroid_accuracy,
    )
)
_register(
    MetricDefinition(
        id="linear_cka",
        kind="similarity",
        value_range=(0.0, 1.0),
        scopes=("cross_checkpoint", *_PAIR_SCOPES),
        requires_alignment=True,
        preprocessing="columns centered over samples; Gram-matrix alignment",
        supported_rank=2,
        allowed_kwargs=_GEOMETRY_KWARGS,
        description="Similarity of sample geometry; invariant to rotation and isotropic scaling.",
        function=geometry.linear_cka,
    )
)
_register(
    MetricDefinition(
        id="rsa_spearman",
        kind="similarity",
        value_range=(-1.0, 1.0),
        scopes=("cross_checkpoint",),
        requires_alignment=True,
        preprocessing="first-order RDM (configured distance); Spearman of upper triangles",
        supported_rank=2,
        allowed_kwargs=("first_order_distance", *_GEOMETRY_KWARGS),
        description="Correlation between representational dissimilarity matrices.",
        function=geometry.rsa_spearman,
    )
)
_register(
    MetricDefinition(
        id="svcca",
        kind="similarity",
        value_range=(0.0, 1.0),
        scopes=("cross_checkpoint",),
        requires_alignment=True,
        preprocessing="centered; SVD to configured retained variance; mean canonical correlation",
        supported_rank=2,
        allowed_kwargs=("variance_retained", *_GEOMETRY_KWARGS),
        description="Similarity between dominant representational subspaces.",
        function=geometry.svcca,
    )
)
_register(
    MetricDefinition(
        id="orthogonal_procrustes",
        kind="distance",
        value_range=(0.0, 1.0),
        scopes=("cross_checkpoint",),
        requires_alignment=True,
        preprocessing="centered, unit Frobenius norm; residual after best orthogonal alignment",
        supported_rank=2,
        allowed_kwargs=_GEOMETRY_KWARGS,
        description="Normalized residual distance after optimal rotation; needs equal widths.",
        function=geometry.orthogonal_procrustes,
    )
)


def get_metric(metric_id: str) -> MetricDefinition:
    return METRICS.get(metric_id, field="analysis.metrics.id")


def list_metric_ids() -> list[str]:
    return METRICS.ids()


def describe_metrics() -> list[dict[str, Any]]:
    return [METRICS.get(metric_id).describe() for metric_id in METRICS.ids()]


def validate_metric_kwargs(metric_id: str, kwargs: dict[str, Any]) -> None:
    """Reject kwargs a metric does not declare (configuration typos must not pass silently)."""
    definition = get_metric(metric_id)
    unknown = sorted(set(kwargs) - set(definition.allowed_kwargs))
    if unknown:
        raise AnalysisError(
            f"Unknown kwargs for metric {metric_id!r}",
            context={"unknown": unknown, "allowed": list(definition.allowed_kwargs)},
        )
