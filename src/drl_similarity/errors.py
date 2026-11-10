"""Domain error hierarchy for the DRL similarity package."""

from __future__ import annotations

from typing import Any


class DrlSimilarityError(Exception):
    """Base error with an actionable message and optional context."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} ({details})"


class ConfigError(DrlSimilarityError):
    """Invalid, missing, or incompatible configuration."""


class RegistryError(DrlSimilarityError):
    """Unknown or duplicate registry identifier."""


class ArtifactError(DrlSimilarityError):
    """Artifact store, checksum, or lineage failure."""


class EnvironmentError(DrlSimilarityError):  # noqa: A001 — domain name matches plan
    """Environment construction or fingerprint failure."""


class CheckpointError(DrlSimilarityError):
    """Checkpoint save, load, or selection failure."""


class ScenarioError(DrlSimilarityError):
    """Scenario collection, curation, or validation failure."""


class ExtractionError(DrlSimilarityError):
    """Activation extraction failure."""


class AnalysisError(DrlSimilarityError):
    """Metric analysis failure."""
