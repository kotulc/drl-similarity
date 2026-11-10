"""Scenario manifests, candidate collection, and validation.

Only the lightweight schema is imported here; ``collect`` loads a training
backend and is imported explicitly by the CLI.
"""

from drl_similarity.scenarios.schema import (
    ClipRecord,
    CollectorInfo,
    ScenarioManifest,
    dump_manifest,
    load_manifest,
    observation_sha256,
)


__all__ = [
    "ClipRecord",
    "CollectorInfo",
    "ScenarioManifest",
    "dump_manifest",
    "load_manifest",
    "observation_sha256",
]
