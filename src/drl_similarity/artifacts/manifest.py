"""Versioned artifact manifests and checksums."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from drl_similarity.errors import ArtifactError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactInput(StrictModel):
    path: str
    sha256: str
    role: str


class ArtifactOutput(StrictModel):
    path: str
    sha256: str
    role: str
    schema_version: int = 1
    bytes: int = 0


class ArtifactManifest(StrictModel):
    schema_version: int = 1
    stage: str
    artifact_id: str
    config_hash: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)
    inputs: list[ArtifactInput] = Field(default_factory=list)
    outputs: list[ArtifactOutput] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> ArtifactManifest:
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(f"Invalid artifact manifest: {path}", context={"error": str(exc)}) from exc


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_dir(path: Path) -> ArtifactManifest:
    """Re-hash every declared input and output; paths are relative to the artifact directory."""
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise ArtifactError(f"Missing manifest.json in {path}")
    manifest = ArtifactManifest.read(manifest_path)

    for item in [*manifest.inputs, *manifest.outputs]:
        file_path = path / item.path
        if not file_path.exists():
            raise ArtifactError(
                f"Missing file declared in manifest: {item.path}", context={"artifact_dir": str(path)}
            )
        actual = sha256_file(file_path)
        if actual != item.sha256:
            raise ArtifactError(
                f"Checksum mismatch for {item.path}",
                context={"artifact_dir": str(path), "expected": item.sha256, "actual": actual},
            )

    return manifest


def capture_package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:  # noqa: BLE001
            versions[name] = "missing"
    return versions


def try_git_commit(repo: Path) -> str | None:
    """Current commit hash of ``repo``, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()
