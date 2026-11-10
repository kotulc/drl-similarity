"""Atomic artifact directory writer."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from drl_similarity.artifacts.manifest import (
    ArtifactManifest,
    ArtifactOutput,
    sha256_file,
)
from drl_similarity.errors import ArtifactError


class ArtifactStore:
    """Create versioned artifact directories with atomic publish."""

    def __init__(
        self,
        root: Path,
        *,
        artifact_id: str,
        config_hash: str,
        stage: str,
        collision: str = "error",
    ) -> None:
        self.root = Path(root)
        self.artifact_id = artifact_id
        self.config_hash = config_hash
        self.stage = stage
        self.collision = collision
        self.target = self.root / artifact_id

    def resolve_target(self, *, force: bool = False) -> Path:
        if not self.target.exists():
            return self.target
        if not force and self.collision == "error":
            raise ArtifactError(
                f"Artifact already exists: {self.target}",
                context={"collision": self.collision},
            )
        # force or version collision: create new version suffix
        version = 2
        while True:
            candidate = self.root / f"{self.artifact_id}-v{version}"
            if not candidate.exists():
                return candidate
            version += 1

    def publish(
        self,
        writer: Callable[[Path], ArtifactManifest],
        *,
        force: bool = False,
    ) -> Path:
        target = self.resolve_target(force=force)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.tmp.",
                dir=str(target.parent),
            )
        )
        try:
            manifest = writer(tmp)
            # Ensure outputs have checksums relative to tmp
            fixed_outputs: list[ArtifactOutput] = []
            for item in manifest.outputs:
                path = tmp / item.path
                if not path.exists():
                    raise ArtifactError(f"Writer did not create declared output {item.path}")
                fixed_outputs.append(
                    ArtifactOutput(
                        path=item.path,
                        sha256=sha256_file(path),
                        role=item.role,
                        schema_version=item.schema_version,
                        bytes=path.stat().st_size,
                    )
                )
            manifest = manifest.model_copy(update={"outputs": fixed_outputs})
            manifest.write(tmp / "manifest.json")
            os.replace(tmp, target)
            return target
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def build_manifest(
        self,
        *,
        inputs: list[Any] | None = None,
        outputs: list[ArtifactOutput] | None = None,
        metadata: dict[str, Any] | None = None,
        git_commit: str | None = None,
        package_versions: dict[str, str] | None = None,
    ) -> ArtifactManifest:
        return ArtifactManifest(
            stage=self.stage,
            artifact_id=self.artifact_id,
            config_hash=self.config_hash,
            git_commit=git_commit,
            package_versions=package_versions or {},
            inputs=inputs or [],
            outputs=outputs or [],
            metadata=metadata or {},
        )
