"""Environment adapter protocols and fingerprint types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import gymnasium as gym


@dataclass(frozen=True)
class EnvironmentDescription:
    env_id: str
    observation_shape: tuple[int, ...]
    observation_dtype: str
    action_space_type: str
    action_n: int | None
    effective_frame_skip: int
    wrappers: tuple[dict[str, Any], ...]
    make_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentFingerprints:
    observation: str
    runtime: str
    description: EnvironmentDescription


class EnvironmentAdapter(Protocol):
    def make(self, *, seed: int, render_mode: str | None = None) -> gym.Env: ...

    def fingerprint(self) -> str: ...

    def describe(self) -> dict[str, object]: ...
