"""Algorithm adapter protocols shared by every training backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from drl_similarity.config import AlgorithmConfig


@dataclass(frozen=True)
class AdapterCapabilities:
    """Framework-neutral adapter metadata stored with run artifacts."""

    adapter_id: str
    recurrent: bool
    action_spaces: tuple[str, ...]
    activation_layers: tuple[str, ...]
    backend_version: str


class LoadedAgent(Protocol):
    """Inference/extraction surface returned after loading a checkpoint."""

    recurrent: bool

    def extract(
        self,
        observations: np.ndarray,
        *,
        layers: tuple[str, ...],
        episode_starts: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]: ...

    def layer_names(self) -> tuple[str, ...]: ...

    def predict(self, observations: np.ndarray, *, deterministic: bool = True) -> np.ndarray: ...


class TrainableAgent(Protocol):
    """Training surface; internal until a second backend proves what is shared."""

    @property
    def num_timesteps(self) -> int: ...

    def learn(self, total_timesteps: int, *, callback: Any = None, progress_bar: bool = False) -> None: ...

    def loaded(self) -> LoadedAgent: ...

    def save(self, path: Path) -> None: ...


class AlgorithmAdapter(Protocol):
    """Owns every training-framework difference behind a stable YAML ID."""

    adapter_id: str

    def build(
        self,
        algorithm: AlgorithmConfig,
        env: gym.Env,
        *,
        seed: int,
        device: str,
        resume: Path | None = None,
    ) -> TrainableAgent: ...

    def capabilities(self, algorithm: AlgorithmConfig) -> AdapterCapabilities: ...

    def load(
        self, checkpoint: Path, algorithm: AlgorithmConfig, env: gym.Env | None, *, device: str
    ) -> LoadedAgent: ...

    def validate(
        self, algorithm: AlgorithmConfig, observation_space: gym.Space, action_space: gym.Space
    ) -> None: ...
