"""Shared fixtures: a tiny deterministic pixel environment and configs that use it (no Atari)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import yaml

from drl_similarity.config import AlgorithmConfig, EnvironmentConfig


FAKE_ENV_ID = "DrlSimFakePixels-v0"
EPISODE_STEPS = 40

ALGORITHM_KWARGS: dict[str, dict[str, Any]] = {
    "sb3.dqn": {"buffer_size": 500, "learning_starts": 20, "batch_size": 8, "train_freq": 4},
    "sb3.ppo": {"n_steps": 16, "batch_size": 8, "n_epochs": 1},
}
ALGORITHM_LAYERS: dict[str, list[str]] = {
    "sb3.dqn": ["conv1", "conv2", "conv3", "conv_flat", "policy_features", "q_values"],
    "sb3.ppo": [
        "conv1", "conv2", "conv3", "conv_flat", "policy_features",
        "policy_latent", "value_latent", "action_logits", "value",
    ],
}  # fmt: skip


class FakePixelEnv(gym.Env):
    """Seeded noise frames; a reward every 5 steps, a life lost every 13, 3 lives, 40 steps."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        self.observation_space = gym.spaces.Box(0, 255, (36, 36), np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.render_mode = render_mode
        self._step = 0

    def _frame(self) -> np.ndarray:
        return self.np_random.integers(0, 256, (36, 36), dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step = 0
        return self._frame(), {"lives": 3}

    def step(self, action):
        self._step += 1
        reward = float(self._step % 5 == 0)
        lives = 3 - self._step // 13
        return self._frame(), reward, self._step >= EPISODE_STEPS, False, {"lives": lives}

    def render(self):
        return np.zeros((36, 36, 3), np.uint8)


if FAKE_ENV_ID not in gym.registry:
    gym.register(FAKE_ENV_ID, entry_point=FakePixelEnv)


def environment_document() -> dict[str, Any]:
    """Environment YAML mapping for the fake env with a 4-frame stack."""
    return {
        "schema_version": 1,
        "kind": "environment",
        "id": "fake-pixels-v1",
        "env_id": FAKE_ENV_ID,
        "wrappers": [
            {"id": "record_episode_statistics"},
            {"id": "frame_stack", "kwargs": {"num_stack": 4}},
        ],
        "expected": {
            "observation_shape": [4, 36, 36],
            "observation_dtype": "uint8",
            "action_space": "discrete",
            "effective_frame_skip": 1,
        },
    }


def algorithm_document(adapter_id: str) -> dict[str, Any]:
    """Algorithm YAML mapping with a tiny budget for ``adapter_id``."""
    return {
        "schema_version": 1,
        "kind": "algorithm",
        "id": f"tiny-{adapter_id}",
        "adapter": adapter_id,
        "policy": "CnnPolicy",
        "policy_kwargs": {"features_extractor": "nature_cnn"},
        "model_kwargs": ALGORITHM_KWARGS[adapter_id],
        "capabilities": {"action_spaces": ["discrete"], "activation_layers": ALGORITHM_LAYERS[adapter_id]},
    }


def write_yaml(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def environment_config() -> EnvironmentConfig:
    return EnvironmentConfig.model_validate(environment_document())


@pytest.fixture(params=sorted(ALGORITHM_KWARGS))
def algorithm_config(request: pytest.FixtureRequest) -> AlgorithmConfig:
    """One tiny algorithm config per registered adapter."""
    return AlgorithmConfig.model_validate(algorithm_document(request.param))
