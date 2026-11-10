"""Shared Stable-Baselines3 adapter: build, load, predict, and hook-based extraction.

Concrete adapters (``dqn.py``, ``ppo.py``) only declare the SB3 class, the allowed
``model_kwargs``, and the mapping from logical layer names to modules/outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import NatureCNN

from drl_similarity.agents.base import AdapterCapabilities
from drl_similarity.config import AlgorithmConfig
from drl_similarity.errors import CheckpointError, ConfigError, ExtractionError


# Friendly YAML IDs -> SB3 classes (YAML never contains import paths)
FEATURES_EXTRACTORS: dict[str, type] = {"nature_cnn": NatureCNN}
POLICIES: tuple[str, ...] = ("CnnPolicy",)

# name -> tensor computed from the policy's own forward pass
Outputs = dict[str, th.Tensor]


@contextmanager
def capture(modules: dict[str, th.nn.Module]) -> Iterator[Outputs]:
    """Record each module's forward output by name; hooks are always removed."""
    captured: Outputs = {}
    handles = [
        module.register_forward_hook(lambda _m, _i, out, name=name: captured.__setitem__(name, out.detach()))
        for name, module in modules.items()
    ]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def nature_cnn_modules(extractor: NatureCNN) -> dict[str, th.nn.Module]:
    """Logical names for NatureCNN: post-ReLU conv maps, flattened map, 512-d features."""
    relus = [module for module in extractor.cnn if isinstance(module, th.nn.ReLU)]
    flatten = [module for module in extractor.cnn if isinstance(module, th.nn.Flatten)]
    if len(relus) != 3 or len(flatten) != 1:
        raise CheckpointError("Expected a NatureCNN extractor with three conv blocks")

    return {
        "conv1": relus[0],
        "conv2": relus[1],
        "conv3": relus[2],
        "conv_flat": flatten[0],
        "policy_features": extractor,
    }


class Sb3LoadedAgent:
    """Inference and extraction surface over a loaded SB3 model."""

    recurrent = False

    def __init__(
        self,
        model: BaseAlgorithm,
        modules: dict[str, th.nn.Module],
        forward: Callable[[BaseAlgorithm, th.Tensor], Outputs],
        output_names: tuple[str, ...],
    ) -> None:
        self.model = model
        self._modules = modules
        self._forward = forward
        self._names = (*modules, *output_names)
        self.model.policy.set_training_mode(False)

    def extract(
        self,
        observations: np.ndarray,
        *,
        layers: tuple[str, ...],
        episode_starts: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Activations for a batch of observations, in sample order (feed-forward)."""
        unknown = sorted(set(layers) - set(self._names))
        if unknown:
            raise ExtractionError(f"Unknown layers {unknown}", context={"valid": list(self._names)})

        tensor, _ = self.model.policy.obs_to_tensor(np.asarray(observations))
        with th.no_grad(), capture(self._modules) as captured:
            outputs = {**self._forward(self.model, tensor), **captured}

        return {name: outputs[name].cpu().numpy() for name in layers}

    def layer_names(self) -> tuple[str, ...]:
        """Every logical layer this agent can extract."""
        return self._names

    def predict(self, observations: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        """Actions for one observation or a batch."""
        actions, _ = self.model.predict(np.asarray(observations), deterministic=deterministic)
        return np.asarray(actions)


class Sb3Trainable:
    """Training surface over an SB3 model."""

    def __init__(self, model: BaseAlgorithm, loaded: Callable[[BaseAlgorithm], Sb3LoadedAgent]) -> None:
        self.model = model
        self._loaded = loaded

    @property
    def num_timesteps(self) -> int:
        return int(self.model.num_timesteps)

    def learn(self, total_timesteps: int, *, callback: Any = None, progress_bar: bool = False) -> None:
        """Train until ``num_timesteps`` reaches ``total_timesteps`` (resume-aware)."""
        remaining = total_timesteps - self.num_timesteps
        if remaining > 0:
            self.model.learn(
                remaining, callback=callback, progress_bar=progress_bar, reset_num_timesteps=False
            )

    def loaded(self) -> Sb3LoadedAgent:
        """Inference view of the model being trained."""
        return self._loaded(self.model)

    def save(self, path: Path) -> None:
        """Write an SB3 ``.zip`` checkpoint at exactly ``path``."""
        self.model.save(path)


class Sb3Adapter:
    """Base for SB3 algorithm adapters; subclasses fill in the class attributes."""

    adapter_id: ClassVar[str]
    model_class: ClassVar[type[BaseAlgorithm]]
    allowed_kwargs: ClassVar[frozenset[str]]
    action_spaces: ClassVar[tuple[type[gym.Space], ...]]
    output_names: ClassVar[tuple[str, ...]]

    def build(
        self,
        algorithm: AlgorithmConfig,
        env: gym.Env,
        *,
        seed: int,
        device: str,
        resume: Path | None = None,
    ) -> Sb3Trainable:
        """New model from YAML, or a checkpoint re-attached to ``env`` when resuming."""
        self.validate(algorithm, env.observation_space, env.action_space)

        if resume is not None:
            model = self._load_model(resume, env, device)
        else:
            policy_kwargs = dict(algorithm.policy_kwargs)
            extractor = policy_kwargs.pop("features_extractor", "nature_cnn")
            policy_kwargs["features_extractor_class"] = FEATURES_EXTRACTORS[extractor]
            model = self.model_class(
                algorithm.policy,
                env,
                seed=seed,
                device=device,
                policy_kwargs=policy_kwargs,
                **algorithm.model_kwargs,
            )

        return Sb3Trainable(model, self._wrap)

    def capabilities(self, algorithm: AlgorithmConfig) -> AdapterCapabilities:
        """Framework-neutral description of this adapter."""
        return AdapterCapabilities(
            adapter_id=self.adapter_id,
            recurrent=False,
            action_spaces=tuple(algorithm.capabilities.action_spaces),
            activation_layers=tuple(algorithm.capabilities.activation_layers),
            backend_version=stable_baselines3.__version__,
        )

    def load(
        self, checkpoint: Path, algorithm: AlgorithmConfig, env: gym.Env | None, *, device: str
    ) -> Sb3LoadedAgent:
        """Load a checkpoint for inference/extraction; no environment is required."""
        return self._wrap(self._load_model(checkpoint, env, device))

    def validate(
        self, algorithm: AlgorithmConfig, observation_space: gym.Space, action_space: gym.Space
    ) -> None:
        """Reject unsupported spaces, policies, extractors, and backend kwargs."""
        if not isinstance(action_space, self.action_spaces):
            raise ConfigError(
                f"{self.adapter_id} does not support action space {type(action_space).__name__}"
            )
        if len(observation_space.shape or ()) != 3:
            raise ConfigError(
                f"{self.adapter_id} with CnnPolicy needs image observations (C, H, W)",
                context={"observation_shape": observation_space.shape},
            )
        if algorithm.policy not in POLICIES:
            raise ConfigError(f"Unsupported policy {algorithm.policy!r}", context={"valid": POLICIES})

        extractor = algorithm.policy_kwargs.get("features_extractor", "nature_cnn")
        if extractor not in FEATURES_EXTRACTORS:
            raise ConfigError(
                f"Unknown features_extractor {extractor!r}", context={"valid": sorted(FEATURES_EXTRACTORS)}
            )

        unknown = sorted(set(algorithm.model_kwargs) - self.allowed_kwargs)
        if unknown:
            raise ConfigError(
                f"Unknown {self.adapter_id} model_kwargs {unknown}",
                context={"allowed": sorted(self.allowed_kwargs)},
            )

    def _forward(self, model: BaseAlgorithm, observations: th.Tensor) -> Outputs:
        """Run the policy once and return functional outputs named in ``output_names``."""
        raise NotImplementedError

    def _load_model(self, checkpoint: Path, env: gym.Env | None, device: str) -> BaseAlgorithm:
        try:
            return self.model_class.load(checkpoint, env=env, device=device)
        except Exception as exc:  # noqa: BLE001 - SB3 raises many error types on bad zips
            raise CheckpointError(f"Failed to load {self.adapter_id} checkpoint {checkpoint}: {exc}") from exc

    def _wrap(self, model: BaseAlgorithm) -> Sb3LoadedAgent:
        modules = nature_cnn_modules(model.policy.features_extractor)
        return Sb3LoadedAgent(model, modules, self._forward, self.output_names)
