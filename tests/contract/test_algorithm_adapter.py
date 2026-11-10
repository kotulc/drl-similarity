"""Contract every registered algorithm adapter must satisfy (runs for DQN and PPO)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch as th

from drl_similarity.agents.registry import ALGORITHM_ADAPTERS, get_algorithm_adapter
from drl_similarity.config import AlgorithmConfig, EnvironmentConfig
from drl_similarity.environments.atari import make_environment
from drl_similarity.errors import ConfigError, ExtractionError, RegistryError
from tests.conftest import ALGORITHM_KWARGS


@pytest.fixture
def loaded(algorithm_config: AlgorithmConfig, environment_config: EnvironmentConfig, tmp_path: Path):
    """Build, briefly train, save, and reload an agent without an environment."""
    adapter = get_algorithm_adapter(algorithm_config.adapter)
    env = make_environment(environment_config, seed=0)
    try:
        agent = adapter.build(algorithm_config, env, seed=0, device="cpu")
        agent.learn(48)
        agent.save(tmp_path / "checkpoint.zip")
    finally:
        env.close()

    return adapter.load(tmp_path / "checkpoint.zip", algorithm_config, None, device="cpu")


@pytest.fixture
def observations() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (6, 4, 36, 36), dtype=np.uint8)


def test_registry_covers_test_adapters() -> None:
    """Every registered adapter is exercised by this suite."""
    assert ALGORITHM_ADAPTERS.ids() == sorted(ALGORITHM_KWARGS)


def test_registry_unknown_id_lists_valid() -> None:
    with pytest.raises(RegistryError, match="sb3.dqn"):
        get_algorithm_adapter("sb3.nope")


def test_predict_is_deterministic(loaded, observations: np.ndarray) -> None:
    first = loaded.predict(observations)

    assert first.shape == (len(observations),)
    assert np.array_equal(first, loaded.predict(observations))


def test_extract_every_declared_layer(loaded, algorithm_config: AlgorithmConfig, observations: np.ndarray) -> None:
    """Declared capabilities match what the loaded agent can extract, in sample order."""
    declared = tuple(algorithm_config.capabilities.activation_layers)
    assert set(declared) == set(loaded.layer_names())

    activations = loaded.extract(observations, layers=declared)

    assert tuple(activations) == declared
    assert all(a.shape[0] == len(observations) and np.isfinite(a).all() for a in activations.values())
    assert activations["policy_features"].shape == (len(observations), 512)


def test_extract_batched_equals_unbatched(loaded, observations: np.ndarray) -> None:
    layers = loaded.layer_names()
    batched = loaded.extract(observations, layers=layers)
    single = [loaded.extract(observation[None], layers=layers) for observation in observations]

    for name in layers:
        stacked = np.concatenate([item[name] for item in single])
        np.testing.assert_allclose(batched[name], stacked, rtol=1e-4, atol=1e-5)


def test_extract_does_not_mutate(loaded, observations: np.ndarray) -> None:
    """Hooks are passive: parameters and predictions are unchanged and hooks are removed."""
    before = [parameter.clone() for parameter in loaded.model.policy.parameters()]
    actions = loaded.predict(observations)

    loaded.extract(observations, layers=loaded.layer_names())

    after = list(loaded.model.policy.parameters())
    assert all(th.equal(a, b) for a, b in zip(before, after, strict=True))
    assert np.array_equal(actions, loaded.predict(observations))
    assert not any(module._forward_hooks for module in loaded.model.policy.modules())


def test_extract_unknown_layer_rejected(loaded, observations: np.ndarray) -> None:
    with pytest.raises(ExtractionError, match="Unknown layers"):
        loaded.extract(observations, layers=("not_a_layer",))


def test_validate_rejects_unknown_kwargs(algorithm_config: AlgorithmConfig, environment_config) -> None:
    adapter = get_algorithm_adapter(algorithm_config.adapter)
    bad = algorithm_config.model_copy(update={"model_kwargs": {"not_a_kwarg": 1}})
    env = make_environment(environment_config, seed=0)
    try:
        with pytest.raises(ConfigError, match="not_a_kwarg"):
            adapter.validate(bad, env.observation_space, env.action_space)
    finally:
        env.close()
