"""Algorithm adapter registry: stable YAML IDs -> lazily imported adapter factories."""

from __future__ import annotations

from collections.abc import Callable

from drl_similarity.agents.base import AlgorithmAdapter
from drl_similarity.registry import Registry


def make_sb3_dqn_adapter() -> AlgorithmAdapter:
    from drl_similarity.agents.sb3.dqn import Sb3DqnAdapter

    return Sb3DqnAdapter()


def make_sb3_ppo_adapter() -> AlgorithmAdapter:
    from drl_similarity.agents.sb3.ppo import Sb3PpoAdapter

    return Sb3PpoAdapter()


# Backends import inside the factories so config validation stays light
ALGORITHM_ADAPTERS: Registry[Callable[[], AlgorithmAdapter]] = Registry("algorithm adapter")
ALGORITHM_ADAPTERS.register("sb3.dqn", make_sb3_dqn_adapter)
ALGORITHM_ADAPTERS.register("sb3.ppo", make_sb3_ppo_adapter)


def get_algorithm_adapter(adapter_id: str) -> AlgorithmAdapter:
    """Instantiate a registered adapter; unknown IDs list the valid ones."""
    return ALGORITHM_ADAPTERS.get(adapter_id, field="algorithm.adapter")()
