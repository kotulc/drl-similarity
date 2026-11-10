"""Explicit wrapper registry for environment construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, RecordEpisodeStatistics, ReshapeObservation

from drl_similarity.config import WrapperSpec
from drl_similarity.errors import EnvironmentError
from drl_similarity.registry import Registry


WrapperFactory = Callable[[gym.Env, dict[str, Any]], gym.Env]
WRAPPER_REGISTRY: Registry[WrapperFactory] = Registry("wrapper")

# Wrapper kwarg that repeats actions, keyed by wrapper id
FRAME_SKIP_KWARGS: dict[str, tuple[str, int]] = {"sb3_atari": ("frame_skip", 4)}


def _frame_stack(env: gym.Env, kwargs: dict[str, Any]) -> gym.Env:
    """Stack the last ``num_stack`` observations on a new leading axis."""
    return FrameStackObservation(env, stack_size=int(kwargs.get("num_stack", 4)))


def _record_episode_statistics(env: gym.Env, kwargs: dict[str, Any]) -> gym.Env:
    """Record raw episode return and length in ``info['episode']``."""
    return RecordEpisodeStatistics(env, **kwargs)


def _sb3_atari(env: gym.Env, kwargs: dict[str, Any]) -> gym.Env:
    """Apply SB3's AtariWrapper and drop its trailing channel axis: (H, W, 1) -> (H, W)."""
    from stable_baselines3.common.atari_wrappers import AtariWrapper

    env = AtariWrapper(env, **kwargs)
    return ReshapeObservation(env, env.observation_space.shape[:-1])


WRAPPER_REGISTRY.register("frame_stack", _frame_stack)
WRAPPER_REGISTRY.register("record_episode_statistics", _record_episode_statistics)
WRAPPER_REGISTRY.register("sb3_atari", _sb3_atari)


def apply_wrappers(
    env: gym.Env, specs: list[WrapperSpec], *, base_frameskip: int
) -> tuple[gym.Env, int]:
    """Apply wrappers in YAML order; return the env and its effective frame skip."""
    effective = int(base_frameskip)

    for spec in specs:
        factory = WRAPPER_REGISTRY.get(spec.id, field="wrappers.id")

        # Only one component may repeat actions
        if spec.id in FRAME_SKIP_KWARGS:
            key, default = FRAME_SKIP_KWARGS[spec.id]
            skip = int(spec.kwargs.get(key, default))
            if skip > 1 and effective > 1:
                raise EnvironmentError(
                    "Double frame skip: more than one component repeats actions",
                    context={"already": effective, "wrapper": spec.id, "wrapper_skip": skip},
                )
            effective *= skip

        try:
            env = factory(env, dict(spec.kwargs))
        except TypeError as exc:
            raise EnvironmentError(
                f"Invalid kwargs for wrapper {spec.id!r}: {exc}", context={"kwargs": spec.kwargs}
            ) from exc

    return env, effective
