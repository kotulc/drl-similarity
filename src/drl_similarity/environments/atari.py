"""Gymnasium (Atari) environment construction, inspection, and fingerprinting."""

from __future__ import annotations

import hashlib
import json
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import gymnasium as gym
import numpy as np

from drl_similarity.config import EnvironmentConfig
from drl_similarity.environments.base import EnvironmentDescription, EnvironmentFingerprints
from drl_similarity.environments.wrappers import apply_wrappers
from drl_similarity.errors import EnvironmentError


# make_kwargs and wrappers that change what the model sees
OBSERVATION_MAKE_KWARGS = ("obs_type", "frameskip", "full_action_space")
OBSERVATION_WRAPPERS = ("sb3_atari", "frame_stack")
OBSERVATION_WRAPPER_KWARGS = ("frame_skip", "screen_size", "num_stack")


def assert_expected_environment(env: gym.Env, config: EnvironmentConfig) -> None:
    """Fail when the constructed spaces differ from the YAML ``expected`` block."""
    expected = config.expected
    got = {
        "observation_shape": list(env.observation_space.shape or ()),
        "observation_dtype": str(env.observation_space.dtype),
        "action_space": type(env.action_space).__name__.lower(),
    }
    want = {key: getattr(expected, key) for key in got}

    if got != want:
        raise EnvironmentError("Environment does not match expected", context={"got": got, "expected": want})


def compute_environment_fingerprints(config: EnvironmentConfig) -> EnvironmentFingerprints:
    """Hash canonical descriptors: observation-only and full runtime fingerprints."""
    env = make_environment(config, seed=0)
    try:
        action_space = str(env.action_space)
        action_n = int(getattr(env.action_space, "n", 0)) or None
    finally:
        env.close()

    wrappers = tuple({"id": w.id, "kwargs": dict(sorted(w.kwargs.items()))} for w in config.wrappers)
    description = EnvironmentDescription(
        env_id=config.env_id,
        observation_shape=tuple(config.expected.observation_shape),
        observation_dtype=config.expected.observation_dtype,
        action_space_type=config.expected.action_space,
        action_n=action_n,
        effective_frame_skip=config.expected.effective_frame_skip,
        wrappers=wrappers,
        make_kwargs=dict(sorted(config.make_kwargs.items())),
    )

    # Reward-only settings (clip_reward, noop_max, ...) are excluded from the observation hash
    observation = {
        "env_id": config.env_id,
        "observation_shape": description.observation_shape,
        "observation_dtype": description.observation_dtype,
        "make_kwargs": {k: v for k, v in description.make_kwargs.items() if k in OBSERVATION_MAKE_KWARGS},
        "wrappers": [
            {"id": w["id"], "kwargs": {k: v for k, v in w["kwargs"].items() if k in OBSERVATION_WRAPPER_KWARGS}}
            for w in wrappers
            if w["id"] in OBSERVATION_WRAPPERS
        ],
    }
    runtime = {
        "description": vars(description),
        "action_space": action_space,
        "versions": {name: _package_version(name) for name in ("ale-py", "gymnasium")},
    }

    return EnvironmentFingerprints(
        observation=_stable_hash(observation), runtime=_stable_hash(runtime), description=description
    )


def inspect_environment(env: gym.Env) -> dict[str, Any]:
    """Reset and take one legal step; report what came back."""
    observation, _ = env.reset()
    next_observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    return {
        "observation_shape": tuple(np.asarray(observation).shape),
        "observation_dtype": str(np.asarray(observation).dtype),
        "next_observation_shape": tuple(np.asarray(next_observation).shape),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "action_n": getattr(env.action_space, "n", None),
        "info_keys": sorted(info),
    }


def make_environment(config: EnvironmentConfig, seed: int, render_mode: str | None = None) -> gym.Env:
    """Build the wrapped environment, seed it, and assert the expected spaces."""
    register_environment_modules(config.registration_modules)
    make_kwargs = dict(config.make_kwargs)
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    try:
        env = gym.make(config.env_id, **make_kwargs)
    except Exception as exc:  # noqa: BLE001 - gymnasium raises many error types here
        raise EnvironmentError(
            f"gymnasium.make failed for {config.env_id}: {exc}", context={"make_kwargs": make_kwargs}
        ) from exc

    try:
        env, effective = apply_wrappers(
            env, list(config.wrappers), base_frameskip=int(make_kwargs.get("frameskip", 1))
        )
        if effective != config.expected.effective_frame_skip:
            raise EnvironmentError(
                "Effective frame skip does not match expected",
                context={"effective": effective, "expected": config.expected.effective_frame_skip},
            )
        assert_expected_environment(env, config)
        env.action_space.seed(seed)
        env.reset(seed=seed)
    except Exception:
        env.close()
        raise

    return env


def register_environment_modules(names: list[str]) -> None:
    """Import modules that register environment IDs (for example ``ale_py``)."""
    for name in names:
        try:
            module = import_module(name)
        except ImportError as exc:
            raise EnvironmentError(f"Failed to import registration module {name!r}: {exc}") from exc
        gym.register_envs(module)


def _package_version(name: str) -> str:
    """Installed distribution version, or ``unknown``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _stable_hash(payload: Any) -> str:
    """SHA-256 of a canonical JSON serialization."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
