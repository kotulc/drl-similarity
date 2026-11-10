"""Environment construction, wrapper registry, frame skip, and fingerprints."""

from __future__ import annotations

import numpy as np
import pytest

from drl_similarity.config import EnvironmentConfig
from drl_similarity.environments.atari import compute_environment_fingerprints, make_environment
from drl_similarity.environments.wrappers import apply_wrappers
from drl_similarity.errors import EnvironmentError
from tests.conftest import environment_document


def _config(**changes: object) -> EnvironmentConfig:
    return EnvironmentConfig.model_validate({**environment_document(), **changes})


def test_make_environment_matches_expected(environment_config: EnvironmentConfig) -> None:
    env = make_environment(environment_config, seed=0)
    try:
        observation, _ = env.reset(seed=0)
        assert np.asarray(observation).shape == (4, 36, 36)
        assert np.asarray(observation).dtype == np.uint8
    finally:
        env.close()


def test_make_environment_seeded_reset_is_deterministic(environment_config: EnvironmentConfig) -> None:
    frames = []
    for _ in range(2):
        env = make_environment(environment_config, seed=7)
        frames.append(np.asarray(env.reset(seed=7)[0]))
        env.close()

    assert np.array_equal(frames[0], frames[1])


@pytest.mark.parametrize(
    "expected",
    [
        {"observation_shape": [4, 84, 84]},
        {"observation_dtype": "float32"},
        {"action_space": "box"},
        {"effective_frame_skip": 4},
    ],
)
def test_make_environment_rejects_unexpected(expected: dict[str, object]) -> None:
    config = _config(expected={**environment_document()["expected"], **expected})

    with pytest.raises(EnvironmentError):
        make_environment(config, seed=0)


def test_apply_wrappers_rejects_double_frame_skip() -> None:
    """A base frameskip > 1 plus a skipping wrapper must fail before the wrapper is built."""
    spec = _config(wrappers=[{"id": "sb3_atari", "kwargs": {"frame_skip": 4}}]).wrappers

    with pytest.raises(EnvironmentError, match="Double frame skip"):
        apply_wrappers(None, list(spec), base_frameskip=4)  # type: ignore[arg-type]


def test_fingerprints_are_stable(environment_config: EnvironmentConfig) -> None:
    first = compute_environment_fingerprints(environment_config)
    second = compute_environment_fingerprints(_config())

    assert (first.observation, first.runtime) == (second.observation, second.runtime)


def test_fingerprints_separate_observation_from_runtime() -> None:
    """A reward-only change alters the runtime fingerprint but not the observation one."""
    base = compute_environment_fingerprints(_config())
    reward_only = compute_environment_fingerprints(
        _config(
            wrappers=[
                {"id": "record_episode_statistics", "kwargs": {"buffer_length": 7}},
                {"id": "frame_stack", "kwargs": {"num_stack": 4}},
            ]
        )
    )

    assert base.observation == reward_only.observation
    assert base.runtime != reward_only.runtime


@pytest.mark.atari
def test_atari_frogger_smoke() -> None:
    """Real ALE environment: 100 steps at the pinned shape with a single 4x frame skip."""
    pytest.importorskip("ale_py")
    from drl_similarity.config import load_config

    config = load_config("configs/environments/atari-frogger.yaml", "environment").value
    env = make_environment(config, seed=0)  # type: ignore[arg-type]
    try:
        _, info = env.reset(seed=0)
        start = info["episode_frame_number"]
        for _ in range(100):
            observation, _, terminated, truncated, info = env.step(0)
            assert np.asarray(observation).shape == (4, 84, 84)
            if terminated or truncated:
                break
        else:
            assert info["episode_frame_number"] - start == 400
    finally:
        env.close()
