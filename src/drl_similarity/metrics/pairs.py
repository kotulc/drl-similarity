"""Pair construction for within- and between-scenario comparisons.

The primary estimates compare frames from *different* clips; pairs within one
clip are excluded so adjacent-frame autocorrelation cannot dominate. Three pair
sets are supported:

``cross_clip_all``
    every step of clip A with every step of clip B;
``cross_clip_aligned_step``
    only steps at the same relative offset in both clips;
``clip_centroids``
    one pair of clip-mean vectors per clip pair (steps are ``-1``).

Every pair has a stable ID so an estimate can be audited back to its frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from drl_similarity.errors import AnalysisError


PAIR_SETS: tuple[str, ...] = ("cross_clip_all", "cross_clip_aligned_step", "clip_centroids")
CENTROID_STEP = -1


@dataclass(frozen=True)
class ClipPair:
    index_a: int
    index_b: int
    clip_id_a: str
    clip_id_b: str
    scenario_a: str
    scenario_b: str

    @property
    def pair_id(self) -> str:
        return clip_pair_id(self.clip_id_a, self.clip_id_b)


def clip_pair_id(clip_id_a: str, clip_id_b: str) -> str:
    return f"{clip_id_a}|{clip_id_b}"


def frame_pair_id(clip_id_a: str, step_a: int, clip_id_b: str, step_b: int) -> str:
    return f"{clip_id_a}@{step_a}|{clip_id_b}@{step_b}"


def validate_pair_set(pair_set: str) -> str:
    if pair_set not in PAIR_SETS:
        raise AnalysisError(
            f"Unknown pair set {pair_set!r}; valid pair sets: {', '.join(PAIR_SETS)}"
        )
    return pair_set


def _check_inputs(clip_ids: Sequence[str], scenarios: Sequence[str]) -> None:
    if len(clip_ids) != len(scenarios):
        raise AnalysisError(
            "clip_ids and scenarios must have the same length",
            context={"clip_ids": len(clip_ids), "scenarios": len(scenarios)},
        )
    if len(set(clip_ids)) != len(clip_ids):
        raise AnalysisError("clip_ids must be unique")


def scenario_members(scenarios: Sequence[str], labels: Sequence[str]) -> dict[str, np.ndarray]:
    """Clip indices per label, in clip order."""
    array = np.asarray(list(scenarios)).astype(str)
    return {label: np.flatnonzero(array == label) for label in labels}


def balanced_clip_indices(
    scenarios: Sequence[str], labels: Sequence[str], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Equal-sized clip subsets per label for balanced between-scenario estimates.

    Labels with more clips than the smallest label are subsampled without
    replacement using ``rng``; the returned indices are sorted.
    """
    members = scenario_members(scenarios, labels)
    sizes = [len(indices) for indices in members.values()]
    if not sizes or min(sizes) == 0:
        return members
    target = min(sizes)
    balanced: dict[str, np.ndarray] = {}
    for label in labels:
        indices = members[label]
        if len(indices) > target:
            indices = np.sort(rng.choice(indices, size=target, replace=False))
        balanced[label] = indices
    return balanced


def within_scenario_clip_pairs(
    clip_ids: Sequence[str], scenarios: Sequence[str], scenario: str
) -> list[ClipPair]:
    """Unordered pairs of distinct clips that share ``scenario``."""
    _check_inputs(clip_ids, scenarios)
    members = [index for index, label in enumerate(scenarios) if label == scenario]
    return [
        ClipPair(a, b, clip_ids[a], clip_ids[b], scenario, scenario)
        for position, a in enumerate(members)
        for b in members[position + 1 :]
    ]


def between_scenario_clip_pairs(
    clip_ids: Sequence[str],
    scenarios: Sequence[str],
    scenario_a: str,
    scenario_b: str,
    *,
    rng: np.random.Generator | None = None,
) -> list[ClipPair]:
    """Every clip of ``scenario_a`` paired with every clip of ``scenario_b``.

    When ``rng`` is given and the two scenarios have different clip counts, the
    larger one is subsampled so both contribute the same number of clips.
    """
    _check_inputs(clip_ids, scenarios)
    if scenario_a == scenario_b:
        raise AnalysisError("between-scenario pairs need two different scenarios")
    members_a = [index for index, label in enumerate(scenarios) if label == scenario_a]
    members_b = [index for index, label in enumerate(scenarios) if label == scenario_b]
    if rng is not None and members_a and members_b and len(members_a) != len(members_b):
        target = min(len(members_a), len(members_b))
        if len(members_a) > target:
            members_a = sorted(rng.choice(members_a, size=target, replace=False).tolist())
        if len(members_b) > target:
            members_b = sorted(rng.choice(members_b, size=target, replace=False).tolist())
    return [
        ClipPair(a, b, clip_ids[a], clip_ids[b], scenario_a, scenario_b)
        for a in members_a
        for b in members_b
    ]


def step_pairs(pair_set: str, n_steps: int) -> np.ndarray:
    """``(pairs, 2)`` array of (step_a, step_b) combinations for one clip pair."""
    validate_pair_set(pair_set)
    if n_steps < 1:
        raise AnalysisError("n_steps must be positive")
    steps = np.arange(n_steps)
    if pair_set == "cross_clip_all":
        grid_a, grid_b = np.meshgrid(steps, steps, indexing="ij")
        return np.stack([grid_a.ravel(), grid_b.ravel()], axis=1)
    if pair_set == "cross_clip_aligned_step":
        return np.stack([steps, steps], axis=1)
    return np.array([[CENTROID_STEP, CENTROID_STEP]])


def build_frame_pairs(clip_pairs: Sequence[ClipPair], pair_set: str, n_steps: int) -> pd.DataFrame:
    """Explicit frame-level pair table for a list of clip pairs.

    Same-clip pairs are rejected: cross-clip pair sets never compare a clip with
    itself.
    """
    steps = step_pairs(pair_set, n_steps)
    rows: list[dict[str, object]] = []
    for pair in clip_pairs:
        if pair.index_a == pair.index_b or pair.clip_id_a == pair.clip_id_b:
            raise AnalysisError(
                "cross-clip pair sets must not contain same-clip pairs",
                context={"clip_id": pair.clip_id_a},
            )
        for step_a, step_b in steps:
            rows.append(
                {
                    "pair_id": frame_pair_id(pair.clip_id_a, int(step_a), pair.clip_id_b, int(step_b)),
                    "pair_set": pair_set,
                    "clip_index_a": pair.index_a,
                    "clip_index_b": pair.index_b,
                    "clip_id_a": pair.clip_id_a,
                    "clip_id_b": pair.clip_id_b,
                    "scenario_a": pair.scenario_a,
                    "scenario_b": pair.scenario_b,
                    "step_a": int(step_a),
                    "step_b": int(step_b),
                }
            )
    columns = [
        "pair_id",
        "pair_set",
        "clip_index_a",
        "clip_index_b",
        "clip_id_a",
        "clip_id_b",
        "scenario_a",
        "scenario_b",
        "step_a",
        "step_b",
    ]
    return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True)
class RepresentativeFrame:
    scenario: str
    clip_index: int
    clip_id: str
    step: int
    frame_rank: int  # 0-based rank of the clip within its scenario
    offset_rank: int  # 0-based rank of the relative step offset


def representative_steps(n_steps: int, count: int = 3) -> list[int]:
    """Evenly spaced relative steps (first, middle, last for ``count=3``)."""
    if n_steps < 1:
        raise AnalysisError("n_steps must be positive")
    values = np.unique(np.rint(np.linspace(0, n_steps - 1, num=min(count, n_steps))).astype(int))
    return [int(value) for value in values]


def representative_frames(
    clip_ids: Sequence[str],
    scenarios: Sequence[str],
    labels: Sequence[str],
    n_steps: int,
    *,
    clips_per_scenario: int = 3,
    steps_per_clip: int = 3,
) -> list[RepresentativeFrame]:
    """Deterministic frames used for the dashboard's frame-pair matrix.

    The first ``clips_per_scenario`` clips of each label (manifest order) are
    used, each at ``steps_per_clip`` evenly spaced relative steps.
    """
    _check_inputs(clip_ids, scenarios)
    steps = representative_steps(n_steps, steps_per_clip)
    frames: list[RepresentativeFrame] = []
    for label in labels:
        members = [index for index, value in enumerate(scenarios) if value == label]
        for frame_rank, clip_index in enumerate(members[:clips_per_scenario]):
            for offset_rank, step in enumerate(steps):
                frames.append(
                    RepresentativeFrame(
                        scenario=label,
                        clip_index=clip_index,
                        clip_id=clip_ids[clip_index],
                        step=step,
                        frame_rank=frame_rank,
                        offset_rank=offset_rank,
                    )
                )
    return frames
