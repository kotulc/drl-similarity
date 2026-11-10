"""Full validation of a curated scenario manifest and its frozen observations.

Every check runs in one pass so a curator can fix all reported problems together.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from drl_similarity.config import ResolvedConfig, StudyConfig
from drl_similarity.errors import DrlSimilarityError, ScenarioError
from drl_similarity.logging import get_logger, log_extra
from drl_similarity.scenarios.schema import (
    SPLITS,
    ClipRecord,
    ScenarioManifest,
    load_manifest,
    observation_sha256,
)
from drl_similarity.study_io import load_run, observations_path, require_study


_MAX_LISTED = 8


def _listed(values: Iterable[str]) -> str:
    items = list(values)
    shown = ", ".join(items[:_MAX_LISTED])
    if len(items) > _MAX_LISTED:
        shown += f", ... (+{len(items) - _MAX_LISTED} more)"
    return shown


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _windows_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def collect_scenario_issues(resolved: ResolvedConfig) -> list[str]:
    """Return every validation failure for the study's scenario dataset."""
    study = require_study(resolved, stage="validate-scenarios")
    cfg = study.scenarios
    manifest_path = Path(cfg.manifest)
    if not manifest_path.exists():
        return [
            f"scenario manifest not found: {manifest_path} "
            "(run drl-sim collect-scenarios, then curate the generated template)"
        ]
    try:
        manifest = load_manifest(manifest_path)
    except ScenarioError as exc:
        return [str(exc)]

    issues: list[str] = []
    if not manifest.clips:
        return [f"scenario manifest contains no clips: {manifest_path}"]

    considered = [clip for clip in manifest.clips if clip.review_status != "rejected"]
    active = manifest.active_clips(require_approved=cfg.review.require_approved)

    issues.extend(_identity_issues(manifest))
    issues.extend(_label_issues(considered, cfg.labels))
    issues.extend(_review_issues(manifest, study))
    issues.extend(_window_issues(considered, study))
    issues.extend(_count_issues(active, study))
    issues.extend(_overlap_issues(active))
    issues.extend(_run_issues(manifest, considered, study))
    issues.extend(_observation_issues(considered, study))
    return issues


def _identity_issues(manifest: ScenarioManifest) -> list[str]:
    issues: list[str] = []
    dup_ids = _duplicates(clip.clip_id for clip in manifest.clips)
    if dup_ids:
        issues.append(f"duplicate clip_id values: {_listed(dup_ids)}")
    dup_keys = _duplicates(clip.observation_key for clip in manifest.clips)
    if dup_keys:
        issues.append(f"duplicate observation_key values: {_listed(dup_keys)}")
    return issues


def _label_issues(clips: list[ClipRecord], labels: list[str]) -> list[str]:
    illegal = sorted({clip.scenario for clip in clips if clip.scenario not in labels})
    if not illegal:
        return []
    offenders = [clip.clip_id for clip in clips if clip.scenario not in labels]
    return [
        f"illegal scenario label(s) {illegal}; configured labels are {labels} "
        f"(clips: {_listed(offenders)})"
    ]


def _review_issues(manifest: ScenarioManifest, study: StudyConfig) -> list[str]:
    review = study.scenarios.review
    issues: list[str] = []
    pending = [clip.clip_id for clip in manifest.clips if clip.review_status == "pending"]
    if review.require_approved and pending:
        issues.append(
            f"{len(pending)} clip(s) still pending review; set review_status to approved or "
            f"rejected: {_listed(pending)}"
        )
    approved = [clip for clip in manifest.clips if clip.review_status == "approved"]
    if review.require_approved and not approved:
        issues.append("no approved clips in the manifest")
    under_reviewed = [
        clip.clip_id for clip in approved if len(clip.reviewers) < review.minimum_reviewers
    ]
    if under_reviewed:
        issues.append(
            f"{len(under_reviewed)} approved clip(s) list fewer than "
            f"{review.minimum_reviewers} reviewer(s) in 'curator': {_listed(under_reviewed)}"
        )
    return issues


def _window_issues(clips: list[ClipRecord], study: StudyConfig) -> list[str]:
    cfg = study.scenarios
    issues: list[str] = []
    wrong_length = [
        f"{clip.clip_id} ({clip.n_steps})" for clip in clips if clip.n_steps != cfg.clip_steps
    ]
    if wrong_length:
        issues.append(
            f"clip length must be exactly {cfg.clip_steps} steps "
            f"(end_step - start_step + 1): {_listed(wrong_length)}"
        )
    long_burn_in = [
        f"{clip.clip_id} ({clip.burn_in_steps})"
        for clip in clips
        if clip.burn_in_steps > cfg.burn_in_steps
    ]
    if long_burn_in:
        issues.append(
            f"burn_in_steps exceeds configured maximum {cfg.burn_in_steps}: {_listed(long_burn_in)}"
        )
    before_start = [clip.clip_id for clip in clips if clip.burn_in_steps > clip.start_step]
    if before_start:
        issues.append(
            f"burn-in window starts before the source episode (burn_in_steps > start_step): "
            f"{_listed(before_start)}"
        )
    return issues


def _count_issues(active: list[ClipRecord], study: StudyConfig) -> list[str]:
    cfg = study.scenarios
    issues: list[str] = []
    counts = {label: 0 for label in cfg.labels}
    split_counts: dict[str, Counter[str]] = {label: Counter() for label in cfg.labels}
    for clip in active:
        if clip.scenario in counts:
            counts[clip.scenario] += 1
            split_counts[clip.scenario][clip.split] += 1
    undersized = {label: n for label, n in counts.items() if n < cfg.minimum_clips_per_label}
    if undersized:
        issues.append(
            f"fewer than {cfg.minimum_clips_per_label} usable clips for label(s): {undersized}"
        )
    if len(set(counts.values())) > 1:
        issues.append(f"usable clip counts are not balanced across labels: {counts}")
    for label in cfg.labels:
        missing = [split for split in SPLITS if split_counts[label][split] == 0]
        if missing and counts[label] > 0:
            issues.append(
                f"label {label!r} has no clips in split(s) {missing}; "
                "held-out analyses need both train and test clips"
            )
    return issues


def _overlap_issues(active: list[ClipRecord]) -> list[str]:
    issues: list[str] = []
    by_episode: dict[str, list[ClipRecord]] = defaultdict(list)
    for clip in active:
        by_episode[clip.source_episode_id].append(clip)
    for episode_id, clips in sorted(by_episode.items()):
        ordered = sorted(clips, key=lambda clip: (clip.start_step, clip.clip_id))
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                if _windows_overlap(
                    first.start_step, first.end_step, second.start_step, second.end_step
                ):
                    issues.append(
                        f"source windows overlap in episode {episode_id}: "
                        f"{first.clip_id} [{first.start_step}, {first.end_step}] and "
                        f"{second.clip_id} [{second.start_step}, {second.end_step}]"
                    )
                elif first.split != second.split and _windows_overlap(
                    first.start_step - first.burn_in_steps,
                    first.end_step,
                    second.start_step - second.burn_in_steps,
                    second.end_step,
                ):
                    issues.append(
                        f"split isolation violated in episode {episode_id}: burn-in of "
                        f"{first.clip_id} ({first.split}) and {second.clip_id} ({second.split}) "
                        "share source frames"
                    )
    by_checksum: dict[str, list[str]] = defaultdict(list)
    for clip in active:
        by_checksum[clip.observation_sha256].append(clip.clip_id)
    for clip_ids in by_checksum.values():
        if len(clip_ids) > 1:
            issues.append(f"clips have byte-identical observations: {_listed(sorted(clip_ids))}")
    return issues


def _run_issues(
    manifest: ScenarioManifest, clips: list[ClipRecord], study: StudyConfig
) -> list[str]:
    issues: list[str] = []
    clip_seeds = {clip.collection_seed for clip in clips}
    for index, ref in enumerate(study.experiments):
        try:
            run = load_run(Path(ref.run_dir), index, require_selected=False)
        except DrlSimilarityError as exc:
            issues.append(
                f"experiment[{index}]: cannot verify environment fingerprint or seed "
                f"separation: {exc}"
            )
            continue
        if run.observation_fingerprint is None:
            issues.append(
                f"experiment[{index}] ({run.run_id}): run has no environment_fingerprint.json; "
                "cannot verify that clips match the model input"
            )
        elif run.observation_fingerprint != manifest.environment_fingerprint:
            issues.append(
                f"experiment[{index}] ({run.run_id}): observation fingerprint mismatch "
                f"(manifest {manifest.environment_fingerprint[:12]}..., "
                f"run {run.observation_fingerprint[:12]}...)"
            )
        overlap = sorted(clip_seeds & run.evaluation_seeds)
        if overlap:
            issues.append(
                f"experiment[{index}] ({run.run_id}): clip collection seeds overlap "
                f"evaluation seeds: {overlap}"
            )
    return issues


def _expected_observation(study: StudyConfig) -> tuple[tuple[int, ...], str] | None:
    for index, ref in enumerate(study.experiments):
        try:
            run = load_run(Path(ref.run_dir), index, require_selected=False)
        except DrlSimilarityError:
            continue
        expected = run.environment.expected
        return tuple(expected.observation_shape), expected.observation_dtype
    return None


def _observation_issues(clips: list[ClipRecord], study: StudyConfig) -> list[str]:
    cfg = study.scenarios
    obs_file = observations_path(study)
    if not obs_file.exists():
        return [f"missing observations file: {obs_file}"]
    expected = _expected_observation(study)
    issues: list[str] = []
    try:
        archive = np.load(obs_file, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - corrupt archives surface as one issue
        return [f"cannot read observations file {obs_file}: {exc}"]
    with archive:
        available = set(archive.files)
        missing = [clip.clip_id for clip in clips if clip.observation_key not in available]
        if missing:
            issues.append(f"observation arrays missing from {obs_file.name}: {_listed(missing)}")
        bad_shape: list[str] = []
        bad_dtype: list[str] = []
        bad_checksum: list[str] = []
        for clip in clips:
            if clip.observation_key not in available:
                continue
            array = archive[clip.observation_key]
            want_frames = clip.burn_in_steps + cfg.clip_steps
            want_shape = (want_frames, *expected[0]) if expected else None
            if array.shape[0] != want_frames or (want_shape and array.shape != want_shape):
                bad_shape.append(f"{clip.clip_id} {tuple(array.shape)}")
            if expected and str(array.dtype) != expected[1]:
                bad_dtype.append(f"{clip.clip_id} ({array.dtype})")
            if observation_sha256(array) != clip.observation_sha256:
                bad_checksum.append(clip.clip_id)
        if bad_shape:
            frame_shape = expected[0] if expected else "(observation shape)"
            issues.append(
                "observation arrays must be shaped (burn_in_steps + "
                f"{cfg.clip_steps}, *{frame_shape}): {_listed(bad_shape)}"
            )
        if bad_dtype and expected:
            issues.append(f"observation dtype must be {expected[1]}: {_listed(bad_dtype)}")
        if bad_checksum:
            issues.append(f"observation checksum mismatch: {_listed(bad_checksum)}")
    return issues


def validate_scenarios(resolved: ResolvedConfig) -> None:
    """Raise ``ScenarioError`` listing every validation failure."""
    issues = collect_scenario_issues(resolved)
    logger = get_logger("scenarios")
    extra = log_extra(stage="validate-scenarios", config_hash=resolved.config_hash)
    if issues:
        logger.error("scenario validation found %d issue(s)", len(issues), extra=extra)
        body = "\n".join(f"  - {issue}" for issue in issues)
        raise ScenarioError(f"Scenario validation failed with {len(issues)} issue(s):\n{body}")
    logger.info("scenario manifest valid", extra=extra)


def load_validated_clips(resolved: ResolvedConfig) -> tuple[ScenarioManifest, list[ClipRecord]]:
    """Validate, then return the manifest and the clips used downstream in manifest order."""
    study = require_study(resolved, stage="scenarios")
    validate_scenarios(resolved)
    manifest = load_manifest(Path(study.scenarios.manifest))
    return manifest, manifest.active_clips(require_approved=study.scenarios.review.require_approved)
