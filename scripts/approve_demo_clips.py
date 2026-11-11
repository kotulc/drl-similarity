"""Auto-approve a balanced set of candidate clips so the DEMO study can run unattended.

A real study requires a human to review every clip (see docs/user-guide.md). This
script skips that review: it approves the first non-overlapping candidates per
label, rejects the rest, and re-splits the approved clips.

Usage::

    python scripts/approve_demo_clips.py --config configs/studies/frogger-demo.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from drl_similarity.config import StudyConfig, load_config
from drl_similarity.scenarios.schema import ClipRecord, assign_splits, dump_manifest, load_manifest


CURATOR = "demo-auto-approved"


def overlaps(clip: ClipRecord, chosen: list[ClipRecord]) -> bool:
    """True when the clip shares source frames (including burn-in) with a chosen clip."""
    return any(
        clip.source_episode_id == other.source_episode_id
        and clip.start_step - clip.burn_in_steps <= other.end_step
        and other.start_step - other.burn_in_steps <= clip.end_step
        for other in chosen
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="study YAML")
    args = parser.parse_args()

    study = load_config(args.config, "study").value
    assert isinstance(study, StudyConfig)
    scenarios = study.scenarios
    manifest = load_manifest(scenarios.manifest)

    # Take turns across labels so one label cannot claim every frame of an episode
    chosen: list[ClipRecord] = []
    counts = dict.fromkeys(scenarios.labels, 0)
    queues = {label: [c for c in manifest.clips if c.scenario == label] for label in scenarios.labels}
    while any(queues.values()) and min(counts.values()) < scenarios.minimum_clips_per_label:
        for label in scenarios.labels:
            while queues[label] and counts[label] < scenarios.minimum_clips_per_label:
                clip = queues[label].pop(0)
                if not overlaps(clip, chosen):
                    chosen.append(clip)
                    counts[label] += 1
                    break

    # Keep the labels balanced at the smallest count found
    keep = min(counts.values())
    approved = {c.clip_id for label in scenarios.labels for c in [x for x in chosen if x.scenario == label][:keep]}
    splits = assign_splits(
        {c.clip_id: c.scenario for c in manifest.clips if c.clip_id in approved},
        train_fraction=scenarios.split.train_fraction,
        seed=scenarios.split.seed,
    )
    for clip in manifest.clips:
        clip.review_status = "approved" if clip.clip_id in approved else "rejected"
        clip.curator = CURATOR if clip.clip_id in approved else None
        clip.split = splits.get(clip.clip_id, clip.split)

    manifest.notes = "DEMO ONLY: clips were approved by scripts/approve_demo_clips.py without human review."
    dump_manifest(manifest, scenarios.manifest)
    print(f"approved {keep} clip(s) per label ({len(approved)} total) in {scenarios.manifest}")


if __name__ == "__main__":
    main()
