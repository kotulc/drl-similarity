"""Page 2 — inspect the frozen clips behind each scenario label."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from drl_similarity.dashboard import charts, state
from drl_similarity.dashboard import components as ui
from drl_similarity.dashboard import data as dd
from drl_similarity.dashboard.data import DashboardData


ALL = ui.ALL


def _choice(label: str, options: list[str], key: str) -> str:
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return str(st.selectbox(label, options, key=key))


def _controls(data: DashboardData) -> pd.DataFrame:
    clips = data.clips
    labels = [label for label in data.study.labels if label in set(clips["scenario"])]
    extra = sorted(set(clips["scenario"]) - set(labels))
    with st.sidebar:
        st.markdown("#### Clip filters")
        scenario = _choice("Scenario", [*labels, *extra], ui.EXPLORER_SCENARIO)
        split = _choice("Split", [ALL, *sorted(clips["split"].unique())], ui.EXPLORER_SPLIT)
        status = _choice("Review status", [ALL, *sorted(clips["review_status"].unique())], ui.EXPLORER_STATUS)
    rows = clips[clips["scenario"] == scenario]
    if split != ALL:
        rows = rows[rows["split"] == split]
    if status != ALL:
        rows = rows[rows["review_status"] == status]
    return rows


def _counts(data: DashboardData) -> None:
    clips = data.clips
    counts = (
        clips.groupby(["scenario", "split", "review_status"]).size().rename("clips").reset_index()
        if len(clips)
        else pd.DataFrame(columns=["scenario", "split", "review_status", "clips"])
    )
    pivot = counts.pivot_table(index="scenario", columns="split", values="clips", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex([label for label in data.study.labels if label in pivot.index])
    pivot["total"] = pivot.sum(axis=1)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Clips per scenario and split**")
        st.dataframe(pivot, use_container_width=True)
    with right:
        st.markdown("**Source**")
        scenarios = data.manifests.get("scenarios", {})
        info = scenarios.get("collector") or {}
        parts = [str(info.get(key)) for key in ("policy", "run_id", "capability", "checkpoint_id") if info.get(key)]
        collector = " · ".join(parts) if parts else "not recorded"
        if scenarios.get("synthetic"):
            collector += " (synthetic frames: no emulator, no policy)"
        st.markdown(
            f"- dataset `{scenarios.get('dataset_id', '?')}` · collector: {collector}\n"
            f"- environment fingerprint: `{str(scenarios.get('environment_fingerprint', ''))[:16]}…`\n"
            f"- clip length: {data.study.clip_steps} steps (+ up to {data.study.burn_in_steps} burn-in frames)\n"
            "- split unit: whole clips; train and test clips never share source frames"
        )


def render(data: DashboardData) -> None:
    ui.header(
        data,
        "Scenario explorer",
        "What do baseline, danger, and goal look like? Review the frozen observations before reading latent results.",
    )
    rows = _controls(data)
    _counts(data)
    st.divider()
    if rows.empty:
        ui.empty_state("No clips match the current filters.")
        return
    clip_ids = rows["clip_id"].astype(str).tolist()
    if st.session_state.get(state.EXPLORER_CLIP) not in clip_ids:
        st.session_state[state.EXPLORER_CLIP] = clip_ids[0]
    clip_id = st.selectbox("Clip", clip_ids, key=state.EXPLORER_CLIP)
    record = rows[rows["clip_id"] == clip_id].iloc[0]
    step = ui.clip_player(data, record)
    options = ui.chart_options(data)
    st.plotly_chart(
        charts.clip_timeline(
            start_step=int(record["start_step"]), anchor_step=int(record["anchor_step"]),
            end_step=int(record["end_step"]), burn_in_steps=int(record["burn_in_steps"]),
            current_step=int(record["start_step"]) + int(step), anchor_event=str(record["anchor_event"]),
            scenario=str(record["scenario"]), options=options,
        ),
        use_container_width=True, key="clip_timeline", config={"displayModeBar": False},
    )  # fmt: skip
    st.caption(
        "The timeline shows what the manifest records: the clip window, burn-in, and anchor event. "
        "Per-step actions, rewards, and lives are not stored with curated clips, and source RGB frames "
        "are only kept in the collection candidates directory."
    )
    meta_left, meta_right = st.columns(2)
    with meta_left:
        st.markdown(
            f"- clip `{record['clip_id']}` · scenario **{record['scenario']}** · split **{record['split']}**\n"
            f"- source episode `{record['source_episode_id']}` · collection seed {int(record['collection_seed'])}\n"
            f"- steps: start {int(record['start_step'])}, anchor {int(record['anchor_step'])} "
            f"({record['anchor_event']}), end {int(record['end_step'])}"
        )
    with meta_right:
        st.markdown(
            f"- review: **{record['review_status']}** · curator: {record['curator'] or '—'}\n"
            f"- used in analysis: {'yes' if bool(record['analysed']) else 'no'}\n"
            f"- observation sha256 `{str(record['observation_sha256'])[:24]}…`\n"
            f"- notes: {record['notes'] or '—'}"
        )
    st.divider()
    st.markdown(f"#### Contact sheet — {record['scenario']}")
    anchor_offset = int(record["anchor_step"]) - int(record["start_step"])
    use_anchor = 0 <= anchor_offset < data.study.clip_steps
    position = st.radio(
        "Frame shown for each clip",
        ["anchor", "first", "last"] if use_anchor else ["first", "last"],
        horizontal=True, key="contact_position",
    )  # fmt: skip
    frames = []
    shown = []
    for item in rows.itertuples():
        offset = {"first": 0, "last": data.study.clip_steps - 1}.get(position)
        if offset is None:
            offset = min(max(int(item.anchor_step) - int(item.start_step), 0), data.study.clip_steps - 1)
        image = ui.frame_image(data, str(item.clip_id), int(offset))
        if image is not None:
            frames.append(image)
            shown.append(str(item.clip_id))
    if frames:
        ui.show_image(dd.contact_sheet(frames, columns=min(8, len(frames))), caption="Clips in order: " + ", ".join(shown))
    else:
        ui.empty_state("No frames could be read for these clips.")
    ui.provenance_expander(
        data,
        [
            f"scenario manifest `{dd.redact_text(data.study.scenario_manifest)}`",
            "the player reads frozen observations; it never replays the emulator",
        ],
    )
