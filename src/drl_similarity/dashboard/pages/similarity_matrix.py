"""Page 1 — case-to-case matrix that drills into a frame-pair matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from drl_similarity.dashboard import charts, state
from drl_similarity.dashboard import components as ui
from drl_similarity.dashboard import data as dd
from drl_similarity.dashboard.data import CELL_METRICS, DashboardData


def _on_row_change(labels: tuple[str, ...]) -> None:
    state.set_row_scenario(st.session_state, st.session_state[state.ROW_SCENARIO], labels)


def _on_column_change(labels: tuple[str, ...]) -> None:
    state.set_column_scenario(st.session_state, st.session_state[state.COLUMN_SCENARIO], labels)


def _on_frames_change() -> None:
    state.set_frames_per_case(st.session_state, st.session_state[state.FRAMES_PER_CASE])


def _swap() -> None:
    state.swap_case_axes(st.session_state)


def _control_bar(data: DashboardData) -> None:
    options = ui.filter_options(data)
    labels = options.scenarios
    first = st.columns([1.1, 1.3, 1.1, 1.1, 0.7])
    first[0].selectbox("Checkpoint", options.checkpoints, key=state.CHECKPOINT)
    first[1].selectbox("Layer", options.layers, key=state.LAYER)
    first[2].selectbox("Row case", labels, key=state.ROW_SCENARIO, on_change=_on_row_change, args=(labels,))
    first[3].selectbox("Column case", labels, key=state.COLUMN_SCENARIO, on_change=_on_column_change, args=(labels,))
    first[4].markdown("&nbsp;")
    first[4].button("⇄ Swap axes", on_click=_swap, key="matrix_swap", use_container_width=True)
    second = st.columns([1.4, 1.3, 1.0, 1.0, 0.8])
    second[0].selectbox(
        "Pair set / aggregation", options.pair_sets, key=state.PAIR_SET,
        help="How frame pairs from two different clips are combined. Linear CKA is always computed "
        "over the aligned relative steps of each clip pair.",
    )  # fmt: skip
    second[1].selectbox("Emphasized metric", options.metrics, key=state.PRIMARY_METRIC, format_func=charts.metric_label)
    second[2].radio(
        "Frames per case", list(range(1, state.MAX_FRAMES_PER_CASE + 1)), key=state.FRAMES_PER_CASE,
        horizontal=True, on_change=_on_frames_change,
    )  # fmt: skip
    second[3].markdown("&nbsp;")
    second[3].toggle("Latent signatures", key=state.SHOW_SIGNATURES)
    second[4].markdown("&nbsp;")
    second[4].button("Reset", on_click=ui.reset_to_defaults, args=(data,), key="matrix_reset", use_container_width=True)


def _filters(data: DashboardData) -> ui.Filters:
    with st.sidebar:
        st.markdown("#### Display")
        runs = ui.filter_options(data).runs
        if len(runs) > 1:
            st.selectbox("Run / seed", runs, key=state.RUN)
        st.toggle("Confidence intervals", key=state.SHOW_CI)
        st.toggle("Raw points", key=state.SHOW_RAW)
    return ui.Filters(
        run_id=str(st.session_state[state.RUN]),
        checkpoint=str(st.session_state[state.CHECKPOINT]),
        layer=str(st.session_state[state.LAYER]),
        pair_set=str(st.session_state[state.PAIR_SET]),
        scenarios=data.study.labels,
        show_ci=bool(st.session_state[state.SHOW_CI]),
        show_raw=bool(st.session_state[state.SHOW_RAW]),
        metric=str(st.session_state[state.PRIMARY_METRIC]),
    )


@dataclass(frozen=True)
class FramePairs:
    """Representative frames on each axis plus the stored values for every pair."""

    row_frames: pd.DataFrame
    column_frames: pd.DataFrame
    values: dict[tuple[int, int], dict[str, float]]
    row_signatures: list[np.ndarray] | None
    column_signatures: list[np.ndarray] | None

    def signatures(self, row: int, column: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self.row_signatures is None or self.column_signatures is None:
            return None
        return self.row_signatures[row], self.column_signatures[column]


def _frame_pairs(data: DashboardData, filters: ui.Filters, selection: state.MatrixSelection, frames: int) -> FramePairs:
    """Look up everything the frame matrix and the inspector show (read once per rerun)."""
    row_frames = dd.frames_for_case(data, selection.row_scenario, frames)
    column_frames = dd.frames_for_case(data, selection.column_scenario, frames)
    values = {
        (int(i), int(j)): dd.frame_pair_values(
            data, run_id=filters.run_id, checkpoint=filters.checkpoint, layer=filters.layer,
            clip_a=str(row_frame["clip_id"]), step_a=int(row_frame["step"]),
            clip_b=str(column_frame["clip_id"]), step_b=int(column_frame["step"]),
        )
        for i, row_frame in row_frames.iterrows()
        for j, column_frame in column_frames.iterrows()
    }  # fmt: skip

    row_signatures = column_signatures = None
    if st.session_state.get(state.SHOW_SIGNATURES, True):
        row_signatures = ui.latent_signatures(data, filters, [frame for _, frame in row_frames.iterrows()])
        column_signatures = ui.latent_signatures(data, filters, [frame for _, frame in column_frames.iterrows()])

    return FramePairs(row_frames, column_frames, values, row_signatures, column_signatures)


def _case_matrix(data: DashboardData, filters: ui.Filters, selection: state.MatrixSelection, frames: int) -> None:
    labels = data.study.labels
    primary = str(filters.metric)
    scale = ui.distance_max(data, filters)
    st.markdown("#### Case-to-case matrix")
    st.caption(
        f"Select one of {len(labels) ** 2} scenario pairs. Every cell compares frames from **different clips** "
        "with four complementary measures; a longer bar is closer (nEuc bars are inverted). "
        "Diagonal cells measure within-case consistency, off-diagonal cells cross-case similarity."
    )
    widths = [1] + [3] * len(labels)
    header = st.columns(widths)
    for index, label in enumerate(labels):
        header[index + 1].markdown(f"**{label}** &nbsp; :gray[column case]")
    for row in labels:
        columns = st.columns(widths)
        columns[0].markdown(f"**{row}**  \n:gray[row case]")
        for index, column in enumerate(labels):
            with columns[index + 1]:
                ui.case_cell(
                    data, filters, row=row, column=column, primary=primary,
                    selected=(row == selection.row_scenario and column == selection.column_scenario),
                    frames_per_case=frames, scale=scale,
                )  # fmt: skip
    table = dd.case_matrix_table(data, run_id=filters.run_id, checkpoint=filters.checkpoint, layer=filters.layer, pair_set=filters.pair_set)
    with st.expander("Case matrix as a heatmap and table (accessible fallback)"):
        ui.plot(
            charts.case_matrix_heatmap(table, primary, ui.chart_options(data), selected=(selection.row_scenario, selection.column_scenario)),
            key="case_heatmap", data=data, frame=table, name="case-matrix",
        )  # fmt: skip
        if len(table):
            st.dataframe(
                table[["row_scenario", "column_scenario", "metric", "pair_set", "estimate", "ci_low", "ci_high", "n_clips", "n_pairs", "warnings"]],
                hide_index=True, use_container_width=True,
            )  # fmt: skip
    ui.metric_note(data, primary, pair_set=dd.metric_pair_set(primary, filters.pair_set))


def _inspector(data: DashboardData, filters: ui.Filters, selection: state.MatrixSelection, pairs: FramePairs) -> None:
    """Persistent side panel for the selected frame pair."""
    st.caption("SELECTED FRAME PAIR")
    if selection.row_frame >= len(pairs.row_frames) or selection.column_frame >= len(pairs.column_frames):
        ui.empty_state("Select a frame pair below to inspect it.")
        return
    row_frame = pairs.row_frames.iloc[selection.row_frame]
    column_frame = pairs.column_frames.iloc[selection.column_frame]
    st.markdown(
        f"#### {selection.row_scenario} F{selection.row_frame + 1} / "
        f"{selection.column_scenario} F{selection.column_frame + 1}"
    )

    image_a = ui.frame_image(data, str(row_frame["clip_id"]), int(row_frame["step"]))
    image_b = ui.frame_image(data, str(column_frame["clip_id"]), int(column_frame["step"]))
    left, right = st.columns(2)
    with left:
        ui.show_image(image_a, caption=f"{row_frame['clip_id']} @ {int(row_frame['step'])}")
    with right:
        ui.show_image(image_b, caption=f"{column_frame['clip_id']} @ {int(column_frame['step'])}")
    signatures = pairs.signatures(selection.row_frame, selection.column_frame)
    if signatures is not None:
        st.plotly_chart(
            charts.latent_signature(*signatures, ui.chart_options(data), labels=("row", "column"), height=70),
            use_container_width=True, key="inspector_signature", config={"displayModeBar": False},
        )  # fmt: skip

    values = pairs.values.get((selection.row_frame, selection.column_frame), {})
    if values:
        ui.metric_bundle(data, values, str(filters.metric), key="inspector_bars", scale=ui.distance_max(data, filters))

    layer_rows = dd.select_rows(data.layers, run_id=filters.run_id, capability=filters.checkpoint, layer=filters.layer)
    features = int(layer_rows["feature_dim"].iloc[0]) if len(layer_rows) else 0
    stats = st.columns(2)
    stats[0].metric("comparison scope", "Within" if selection.within_case else "Across")
    stats[1].metric("latent features", features)
    if selection.within_case:
        st.success("**Within-case consistency.** Do frames from different clips of one case share a representation?")
    else:
        st.info("**Cross-case separation.** Does the model keep semantically different cases distinct?")

    for index, clip_id in enumerate(dict.fromkeys([str(row_frame["clip_id"]), str(column_frame["clip_id"])])):
        if st.button(f"Open {clip_id} in Scenario Explorer", key=f"open_clip_{index}", use_container_width=True):
            ui.open_clip_in_explorer(data, clip_id)


def _frame_matrix(data: DashboardData, filters: ui.Filters, selection: state.MatrixSelection, pairs: FramePairs) -> None:
    primary = str(filters.metric)
    st.markdown(f"#### Frame matrix: {selection.row_scenario} rows / {selection.column_scenario} columns")
    st.caption(
        "Frame *k* is taken from the *k*-th representative clip of its case at an early, middle, or late step. "
        "Cosine, Pearson, and normalised Euclidean compare the two frames' latent vectors. **CKA cannot be "
        "computed from two single frames**: the value shown is for the two source clips over their aligned "
        "steps. Single frame pairs have no bootstrap interval. Click a cell to inspect it above."
    )
    if pairs.row_frames.empty or pairs.column_frames.empty:
        ui.empty_state("No representative frames are stored for this case pair.")
        return
    if st.session_state.get(state.SHOW_SIGNATURES, True) and pairs.row_signatures is None:
        st.caption("Latent signatures are unavailable (activation arrays not found).")

    scale = max(
        [ui.distance_max(data, filters)]
        + [v["normalized_euclidean"] for v in pairs.values.values() if np.isfinite(v.get("normalized_euclidean", np.nan))]
    )
    widths = [0.9] + [3] * len(pairs.column_frames)
    header = st.columns(widths)
    for j in range(len(pairs.column_frames)):
        header[j + 1].markdown(f"**{selection.column_scenario} F{j + 1}**")
    for i, row_frame in pairs.row_frames.iterrows():
        columns = st.columns(widths)
        columns[0].markdown(f"**{selection.row_scenario} F{int(i) + 1}**")
        for j, column_frame in pairs.column_frames.iterrows():
            with columns[int(j) + 1]:
                ui.frame_pair_cell(
                    data, row_index=int(i), column_index=int(j), row_frame=row_frame, column_frame=column_frame,
                    values=pairs.values[(int(i), int(j))], primary=primary,
                    selected=(int(i) == selection.row_frame and int(j) == selection.column_frame),
                    scale=scale, signatures=pairs.signatures(int(i), int(j)),
                )  # fmt: skip


def _pair_details(data: DashboardData, filters: ui.Filters, selection: state.MatrixSelection, pairs: FramePairs) -> None:
    """Intervals, sample counts, input change, and clip metadata for the selected pair."""
    if selection.row_frame >= len(pairs.row_frames) or selection.column_frame >= len(pairs.column_frames):
        return
    row_frame = pairs.row_frames.iloc[selection.row_frame]
    column_frame = pairs.column_frames.iloc[selection.column_frame]
    pair_values = pairs.values.get((selection.row_frame, selection.column_frame), {})
    case_rows = dd.case_cell_rows(
        data, run_id=filters.run_id, checkpoint=filters.checkpoint, layer=filters.layer,
        pair_set=filters.pair_set, row=selection.row_scenario, column=selection.column_scenario,
    )  # fmt: skip
    with st.expander("Selected pair: all metric values, intervals, and source clips", expanded=True):
        table = pd.DataFrame(
            [
                {
                    "metric": charts.metric_label(metric),
                    "this frame pair": pair_values.get(metric, np.nan),
                    "case estimate": case_rows[metric]["estimate"] if metric in case_rows else np.nan,
                    "case CI low": case_rows[metric]["ci_low"] if metric in case_rows else np.nan,
                    "case CI high": case_rows[metric]["ci_high"] if metric in case_rows else np.nan,
                    "direction": dd.metric_direction(data, metric),
                }
                for metric in CELL_METRICS
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)

        layer_rows = dd.select_rows(data.layers, run_id=filters.run_id, capability=filters.checkpoint, layer=filters.layer)
        pooling = str(layer_rows["pooling"].iloc[0]) if len(layer_rows) else "unknown"
        n_clips = int(next(iter(case_rows.values()))["n_clips"]) if case_rows else 0
        image_a = ui.frame_image(data, str(row_frame["clip_id"]), int(row_frame["step"]))
        image_b = ui.frame_image(data, str(column_frame["clip_id"]), int(column_frame["step"]))
        change = "n/a"
        if image_a is not None and image_b is not None and image_a.shape == image_b.shape:
            change = f"{float(np.abs(image_a.astype(float) - image_b.astype(float)).mean() / 255.0):.4f}"
        st.markdown(
            f"- layer `{filters.layer}` · pooling `{pooling}` · {selection.scope}\n"
            f"- independent clips behind the case estimate: **{n_clips}** (a single frame pair is one observation)\n"
            f"- input-space change between the two newest frames (mean |Δpixel| / 255): **{change}**"
        )
        clip_ids = [str(row_frame["clip_id"]), str(column_frame["clip_id"])]
        meta = data.clips[data.clips["clip_id"].isin(clip_ids)][
            ["clip_id", "scenario", "split", "source_episode_id", "collection_seed", "start_step", "anchor_step", "end_step", "anchor_event"]
        ]
        st.dataframe(meta, hide_index=True, use_container_width=True)


def _secondary_tabs(data: DashboardData, filters: ui.Filters, selection: state.MatrixSelection) -> None:
    options = ui.chart_options(data)
    primary = str(filters.metric)
    pair_set = dd.metric_pair_set(primary, filters.pair_set)
    clip_pairs = dd.select_rows(
        data.frame_pairs, run_id=filters.run_id, checkpoint=filters.checkpoint, layer=filters.layer,
        metric=primary, pair_kind="clip_pair", pair_set=pair_set,
    )  # fmt: skip
    tabs = st.tabs(["Clip-level distribution", "Temporal lag", "All-clip heatmap", "Latent vs input change"])
    with tabs[0]:
        ui.plot(
            charts.clip_pair_distribution(clip_pairs, primary, selection.row_scenario, selection.column_scenario, options),
            key="clip_distribution", data=data, frame=clip_pairs, name="clip-pairs",
        )  # fmt: skip
        st.caption("Each point is one pair of different clips (the mean over that pair's frame pairs). Points sharing a clip are not independent.")
    with tabs[1]:
        temporal = dd.select_rows(
            data.summaries, run_id=filters.run_id, layer=filters.layer, metric="temporal_cosine",
            scope="temporal", scenario=selection.row_scenario,
        )  # fmt: skip
        ui.plot(charts.temporal_lag_curve(temporal, options), key="temporal_lag", data=data, frame=temporal, name="temporal-lag")
        st.caption(
            f"Row case **{selection.row_scenario}**, all capabilities. The dashed line is the same cosine on raw "
            "input pixels: a latent curve that only tracks it reflects static frames, not learned stability."
        )
    with tabs[2]:
        analysed = data.clips[data.clips["analysed"]]
        ui.plot(charts.clip_pair_heatmap(clip_pairs, analysed, primary, options), key="clip_heatmap", data=data, frame=clip_pairs, name="clip-heatmap")
    with tabs[3]:
        transitions = dd.select_rows(
            data.frame_pairs, run_id=filters.run_id, checkpoint=filters.checkpoint, layer=filters.layer,
            metric="temporal_cosine", pair_kind="temporal",
        )  # fmt: skip
        ui.plot(charts.latent_vs_input_change(transitions, options), key="latent_input", data=data, frame=transitions, name="latent-vs-input")


def render(data: DashboardData) -> None:
    ui.header(
        data,
        "Similarity matrix explorer",
        "How similar are latent representations within one scenario case and across cases, for one checkpoint and layer?",
    )
    _control_bar(data)
    filters = _filters(data)
    selection = state.read_matrix_state(st.session_state)  # read once per rerun
    frames = state.clamp_frames_per_case(st.session_state[state.FRAMES_PER_CASE])
    pairs = _frame_pairs(data, filters, selection, frames)

    # Matrix-first layout: case matrix with the selected-pair inspector beside it
    matrix, inspector = st.columns([3.4, 1], gap="medium")
    with matrix:
        _case_matrix(data, filters, selection, frames)
    with inspector:
        _inspector(data, filters, selection, pairs)
    st.divider()
    _frame_matrix(data, filters, selection, pairs)
    _pair_details(data, filters, selection, pairs)
    st.divider()
    _secondary_tabs(data, filters, selection)
    ui.provenance_expander(
        data,
        [
            f"checkpoint `{filters.checkpoint}`, layer `{filters.layer}`, pair set `{filters.pair_set}`",
            "same-clip pairs are excluded from every within- and between-case estimate",
        ],
    )
