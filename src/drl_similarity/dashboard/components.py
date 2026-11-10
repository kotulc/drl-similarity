"""Reusable Streamlit building blocks shared by the dashboard pages.

Pages compose these components and never read files themselves. Binary
artifacts (frozen observations, stored activation vectors) are read through
cached wrappers keyed by the metrics-manifest checksum, never by path alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from drl_similarity.dashboard import charts, state
from drl_similarity.dashboard import data as dd
from drl_similarity.dashboard.data import CELL_METRICS, DashboardData
from drl_similarity.errors import DrlSimilarityError


PAGES_KEY = "_dashboard_pages"
WARNINGS_KEY = "_dashboard_default_warnings"
EXPLORER_SCENARIO = "explorer_scenario"
EXPLORER_SPLIT = "explorer_split"
EXPLORER_STATUS = "explorer_status"
EXPLORER_STEP = "explorer_step"
ALL = "all"

BADGES = {"ok": ":green[● valid]", "warning": ":orange[● valid with warnings]", "error": ":red[● invalid]"}
SEVERITY_ICONS = {"ok": "✅", "info": "ℹ️", "warning": "⚠️", "error": "❌"}
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


@dataclass(frozen=True)
class Filters:
    run_id: str
    checkpoint: str
    layer: str
    pair_set: str
    scenarios: tuple[str, ...]
    show_ci: bool
    show_raw: bool
    metric: str | None = None


# --------------------------------------------------------------------------- state and options


def filter_options(data: DashboardData) -> state.FilterOptions:
    cell_metrics = tuple(m for m in CELL_METRICS if m in data.study.metrics)
    return state.FilterOptions(
        runs=tuple(data.runs["run_id"].astype(str)),
        checkpoints=data.study.capabilities,
        layers=data.study.layers,
        pair_sets=data.study.pair_sets,
        metrics=cell_metrics or CELL_METRICS,
        scenarios=data.study.labels,
    )


def filter_defaults(data: DashboardData) -> state.FilterDefaults:
    defaults = data.dashboard.default_filters
    return state.FilterDefaults(
        checkpoint=defaults.checkpoint,
        layer=defaults.layer,
        pair_set=defaults.pair_set,
        primary_metric=defaults.primary_metric,
        row_scenario=defaults.row_scenario,
        column_scenario=defaults.column_scenario,
        frames_per_case=defaults.frames_per_case,
        show_ci=data.dashboard.charts.confidence_intervals,
        show_raw=data.dashboard.charts.show_raw_points,
    )


def ensure_state(data: DashboardData) -> None:
    """Initialise session state from the dashboard YAML and keep it valid on every rerun."""
    warnings = state.init_state(st.session_state, filter_options(data), filter_defaults(data))
    if WARNINGS_KEY not in st.session_state:
        st.session_state[WARNINGS_KEY] = list(warnings)
    # Widget-backed values are dropped when their widget is not rendered on a page;
    # re-assigning them keeps selections across page changes.
    for key in state.PERSISTENT_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def reset_to_defaults(data: DashboardData) -> None:
    """Callback: restore the dashboard YAML defaults."""
    st.session_state[WARNINGS_KEY] = state.reset_state(
        st.session_state, filter_options(data), filter_defaults(data)
    )


def chart_options(data: DashboardData, *, height: int = 380) -> charts.ChartOptions:
    return charts.ChartOptions(
        capability_order=_merge(data.dashboard.theme.capability_order, data.study.capabilities),
        scenario_order=_merge(data.dashboard.theme.scenario_order, data.study.labels),
        show_ci=bool(st.session_state.get(state.SHOW_CI, True)),
        show_raw=bool(st.session_state.get(state.SHOW_RAW, True)),
        max_heatmap_samples=int(data.dashboard.charts.max_heatmap_samples),
        height=height,
    )


def _merge(preferred: list[str] | tuple[str, ...], present: tuple[str, ...]) -> tuple[str, ...]:
    head = [item for item in preferred if item in present]
    return tuple(head + [item for item in present if item not in head])


# --------------------------------------------------------------------------- shell


def header(data: DashboardData, title: str, purpose: str) -> None:
    study = data.study
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### {title}")
        st.caption(purpose)
    with right:
        st.markdown(f"**{study.title}** &nbsp; {BADGES[data.status]}")
        st.caption(
            f"study `{study.study_id}` · config `{study.config_hash[:12]}` · "
            f"artifact `{study.metrics_version}` · {study.n_clips} clips × {study.clip_steps} steps"
        )
    for issue in data.quality:
        if issue.code == "synthetic_data":
            st.error(f"**Synthetic data.** {issue.message}", icon="🧪")
        elif issue.code == "single_seed":
            st.warning(issue.message, icon="⚠️")
        elif issue.severity == "error":
            st.error(issue.message, icon="❌")
    for warning in st.session_state.get(WARNINGS_KEY, []):
        st.warning(f"Dashboard default ignored: {warning}", icon="⚠️")


def sidebar_filters(
    data: DashboardData,
    *,
    show: tuple[str, ...] = ("checkpoint", "layer", "pair_set", "scenarios", "ci", "raw"),
    metrics: tuple[str, ...] | None = None,
    metric_key: str | None = None,
    metric_title: str = "Metric",
) -> Filters:
    """Global filters. Controls with no effect on the current page are not rendered."""
    options = filter_options(data)
    metric: str | None = None
    with st.sidebar:
        st.markdown("#### Filters")
        if len(options.runs) > 1:
            st.selectbox("Run / seed", options.runs, key=state.RUN)
        if "checkpoint" in show:
            st.selectbox("Checkpoint capability", options.checkpoints, key=state.CHECKPOINT)
        if "layer" in show:
            st.selectbox("Layer", options.layers, key=state.LAYER)
        if "scenarios" in show:
            st.multiselect("Scenarios", options.scenarios, key=state.SCENARIOS)
        if "pair_set" in show:
            st.selectbox(
                "Pair set / aggregation",
                options.pair_sets,
                key=state.PAIR_SET,
                help="cross_clip_all: every frame pair from different clips · "
                "cross_clip_aligned_step: same relative step only · clip_centroids: one mean vector per clip",
            )
        if metrics and metric_key:
            if st.session_state.get(metric_key) not in metrics:
                st.session_state[metric_key] = metrics[0]
            metric = st.selectbox(metric_title, metrics, key=metric_key, format_func=charts.metric_label)
        if "ci" in show:
            st.toggle("Confidence intervals", key=state.SHOW_CI)
        if "raw" in show:
            st.toggle("Raw points", key=state.SHOW_RAW)
        st.button("Reset to YAML defaults", on_click=reset_to_defaults, args=(data,), key="sidebar_reset")
    selected = tuple(st.session_state.get(state.SCENARIOS) or options.scenarios)
    return Filters(
        run_id=str(st.session_state[state.RUN]),
        checkpoint=str(st.session_state[state.CHECKPOINT]),
        layer=str(st.session_state[state.LAYER]),
        pair_set=str(st.session_state[state.PAIR_SET]),
        scenarios=tuple(label for label in options.scenarios if label in selected),
        show_ci=bool(st.session_state[state.SHOW_CI]),
        show_raw=bool(st.session_state[state.SHOW_RAW]),
        metric=metric,
    )


def empty_state(message: str) -> None:
    st.info(message, icon="ℹ️")


def metric_note(data: DashboardData, metric: str, *, n_clips: int | None = None, pair_set: str | None = None) -> None:
    """Direction, normalisation, pair definition, independent clip count, and bootstrap seed."""
    info = data.metric_info.get(metric, {})
    boot = data.study.bootstrap_info
    parts = [f"**{charts.metric_label(metric)}** — {info.get('direction', dd.metric_direction(data, metric))}"]
    if info.get("preprocessing"):
        parts.append(f"normalisation: {info['preprocessing']}")
    if pair_set:
        parts.append(f"pairs: `{pair_set}`, never within one clip")
    parts.append(f"independent clips: {n_clips if n_clips is not None else data.study.n_clips}")
    if boot:
        parts.append(
            f"{int(100 * float(boot.get('confidence_level', 0.95)))}% percentile interval, "
            f"{boot.get('replicates')} clip resamples, seed {boot.get('seed')}"
        )
    st.caption(" · ".join(parts))


def plot(
    fig: go.Figure,
    *,
    key: str,
    data: DashboardData | None = None,
    frame: pd.DataFrame | None = None,
    name: str | None = None,
) -> None:
    """Render a chart and, when enabled, CSV/PNG downloads for exactly what is shown."""
    st.plotly_chart(fig, use_container_width=True, key=key, config=PLOT_CONFIG)
    if data is None or not data.dashboard.downloads.enabled:
        return
    formats = data.dashboard.downloads.formats
    name = name or key
    columns = st.columns([1, 1, 1])
    if frame is not None and len(frame) and "csv" in formats:
        columns[0].download_button(
            "CSV", dd.to_csv_bytes(frame), file_name=f"{data.study.study_id}-{name}.csv",
            mime="text/csv", key=f"{key}_csv", help="Download the rows behind this chart",
        )  # fmt: skip
    if "png" in formats and columns[1].button("PNG", key=f"{key}_png_prepare", help="Render this chart as a PNG"):
        try:
            image = fig.to_image(format="png", scale=2)
        except Exception as exc:  # noqa: BLE001 - kaleido/plotly version mismatches surface here
            columns[2].caption(
                f"Server-side PNG export is unavailable ({type(exc).__name__}). "
                "Use the camera icon in the chart toolbar instead."
            )
        else:
            columns[2].download_button(
                "Save PNG", image, file_name=f"{data.study.study_id}-{name}.png", mime="image/png", key=f"{key}_png"
            )


def provenance_expander(data: DashboardData, lines: list[str] | None = None) -> None:
    study = data.study
    with st.expander("Provenance"):
        st.markdown(
            "\n".join(
                [
                    f"- study `{study.study_id}`, config hash `{study.config_hash}`",
                    f"- metrics artifact `{study.metrics_version}`, manifest sha256 `{study.metrics_manifest_sha256[:16]}…`",
                    f"- generated {study.generated_at}",
                    f"- bootstrap: {study.bootstrap_info}",
                    *[f"- {line}" for line in (lines or [])],
                ]
            )
        )


def quality_table(data: DashboardData) -> None:
    rows = [
        {"status": SEVERITY_ICONS.get(issue.severity, ""), "check": issue.code, "detail": issue.message}
        for issue in data.quality
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- cached binary readers


@st.cache_data(show_spinner=False, max_entries=512)
def _observations(observations_file: str, observation_key: str, cache_key: str) -> np.ndarray:
    del cache_key  # explicit checksum key: a new metrics manifest invalidates cached frames
    return dd.read_clip_observations(observations_file, observation_key)


@st.cache_data(show_spinner=False, max_entries=256)
def _latents(activations_dir: str, relpath: str, picks: tuple[tuple[int, int], ...], cache_key: str) -> np.ndarray:
    del cache_key
    return dd.read_latent_vectors(activations_dir, relpath, picks)


@st.cache_data(show_spinner=False, max_entries=64)
def _gif(observations_file: str, observation_key: str, burn_in: int, fps: int, cache_key: str) -> bytes:
    frames = _observations(observations_file, observation_key, cache_key)[burn_in:]
    return dd.clip_gif(np.stack([dd.newest_frame(frame) for frame in frames]), fps=fps)


def clip_record(data: DashboardData, clip_id: str) -> pd.Series | None:
    rows = data.clips[data.clips["clip_id"] == clip_id]
    return None if rows.empty else rows.iloc[0]


def clip_observations(data: DashboardData, clip_id: str, *, include_burn_in: bool = False) -> np.ndarray | None:
    """Frozen observations of a clip ``(steps, *obs_shape)``; ``None`` when unreadable."""
    record = clip_record(data, clip_id)
    if record is None:
        return None
    try:
        frames = _observations(
            data.study.observations_file, str(record["observation_key"]), data.study.metrics_manifest_sha256
        )
    except DrlSimilarityError:
        return None
    return frames if include_burn_in else frames[int(record["burn_in_steps"]) :]


def frame_image(data: DashboardData, clip_id: str, step: int) -> np.ndarray | None:
    """Newest 2-D frame of the observation at a relative clip step."""
    frames = clip_observations(data, clip_id)
    if frames is None or not 0 <= step < len(frames):
        return None
    return dd.newest_frame(frames[step])


def upscale(image: np.ndarray, factor: int = 2) -> np.ndarray:
    return np.kron(image, np.ones((factor, factor), dtype=image.dtype))


def ribbon(images: list[np.ndarray | None], *, gap: int = 3, divider_after: int | None = None) -> np.ndarray | None:
    """Frames side by side; an optional wider divider separates row-case and column-case frames."""
    valid = [image for image in images if image is not None]
    if not valid:
        return None
    height = valid[0].shape[0]
    parts: list[np.ndarray] = []
    for index, image in enumerate(images):
        if image is None:
            continue
        if parts:
            width = gap * 4 if divider_after is not None and index == divider_after else gap
            parts.append(np.full((height, width), 255, dtype=np.uint8))
        parts.append(np.asarray(image, dtype=np.uint8))
    return np.concatenate(parts, axis=1)


def show_image(image: np.ndarray | None, *, caption: str | None = None) -> None:
    if image is None:
        st.caption("frame unavailable")
        return
    st.image(image, caption=caption, use_container_width=True, clamp=True)


# --------------------------------------------------------------------------- metric bundles


def distance_max(data: DashboardData, filters: Filters) -> float:
    """Shared scale for the normalised-Euclidean bars: the largest case estimate for this layer."""
    rows = dd.select_rows(
        data.summaries, run_id=filters.run_id, layer=filters.layer, metric="normalized_euclidean",
        scope=["within_scenario", "between_scenario"],
    )  # fmt: skip
    top = float(np.nanmax(rows["estimate"].to_numpy(dtype=float))) if len(rows) else 1.0
    return max(top, 1e-6) * 1.05


def metric_bundle(
    data: DashboardData,
    values: dict[str, float],
    primary: str,
    *,
    key: str,
    scale: float,
    intervals: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Cosine, Pearson, CKA, and normalised Euclidean as four bars; the primary metric is accented."""
    fig = charts.metric_bars(values, primary, chart_options(data), intervals=intervals, distance_max=scale)
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


def case_cell(
    data: DashboardData,
    filters: Filters,
    *,
    row: str,
    column: str,
    primary: str,
    selected: bool,
    frames_per_case: int,
    scale: float,
) -> None:
    """One cell of the case-to-case matrix."""
    rows = dd.case_cell_rows(
        data, run_id=filters.run_id, checkpoint=filters.checkpoint, layer=filters.layer,
        pair_set=filters.pair_set, row=row, column=column,
    )  # fmt: skip
    scope = "Within case" if row == column else "Across cases"
    with st.container(border=True):
        st.caption(scope)
        st.markdown(f"{'🔷 ' if selected else ''}**{row} / {column}**")

        # Row-case frames, then column-case frames, so every ribbon has the same width
        row_frames = dd.frames_for_case(data, row, frames_per_case)
        column_frames = dd.frames_for_case(data, column, frames_per_case)
        images = [frame_image(data, str(f.clip_id), int(f.step)) for f in row_frames.itertuples()]
        images += [frame_image(data, str(f.clip_id), int(f.step)) for f in column_frames.itertuples()]
        show_image(ribbon(images, divider_after=len(row_frames)))

        if not rows:
            st.caption("No metric rows for this cell.")
        else:
            metric_bundle(
                data,
                {metric: float(item["estimate"]) for metric, item in rows.items()},
                primary,
                key=f"case_bars_{row}_{column}",
                scale=scale,
                intervals={m: (float(i["ci_low"]), float(i["ci_high"])) for m, i in rows.items()},
            )

        # Intervals are in the bar hover text and in the fallback table below the matrix
        st.button(
            "Selected" if selected else "Open",
            key=f"case_open_{row}_{column}",
            on_click=_select_case,
            args=(row, column, tuple(data.study.labels)),
            type="primary" if selected else "secondary",
            use_container_width=True,
            help=f"Open frame pairs: row case {row}, column case {column}",
        )


def _select_case(row: str, column: str, labels: tuple[str, ...]) -> None:
    warning = state.set_case_pair(st.session_state, row, column, labels)
    if warning:
        st.session_state[WARNINGS_KEY] = [warning]


def _select_frames(row_frame: int, column_frame: int) -> None:
    state.set_frame_pair(st.session_state, row_frame, column_frame)


def frame_pair_cell(
    data: DashboardData,
    *,
    row_index: int,
    column_index: int,
    row_frame: pd.Series,
    column_frame: pd.Series,
    values: dict[str, float],
    primary: str,
    selected: bool,
    scale: float,
    signatures: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    """One cell of the frame-pair matrix."""
    same_frame = row_frame["clip_id"] == column_frame["clip_id"] and int(row_frame["step"]) == int(column_frame["step"])
    with st.container(border=True):
        show_image(
            ribbon(
                [
                    frame_image(data, str(row_frame["clip_id"]), int(row_frame["step"])),
                    frame_image(data, str(column_frame["clip_id"]), int(column_frame["step"])),
                ],
                gap=6,
            )
        )
        st.caption(
            f"{row_frame['clip_id']} @ step {int(row_frame['step'])} vs "
            f"{column_frame['clip_id']} @ step {int(column_frame['step'])}"
            + (" · same frame" if same_frame else "")
        )
        if signatures is not None:
            st.plotly_chart(
                charts.latent_signature(signatures[0], signatures[1], chart_options(data), labels=("row", "column"), height=70),
                use_container_width=True, key=f"signature_{row_index}_{column_index}",
                config={"displayModeBar": False},
            )  # fmt: skip
        if values:
            metric_bundle(data, values, primary, key=f"frame_bars_{row_index}_{column_index}", scale=scale)
        else:
            st.caption("No stored values for this frame pair.")
        st.button(
            "Inspecting" if selected else "Inspect",
            key=f"frame_open_{row_index}_{column_index}",
            on_click=_select_frames,
            args=(row_index, column_index),
            type="primary" if selected else "secondary",
            use_container_width=True,
        )


def latent_signatures(
    data: DashboardData, filters: Filters, frames: list[pd.Series]
) -> list[np.ndarray] | None:
    """Block-averaged stored activation vectors for the given representative frames."""
    relpath = dd.activation_path(data, filters.run_id, filters.checkpoint, filters.layer)
    if relpath is None or data.study.activations_dir is None or not frames:
        return None
    index = {str(row.clip_id): int(row.clip_index) for row in data.clips.itertuples() if int(row.clip_index) >= 0}
    try:
        picks = tuple((index[str(frame["clip_id"])], int(frame["step"])) for frame in frames)
        vectors = _latents(data.study.activations_dir, relpath, picks, data.study.metrics_manifest_sha256)
    except (KeyError, DrlSimilarityError, OSError):
        return None
    return [dd.latent_signature(vector) for vector in vectors]


def open_clip_in_explorer(data: DashboardData, clip_id: str) -> None:
    """Point the Scenario Explorer at a clip and switch to it."""
    record = clip_record(data, clip_id)
    if record is None:
        return
    st.session_state[EXPLORER_SCENARIO] = str(record["scenario"])
    st.session_state[EXPLORER_SPLIT] = ALL
    st.session_state[EXPLORER_STATUS] = ALL
    st.session_state[state.EXPLORER_CLIP] = clip_id
    st.session_state.pop(EXPLORER_STEP, None)
    page = st.session_state.get(PAGES_KEY, {}).get("scenario_explorer")
    if page is not None:
        st.switch_page(page)
    else:
        st.info(f"Open the Scenario Explorer page to view clip {clip_id}.")


def page_link(page_id: str, label: str, *, icon: str | None = None) -> None:
    page = st.session_state.get(PAGES_KEY, {}).get(page_id)
    if page is not None:
        st.page_link(page, label=label, icon=icon)


def clip_player(data: DashboardData, record: pd.Series, *, key: str = "player") -> int:
    """Frame slider plus looping playback of frozen observations; returns the selected step."""
    frames = clip_observations(data, str(record["clip_id"]))
    if frames is None:
        empty_state("Observations for this clip could not be read.")
        return 0
    n_steps = len(frames)
    if int(st.session_state.get(EXPLORER_STEP, 0)) >= n_steps:
        st.session_state[EXPLORER_STEP] = 0
    fps = int(data.dashboard.charts.clip_fps)
    controls = st.columns([3, 1])
    step = controls[0].slider("Relative frame (0 = first clip step)", 0, n_steps - 1, key=EXPLORER_STEP) if n_steps > 1 else 0
    playing = controls[1].toggle(f"Play ({fps} fps)", key=f"{key}_play")
    left, right = st.columns([1, 2])
    with left:
        if playing:
            try:
                st.image(
                    _gif(
                        data.study.observations_file, str(record["observation_key"]),
                        int(record["burn_in_steps"]), fps, data.study.metrics_manifest_sha256,
                    ),
                    caption="Looping playback of the frozen clip (newest frame of each stack)",
                    use_container_width=True,
                )  # fmt: skip
            except Exception as exc:  # noqa: BLE001 - playback is optional
                st.caption(f"Playback unavailable: {exc}")
        else:
            show_image(upscale(dd.newest_frame(frames[step]), 3), caption=f"Newest frame at relative step {step}")
    with right:
        show_image(
            upscale(dd.stack_strip(frames[step]), 2),
            caption="Preprocessed frame stack fed to the policy, oldest → newest (rightmost is the newest frame)",
        )
    return int(step)
