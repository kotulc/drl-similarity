"""Pure Plotly figure builders.

Every function takes already-filtered data plus an immutable :class:`ChartOptions`,
validates required columns, and returns a ``plotly.graph_objects.Figure``. Nothing
here imports Streamlit, reads files, or computes study metrics. Colours and
category order are fixed by key and never depend on the current filter order.
When no rows match, an annotated empty figure is returned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CAPABILITY_COLORS: dict[str, str] = {"weak": "#8c96a3", "medium": "#e09f3e", "strong": "#1f6feb"}
SCENARIO_COLORS: dict[str, str] = {"baseline": "#4c78a8", "danger": "#d1495b", "goal": "#3a9d5d"}
NEUTRAL_COLOR = "#b5b5b5"
_EXTRA_COLORS: tuple[str, ...] = ("#7b5ea7", "#17a2b8", "#b07c3b", "#c05299", "#5f6b7a", "#8a9a2b")
CAPABILITY_SYMBOLS: dict[str, str] = {"weak": "circle", "medium": "diamond", "strong": "star"}
_EXTRA_SYMBOLS: tuple[str, ...] = ("square", "cross", "x", "triangle-up")
SIMILARITY_SCALE = "Viridis"
DISTANCE_SCALE = "Viridis_r"
DIVERGING_SCALE = "RdBu"

METRIC_LABELS: dict[str, str] = {
    "cosine": "Cosine similarity",
    "correlation": "Pearson correlation",
    "linear_cka": "Linear CKA",
    "normalized_euclidean": "Normalized Euclidean distance",
    "centroid_dispersion": "Centroid dispersion",
    "temporal_cosine": "Temporal cosine",
    "within_between_gap": "Within − between gap",
    "silhouette": "Silhouette score",
    "nearest_centroid_accuracy": "Held-out nearest-centroid accuracy",
    "rsa_spearman": "RSA (Spearman)",
    "svcca": "SVCCA",
    "orthogonal_procrustes": "Procrustes disparity",
    "raw_coordinate_cosine": "Raw-coordinate cosine (diagnostic)",
}
METRIC_SHORT: dict[str, str] = {
    "cosine": "cos",
    "correlation": "r",
    "linear_cka": "CKA",
    "normalized_euclidean": "nEuc",
}
DISTANCE_METRICS: frozenset[str] = frozenset({"normalized_euclidean", "orthogonal_procrustes", "centroid_dispersion"})


@dataclass(frozen=True)
class ChartOptions:
    capability_order: tuple[str, ...] = ("weak", "medium", "strong")
    scenario_order: tuple[str, ...] = ("baseline", "danger", "goal")
    show_ci: bool = True
    show_raw: bool = True
    max_heatmap_samples: int = 500
    height: int = 380
    template: str = "plotly_white"


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def metric_direction(metric: str, kind: str | None = None) -> str:
    if kind == "distance" or (kind is None and metric in DISTANCE_METRICS):
        return "lower = more similar"
    if kind == "score":
        return "higher = better separated"
    return "higher = more similar"


def capability_color(capability: str, options: ChartOptions) -> str:
    if capability in CAPABILITY_COLORS:
        return CAPABILITY_COLORS[capability]
    if capability in options.capability_order:
        return _EXTRA_COLORS[options.capability_order.index(capability) % len(_EXTRA_COLORS)]
    return NEUTRAL_COLOR


def scenario_color(scenario: str, options: ChartOptions) -> str:
    if scenario in SCENARIO_COLORS:
        return SCENARIO_COLORS[scenario]
    if scenario in options.scenario_order:
        return _EXTRA_COLORS[options.scenario_order.index(scenario) % len(_EXTRA_COLORS)]
    return NEUTRAL_COLOR


def capability_symbol(capability: str, options: ChartOptions) -> str:
    if capability in CAPABILITY_SYMBOLS:
        return CAPABILITY_SYMBOLS[capability]
    if capability in options.capability_order:
        return _EXTRA_SYMBOLS[options.capability_order.index(capability) % len(_EXTRA_SYMBOLS)]
    return "circle-open"


def _ordered(values: Sequence[str], order: Sequence[str]) -> list[str]:
    present = list(dict.fromkeys(str(value) for value in values))
    head = [item for item in order if item in present]
    return head + [item for item in present if item not in head]


def _layout(fig: go.Figure, options: ChartOptions, *, title: str = "", height: int | None = None) -> go.Figure:
    """Shared template, title, margins, and a legend below the plot area."""
    fig.update_layout(
        template=options.template,
        title={"text": title, "font": {"size": 14}, "yref": "container", "y": 0.97, "yanchor": "top"},
        height=height or options.height,
        margin={"l": 60, "r": 20, "t": 70 if title else 24, "b": 50},
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0},
        hoverlabel={"align": "left"},
    )
    fig.update_yaxes(title_font={"size": 11})
    return fig


def empty_figure(message: str, options: ChartOptions | None = None, *, height: int | None = None) -> go.Figure:
    """Annotated placeholder used whenever no rows match."""
    options = options or ChartOptions()
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font={"size": 14, "color": "#666666"},
    )  # fmt: skip
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(meta={"empty": True, "message": message})
    return _layout(fig, options, height=height or 240)


def _missing(frame: pd.DataFrame | None, columns: Sequence[str]) -> str | None:
    if frame is None:
        return "No data available."
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        return f"Data is missing required column(s): {', '.join(absent)}."
    if frame.empty:
        return "No rows match the current filters."
    return None


def _error_bars(frame: pd.DataFrame, options: ChartOptions) -> dict | None:
    if not options.show_ci or frame[["ci_low", "ci_high"]].isna().all().all():
        return None
    return {
        "type": "data",
        "symmetric": False,
        "array": (frame["ci_high"] - frame["estimate"]).clip(lower=0).fillna(0).to_numpy(),
        "arrayminus": (frame["estimate"] - frame["ci_low"]).clip(lower=0).fillna(0).to_numpy(),
        "thickness": 1.2,
        "width": 4,
    }


# --------------------------------------------------------------------------- training overview


def learning_curve(evaluations: pd.DataFrame, selected: pd.DataFrame, options: ChartOptions) -> go.Figure:
    """Selection-suite return versus training step, one line per training seed."""
    problem = _missing(evaluations, ["run_id", "training_seed", "training_step", "suite", "return", "eval_seed"])
    if problem:
        return empty_figure(problem, options)
    rows = evaluations[evaluations["suite"] == "selection"]
    if rows.empty:
        return empty_figure("No selection-suite evaluations.", options)
    fig = go.Figure()
    line_colors = ("#444444", "#7b5ea7", "#17a2b8", "#b07c3b")
    for index, ((run_id, seed), group) in enumerate(rows.groupby(["run_id", "training_seed"], sort=True)):
        stats = group.groupby("training_step")["return"].agg(["mean", "std", "count"]).reset_index()
        color = line_colors[index % len(line_colors)]
        if options.show_ci and (stats["count"] > 1).any():
            spread = stats["std"].fillna(0.0)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([stats["training_step"], stats["training_step"][::-1]]),
                    y=np.concatenate([stats["mean"] + spread, (stats["mean"] - spread)[::-1]]),
                    fill="toself", fillcolor="rgba(120,120,120,0.15)", line={"width": 0},
                    hoverinfo="skip", showlegend=False, name=f"{run_id} ±1 SD",
                )
            )  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=stats["training_step"], y=stats["mean"], mode="lines+markers",
                line={"color": color, "width": 2}, marker={"size": 5},
                name=f"{run_id} (seed {seed})",
                customdata=np.stack([stats["count"], stats["std"].fillna(0.0)], axis=1),
                hovertemplate="step %{x}<br>mean return %{y:.2f}<br>episodes %{customdata[0]}"
                "<br>SD %{customdata[1]:.2f}<extra>" + str(run_id) + "</extra>",
            )
        )  # fmt: skip
        if options.show_raw:
            fig.add_trace(
                go.Scatter(
                    x=group["training_step"], y=group["return"], mode="markers",
                    marker={"color": color, "size": 4, "opacity": 0.35},
                    name="evaluation episodes", showlegend=index == 0,
                    customdata=group[["eval_seed"]].to_numpy(),
                    hovertemplate="step %{x}<br>return %{y:.2f}<br>eval seed %{customdata[0]}<extra></extra>",
                )
            )  # fmt: skip
    if selected is not None and len(selected):
        for capability in _ordered(selected["capability"].tolist(), options.capability_order):
            picks = selected[selected["capability"] == capability]
            fig.add_trace(
                go.Scatter(
                    x=picks["training_step"], y=picks["score"], mode="markers",
                    marker={
                        "symbol": capability_symbol(capability, options), "size": 16,
                        "color": capability_color(capability, options),
                        "line": {"color": "#ffffff", "width": 1.5},
                    },
                    name=f"{capability} checkpoint",
                    customdata=picks[["checkpoint_id", "run_id"]].to_numpy(),
                    hovertemplate=f"{capability}<br>%{{customdata[0]}}<br>step %{{x}}"
                    "<br>selection score %{y:.2f}<extra>%{customdata[1]}</extra>",
                )
            )  # fmt: skip
            for step in picks["training_step"]:
                fig.add_vline(x=float(step), line={"color": capability_color(capability, options), "dash": "dot", "width": 1})
    fig.update_xaxes(title_text="Training step")
    fig.update_yaxes(title_text="Evaluation return (selection seeds; higher = more capable)")
    return _layout(fig, options, title="Evaluation return during training")


def final_performance(evaluations: pd.DataFrame, selected: pd.DataFrame, options: ChartOptions) -> go.Figure:
    """Final-suite return of each selected checkpoint with individual episode points."""
    problem = _missing(evaluations, ["run_id", "suite", "return", "eval_seed", "checkpoint_id"])
    if problem:
        return empty_figure(problem, options)
    rows = evaluations[evaluations["suite"] == "final"].merge(
        selected[["run_id", "checkpoint_id", "capability"]], on=["run_id", "checkpoint_id"]
    )
    if rows.empty:
        return empty_figure("No final-suite evaluations. Run: drl-sim evaluate --config <experiment.yaml>", options)

    fig = go.Figure()
    for capability in _ordered(rows["capability"].tolist(), options.capability_order):
        group = rows[rows["capability"] == capability]
        values = group["return"].to_numpy(dtype=float)
        sem = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        color = capability_color(capability, options)
        if options.show_raw:
            fig.add_trace(
                go.Scatter(
                    x=[capability] * values.size, y=values, mode="markers",
                    marker={"color": color, "size": 7, "opacity": 0.4}, showlegend=False,
                    customdata=group[["eval_seed", "checkpoint_id"]].to_numpy(),
                    hovertemplate="return %{y:.2f}<br>eval seed %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                )
            )  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=[capability], y=[float(values.mean())], mode="markers",
                marker={"color": color, "size": 12, "symbol": "diamond"},
                error_y={"type": "data", "array": [1.96 * sem], "thickness": 1.5, "width": 6} if options.show_ci else None,
                showlegend=False,
                hovertemplate=f"mean %{{y:.2f}} ± 1.96 SEM<br>n = {values.size} episodes<extra>{capability}</extra>",
            )
        )  # fmt: skip
    fig.update_xaxes(title_text="Selected checkpoint", type="category")
    fig.update_yaxes(title_text="Final return (held-out seeds)")
    return _layout(fig, options, title="Final performance (mean ± 1.96 SEM)")


# --------------------------------------------------------------------------- matrix explorer


def _scale_for(metric: str, kind: str | None) -> str:
    return DISTANCE_SCALE if metric_direction(metric, kind).startswith("lower") else SIMILARITY_SCALE


def case_matrix_heatmap(
    cells: pd.DataFrame,
    metric: str,
    options: ChartOptions,
    *,
    selected: tuple[str, str] | None = None,
) -> go.Figure:
    """Case-to-case overview heatmap of one metric (rows × columns of scenarios)."""
    columns = ["row_scenario", "column_scenario", "metric", "estimate", "ci_low", "ci_high", "n_clips", "metric_kind"]
    problem = _missing(cells, columns)
    if problem:
        return empty_figure(problem, options)
    rows = cells[cells["metric"] == metric]
    if rows.empty:
        return empty_figure(f"No {metric_label(metric)} values for this selection.", options)
    labels = _ordered(pd.concat([rows["row_scenario"], rows["column_scenario"]]).tolist(), options.scenario_order)
    pivot = rows.pivot_table(index="row_scenario", columns="column_scenario", values="estimate", aggfunc="first")
    pivot = pivot.reindex(index=labels, columns=labels)
    keyed = rows.set_index(["row_scenario", "column_scenario"])
    custom = np.empty((len(labels), len(labels), 3), dtype=object)
    for i, row_label in enumerate(labels):
        for j, column_label in enumerate(labels):
            if (row_label, column_label) in keyed.index:
                item = keyed.loc[(row_label, column_label)]
                item = item.iloc[0] if isinstance(item, pd.DataFrame) else item
                custom[i, j] = (item["ci_low"], item["ci_high"], item["n_clips"])
            else:
                custom[i, j] = (np.nan, np.nan, 0)
    kind = str(rows["metric_kind"].iloc[0])
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float), x=labels, y=labels, customdata=custom,
            colorscale=_scale_for(metric, kind), texttemplate="%{z:.3f}",
            colorbar={"title": METRIC_SHORT.get(metric, metric)},
            hovertemplate="row %{y} × column %{x}<br>" + metric_label(metric) + " %{z:.4f}"
            "<br>CI [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>independent clips %{customdata[2]}<extra></extra>",
        )
    )  # fmt: skip
    if selected and selected[0] in labels and selected[1] in labels:
        i, j = labels.index(selected[0]), labels.index(selected[1])
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line={"color": "#111111", "width": 3})
    fig.update_xaxes(title_text="Column case", type="category", side="top")
    fig.update_yaxes(title_text="Row case", type="category", autorange="reversed")
    return _layout(fig, options, title=f"{metric_label(metric)} ({metric_direction(metric, kind)})", height=340)


ACCENT_COLOR = "#1f6feb"


def metric_bars(
    values: dict[str, float],
    primary: str,
    options: ChartOptions | None = None,
    *,
    intervals: dict[str, tuple[float, float]] | None = None,
    distance_max: float = 1.0,
) -> go.Figure:
    """One labelled bar per cell metric; bar length is closeness, the primary metric is accented.

    Similarities fill by value on [0, 1]; distances fill by ``1 - value / distance_max``
    so a longer bar always means more alike. The printed number is the raw value.
    """
    options = options or ChartOptions()
    metrics = [metric for metric in METRIC_SHORT if metric in values and np.isfinite(values[metric])]
    if not metrics:
        return empty_figure("n/a", options, height=58)

    intervals = intervals or {}
    labels, fills, colors, texts, hovers = [], [], [], [], []
    for metric in reversed(metrics):  # plotly draws the first category at the bottom
        value = float(values[metric])
        closeness = 1.0 - value / max(distance_max, 1e-9) if metric in DISTANCE_METRICS else value
        hover = f"{metric_label(metric)} {value:.4f} ({metric_direction(metric)})"
        if options.show_ci and metric in intervals and all(np.isfinite(intervals[metric])):
            hover += f"<br>clip-bootstrap interval [{intervals[metric][0]:.4f}, {intervals[metric][1]:.4f}]"
        emphasis = metric == primary
        labels.append(f"<b>{METRIC_SHORT[metric]}</b>" if emphasis else METRIC_SHORT[metric])
        fills.append(float(np.clip(closeness, 0.0, 1.0)))
        colors.append(ACCENT_COLOR if emphasis else NEUTRAL_COLOR)
        texts.append(f"<b>{value:.3f}</b>" if emphasis else f"{value:.3f}")
        hovers.append(hover)

    track = go.Bar(x=[1.0] * len(labels), y=labels, orientation="h", marker={"color": "#eceff3"}, hoverinfo="skip")
    bars = go.Bar(
        x=fills, y=labels, orientation="h", marker={"color": colors},
        hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>",
    )  # fmt: skip
    fig = go.Figure([track, bars])
    for label, text in zip(labels, texts, strict=True):
        fig.add_annotation(x=1.03, y=label, text=text, showarrow=False, xanchor="left", font={"size": 11})
    fig.update_xaxes(range=[0, 1.3], visible=False, fixedrange=True)
    fig.update_yaxes(tickfont={"size": 11}, fixedrange=True)
    fig.update_layout(
        template=options.template, height=22 * len(labels) + 8, margin={"l": 4, "r": 4, "t": 2, "b": 2},
        showlegend=False, barmode="overlay", bargap=0.72,
    )  # fmt: skip
    return fig


def latent_signature(
    vector_a: np.ndarray | None,
    vector_b: np.ndarray | None,
    options: ChartOptions | None = None,
    *,
    labels: tuple[str, str] = ("row frame", "column frame"),
    height: int = 96,
) -> go.Figure:
    """Two stored activation vectors (block-averaged) as aligned heat strips."""
    options = options or ChartOptions()
    if vector_a is None or vector_b is None or len(vector_a) == 0 or len(vector_b) == 0:
        return empty_figure("Latent vectors unavailable.", options, height=height)
    a = np.asarray(vector_a, dtype=float).ravel()
    b = np.asarray(vector_b, dtype=float).ravel()
    size = min(a.size, b.size)
    stacked = np.stack([a[:size], b[:size]])
    limit = float(np.nanmax(np.abs(stacked))) or 1.0
    signed = bool((stacked < 0).any())
    fig = go.Figure(
        go.Heatmap(
            z=stacked, y=list(labels), colorscale=DIVERGING_SCALE if signed else "Blues",
            zmin=-limit if signed else 0.0, zmax=limit, showscale=False,
            hovertemplate="%{y}<br>feature block %{x}<br>activation %{z:.3f}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_xaxes(visible=False)
    fig.update_yaxes(tickfont={"size": 9}, autorange="reversed")
    fig.update_layout(template=options.template, height=height, margin={"l": 70, "r": 4, "t": 4, "b": 4})
    return fig


def clip_pair_distribution(
    pairs: pd.DataFrame,
    metric: str,
    row_scenario: str,
    column_scenario: str,
    options: ChartOptions,
) -> go.Figure:
    """Clip-pair values: within the row case, within the column case, and across them."""
    problem = _missing(pairs, ["scenario_a", "scenario_b", "clip_id_a", "clip_id_b", "value", "n_frame_pairs"])
    if problem:
        return empty_figure(problem, options)
    groups: list[tuple[str, pd.DataFrame, str]] = []
    within_row = pairs[(pairs["scenario_a"] == row_scenario) & (pairs["scenario_b"] == row_scenario)]
    groups.append((f"within {row_scenario}", within_row, scenario_color(row_scenario, options)))
    if column_scenario != row_scenario:
        within_column = pairs[(pairs["scenario_a"] == column_scenario) & (pairs["scenario_b"] == column_scenario)]
        groups.append((f"within {column_scenario}", within_column, scenario_color(column_scenario, options)))
        across = pairs[
            ((pairs["scenario_a"] == row_scenario) & (pairs["scenario_b"] == column_scenario))
            | ((pairs["scenario_a"] == column_scenario) & (pairs["scenario_b"] == row_scenario))
        ]
        groups.append((f"{row_scenario} × {column_scenario}", across, NEUTRAL_COLOR))
    if all(group.empty for _, group, _ in groups):
        return empty_figure("No clip pairs match the current filters.", options)
    fig = go.Figure()
    for name, group, color in groups:
        if group.empty:
            continue
        fig.add_trace(
            go.Violin(
                x=[name] * len(group), y=group["value"], name=f"{name} (n={len(group)} clip pairs)",
                line={"color": color}, fillcolor=color, opacity=0.55, box={"visible": True},
                meanline={"visible": True}, points="all" if options.show_raw else False,
                pointpos=0, jitter=0.35, marker={"size": 4, "opacity": 0.6},
                customdata=group[["clip_id_a", "clip_id_b", "n_frame_pairs"]].to_numpy(),
                hovertemplate="%{customdata[0]} × %{customdata[1]}<br>value %{y:.4f}"
                "<br>frame pairs averaged %{customdata[2]}<extra></extra>",
            )
        )  # fmt: skip
    fig.update_xaxes(title_text="Clip-pair group (pairs of different clips only)", type="category")
    fig.update_yaxes(title_text=f"{metric_label(metric)} ({metric_direction(metric)})")
    return _layout(fig, options, title="Clip-level distribution")


def temporal_lag_curve(rows: pd.DataFrame, options: ChartOptions) -> go.Figure:
    """Temporal cosine versus lag per capability, with the input-pixel control."""
    problem = _missing(rows, ["checkpoint", "lag", "estimate", "ci_low", "ci_high", "input_control", "n_clips"])
    if problem:
        return empty_figure(problem, options)
    fig = go.Figure()
    for capability in _ordered(rows["checkpoint"].tolist(), options.capability_order):
        group = rows[rows["checkpoint"] == capability].sort_values("lag")
        fig.add_trace(
            go.Scatter(
                x=group["lag"], y=group["estimate"], mode="lines+markers",
                line={"color": capability_color(capability, options), "width": 2},
                marker={"symbol": capability_symbol(capability, options), "size": 9},
                error_y=_error_bars(group, options), name=f"{capability} latent",
                customdata=group[["n_clips", "ci_low", "ci_high"]].to_numpy(),
                hovertemplate=f"{capability}<br>lag %{{x}}<br>cosine %{{y:.4f}}"
                "<br>CI [%{customdata[1]:.4f}, %{customdata[2]:.4f}]<br>independent clips %{customdata[0]}<extra></extra>",
            )
        )  # fmt: skip
    control = rows.dropna(subset=["input_control"]).groupby("lag")["input_control"].mean().reset_index()
    if len(control):
        fig.add_trace(
            go.Scatter(
                x=control["lag"], y=control["input_control"], mode="lines+markers",
                line={"color": "#222222", "dash": "dash", "width": 1.5}, marker={"size": 6},
                name="input pixels (control)",
                hovertemplate="lag %{x}<br>input-pixel cosine %{y:.4f}<extra></extra>",
            )
        )  # fmt: skip
    fig.update_xaxes(title_text="Lag (agent steps within a clip)", dtick=1)
    fig.update_yaxes(title_text="Cosine similarity (higher = smoother)")
    return _layout(fig, options, title="Temporal smoothness versus input-space change")


def clip_pair_heatmap(pairs: pd.DataFrame, clips: pd.DataFrame, metric: str, options: ChartOptions) -> go.Figure:
    """All analysed clips × clips for one metric, ordered by scenario with boundaries."""
    problem = _missing(pairs, ["clip_id_a", "clip_id_b", "value"]) or _missing(clips, ["clip_id", "scenario", "clip_index"])
    if problem:
        return empty_figure(problem, options)
    order = clips.copy()
    order["scenario_rank"] = order["scenario"].map(
        {label: rank for rank, label in enumerate(_ordered(order["scenario"].tolist(), options.scenario_order))}
    )
    order = order.sort_values(["scenario_rank", "clip_index"], kind="mergesort")
    truncated = len(order) > options.max_heatmap_samples
    if truncated:
        step = int(np.ceil(len(order) / options.max_heatmap_samples))
        order = order.iloc[::step]
    ids = order["clip_id"].astype(str).tolist()
    position = {clip_id: index for index, clip_id in enumerate(ids)}
    matrix = np.full((len(ids), len(ids)), np.nan)
    used = pairs[pairs["clip_id_a"].isin(position) & pairs["clip_id_b"].isin(position)]
    a = used["clip_id_a"].map(position).to_numpy()
    b = used["clip_id_b"].map(position).to_numpy()
    matrix[a, b] = used["value"].to_numpy(dtype=float)
    matrix[b, a] = used["value"].to_numpy(dtype=float)
    scenarios = order["scenario"].astype(str).tolist()
    custom = np.empty((len(ids), len(ids), 2), dtype=object)
    custom[..., 0] = np.asarray(scenarios, dtype=object)[:, None]
    custom[..., 1] = np.asarray(scenarios, dtype=object)[None, :]
    fig = go.Figure(
        go.Heatmap(
            z=matrix, x=ids, y=ids, customdata=custom, colorscale=_scale_for(metric, None),
            colorbar={"title": METRIC_SHORT.get(metric, metric)}, hoverongaps=False,
            hovertemplate="%{y} (%{customdata[0]}) × %{x} (%{customdata[1]})<br>value %{z:.4f}<extra></extra>",
        )
    )  # fmt: skip
    boundaries = [index for index in range(1, len(scenarios)) if scenarios[index] != scenarios[index - 1]]
    for boundary in boundaries:
        fig.add_shape(type="line", x0=boundary - 0.5, x1=boundary - 0.5, y0=-0.5, y1=len(ids) - 0.5, line={"color": "#111111", "width": 1.5})
        fig.add_shape(type="line", y0=boundary - 0.5, y1=boundary - 0.5, x0=-0.5, x1=len(ids) - 0.5, line={"color": "#111111", "width": 1.5})
    fig.update_xaxes(type="category", showticklabels=len(ids) <= 40, tickangle=60)
    fig.update_yaxes(type="category", showticklabels=len(ids) <= 40, autorange="reversed")
    suffix = f" — showing {len(ids)} evenly spaced clips" if truncated else f" — {len(ids)} clips"
    return _layout(
        fig, options, height=max(options.height, 460),
        title=f"{metric_label(metric)} between clips, ordered by scenario{suffix} (diagonal: same clip, excluded)",
    )  # fmt: skip


def latent_vs_input_change(pairs: pd.DataFrame, options: ChartOptions) -> go.Figure:
    """Latent change (1 − temporal cosine) against input-pixel change, per frame transition."""
    problem = _missing(pairs, ["scenario_a", "clip_id_a", "step_a", "step_b", "value", "input_change", "lag"])
    if problem:
        return empty_figure(problem, options)
    rows = pairs.dropna(subset=["value", "input_change"])
    if rows.empty:
        return empty_figure("No input-change control values are available.", options)
    fig = go.Figure()
    for scenario in _ordered(rows["scenario_a"].tolist(), options.scenario_order):
        group = rows[rows["scenario_a"] == scenario]
        fig.add_trace(
            go.Scatter(
                x=group["input_change"], y=1.0 - group["value"], mode="markers",
                marker={"color": scenario_color(scenario, options), "size": 6, "opacity": 0.65},
                name=f"{scenario} (n={group['clip_id_a'].nunique()} clips)",
                customdata=group[["clip_id_a", "step_a", "step_b", "lag"]].to_numpy(),
                hovertemplate="%{customdata[0]} steps %{customdata[1]}→%{customdata[2]} (lag %{customdata[3]})"
                "<br>input change %{x:.4f}<br>latent change %{y:.4f}<extra></extra>",
            )
        )  # fmt: skip
    fig.update_xaxes(title_text="Input pixel change (mean absolute difference, 0–1)")
    fig.update_yaxes(title_text="Latent change (1 − cosine; higher = larger change)")
    return _layout(fig, options, title="Latent change versus input change (frames within a clip are not independent)")


# --------------------------------------------------------------------------- scenario separation


def metric_by_capability(
    rows: pd.DataFrame,
    options: ChartOptions,
    *,
    metric: str,
    group_column: str = "layer",
    group_order: Sequence[str] = (),
    chance: float | None = None,
    title: str | None = None,
) -> go.Figure:
    """Estimates with clip-bootstrap intervals by capability, one series per group."""
    problem = _missing(rows, ["checkpoint", "estimate", "ci_low", "ci_high", "n_clips", group_column])
    if problem:
        return empty_figure(problem, options)
    capabilities = _ordered(rows["checkpoint"].tolist(), options.capability_order)
    groups = _ordered(rows[group_column].tolist(), group_order)
    fig = go.Figure()
    palette = ("#35495e", "#7b5ea7", "#17a2b8", "#b07c3b", "#c05299", "#5f6b7a", "#8a9a2b", "#d1495b")
    for index, group_name in enumerate(groups):
        group = rows[rows[group_column] == group_name].copy()
        group["rank"] = group["checkpoint"].map({name: rank for rank, name in enumerate(capabilities)})
        group = group.sort_values("rank")
        flags = group["warnings"].fillna("").to_numpy() if "warnings" in group.columns else np.full(len(group), "")
        fig.add_trace(
            go.Scatter(
                x=group["checkpoint"], y=group["estimate"], mode="lines+markers",
                line={"color": palette[index % len(palette)], "width": 1.5},
                marker={"size": 9}, error_y=_error_bars(group, options), name=str(group_name),
                customdata=np.stack([group["ci_low"], group["ci_high"], group["n_clips"], flags], axis=1),
                hovertemplate=f"{group_name}<br>%{{x}}<br>estimate %{{y:.4f}}"
                "<br>CI [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>independent clips %{customdata[2]}"
                "<br>%{customdata[3]}<extra></extra>",
            )
        )  # fmt: skip
    if chance is not None:
        fig.add_hline(y=chance, line={"color": "#888888", "dash": "dash"}, annotation_text=f"chance = {chance:.3f}")
    fig.update_xaxes(title_text="Checkpoint capability", type="category", categoryorder="array", categoryarray=capabilities)
    fig.update_yaxes(title_text=f"{metric_label(metric)} ({metric_direction(metric, 'score')})")
    return _layout(fig, options, title=title or metric_label(metric))


def confusion_matrix(rows: pd.DataFrame, options: ChartOptions) -> go.Figure:
    """Held-out confusion matrix normalised by true class; raw counts on hover."""
    problem = _missing(rows, ["scenario", "scenario_b", "estimate", "n_pairs", "n_clips"])
    if problem:
        return empty_figure(problem, options)
    labels = _ordered(pd.concat([rows["scenario"], rows["scenario_b"]]).tolist(), options.scenario_order)
    rates = rows.pivot_table(index="scenario", columns="scenario_b", values="estimate", aggfunc="first").reindex(index=labels, columns=labels)
    counts = rows.pivot_table(index="scenario", columns="scenario_b", values="n_pairs", aggfunc="first").reindex(index=labels, columns=labels)
    totals = rows.groupby("scenario")["n_clips"].first().reindex(labels)
    custom = np.stack([counts.to_numpy(dtype=float), np.repeat(totals.to_numpy(dtype=float)[:, None], len(labels), axis=1)], axis=-1)
    fig = go.Figure(
        go.Heatmap(
            z=rates.to_numpy(dtype=float), x=labels, y=labels, customdata=custom, zmin=0.0, zmax=1.0,
            colorscale="Blues", texttemplate="%{z:.2f}", colorbar={"title": "rate"},
            hovertemplate="true %{y} → predicted %{x}<br>rate %{z:.3f}"
            "<br>count %{customdata[0]:.0f} of %{customdata[1]:.0f} held-out clips<extra></extra>",
        )
    )  # fmt: skip
    fig.update_xaxes(title_text="Predicted scenario", type="category")
    fig.update_yaxes(title_text="True scenario (rows sum to 1)", type="category", autorange="reversed")
    return _layout(fig, options, title="Held-out nearest-centroid confusion")


def embedding_scatter(rows: pd.DataFrame, options: ChartOptions, *, embedding: str = "pca") -> go.Figure:
    """2-D projection coloured by scenario and symbolised by capability."""
    columns = ["checkpoint", "scenario", "clip_id", "split", "step", "x", "y", "explained_variance_x", "explained_variance_y"]
    problem = _missing(rows, columns)
    if problem:
        return empty_figure(problem, options)
    fig = go.Figure()
    for capability in _ordered(rows["checkpoint"].tolist(), options.capability_order):
        for scenario in _ordered(rows["scenario"].tolist(), options.scenario_order):
            group = rows[(rows["checkpoint"] == capability) & (rows["scenario"] == scenario)]
            if group.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=group["x"], y=group["y"], mode="markers",
                    marker={
                        "color": scenario_color(scenario, options), "symbol": capability_symbol(capability, options),
                        "size": 10, "opacity": 0.8, "line": {"color": "#ffffff", "width": 0.5},
                    },
                    name=f"{scenario} · {capability}",
                    customdata=group[["clip_id", "split", "step"]].to_numpy(),
                    hovertemplate=f"{scenario} · {capability}<br>%{{customdata[0]}} (%{{customdata[1]}})"
                    "<br>x %{x:.3f}, y %{y:.3f}<extra></extra>",
                )
            )  # fmt: skip
    if embedding == "pca" and rows["checkpoint"].nunique() == 1:
        ratio_x = float(rows["explained_variance_x"].iloc[0])
        ratio_y = float(rows["explained_variance_y"].iloc[0])
        fig.update_xaxes(title_text=f"PC 1 ({ratio_x:.1%} variance)")
        fig.update_yaxes(title_text=f"PC 2 ({ratio_y:.1%} variance)" if np.isfinite(ratio_y) else "PC 2")
    elif embedding == "pca":
        fig.update_xaxes(title_text="PC 1 (fitted separately per checkpoint; see table for explained variance)")
        fig.update_yaxes(title_text="PC 2")
    else:
        fig.update_xaxes(title_text=f"{embedding.upper()} 1 (visualisation only)")
        fig.update_yaxes(title_text=f"{embedding.upper()} 2 (visualisation only)")
    return _layout(fig, options, title=f"{embedding.upper()} of clip centroids", height=max(options.height, 440))


def centroid_distance_matrix(rows: pd.DataFrame, options: ChartOptions, *, metric: str = "normalized_euclidean") -> go.Figure:
    """Scenario × scenario matrix (diagonal: within scenario) to show which labels separate."""
    problem = _missing(rows, ["scenario", "scenario_b", "estimate", "ci_low", "ci_high", "n_clips"])
    if problem:
        return empty_figure(problem, options)
    renamed = rows.rename(columns={"scenario": "row_scenario", "scenario_b": "column_scenario"}).assign(metric=metric)
    if "metric_kind" not in renamed.columns:
        renamed["metric_kind"] = "distance" if metric in DISTANCE_METRICS else "similarity"
    fig = case_matrix_heatmap(renamed, metric, options)
    fig.update_layout(title={"text": f"Clip-centroid {metric_label(metric).lower()} between scenarios (diagonal = within scenario)"})
    return fig


# --------------------------------------------------------------------------- cross-checkpoint geometry


def _cross_heatmap(
    rows: pd.DataFrame, x_column: str, y_column: str, x_order: Sequence[str], y_order: Sequence[str],
    metric: str, options: ChartOptions, *, title: str, x_title: str, y_title: str,
) -> go.Figure:  # fmt: skip
    xs = _ordered(rows[x_column].tolist(), x_order)
    ys = _ordered(rows[y_column].tolist(), y_order)
    values = np.full((len(ys), len(xs)), np.nan)
    custom = np.empty((len(ys), len(xs), 5), dtype=object)
    custom[...] = ""
    for _, row in rows.iterrows():
        i, j = ys.index(str(row[y_column])), xs.index(str(row[x_column]))
        if bool(row["applicable"]):
            values[i, j] = float(row["estimate"])
        custom[i, j] = (
            row["ci_low"], row["ci_high"], row["n_clips"], row["n_samples"],
            "" if bool(row["applicable"]) else f"not applicable: {row['reason']}",
        )  # fmt: skip
    kind = str(rows["metric_kind"].iloc[0])
    lowest = float(np.nanmin(values)) if np.isfinite(values).any() else 0.0
    fig = go.Figure(
        go.Heatmap(
            z=values, x=xs, y=ys, customdata=custom, colorscale=_scale_for(metric, kind),
            zmin=-1.0 if lowest < 0 else 0.0, zmax=1.0, texttemplate="%{z:.3f}", hoverongaps=False,
            colorbar={"title": metric_label(metric)},
            hovertemplate="%{y} × %{x}<br>" + metric_label(metric) + " %{z:.4f}"
            "<br>CI [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>independent clips %{customdata[2]}"
            "<br>aligned samples %{customdata[3]}<br>%{customdata[4]}<extra></extra>",
        )
    )  # fmt: skip
    for i, y_value in enumerate(ys):
        for j, x_value in enumerate(xs):
            if np.isnan(values[i, j]) and custom[i, j, 4]:
                fig.add_annotation(x=x_value, y=y_value, text="n/a", showarrow=False, font={"color": "#777777", "size": 11})
    fig.update_xaxes(title_text=x_title, type="category", categoryorder="array", categoryarray=xs)
    fig.update_yaxes(title_text=y_title, type="category", categoryorder="array", categoryarray=ys, autorange="reversed")
    return _layout(fig, options, title=f"{title} ({metric_direction(metric, kind)})", height=max(options.height, 400))


_CROSS_COLUMNS = [
    "checkpoint_a", "checkpoint_b", "layer_a", "layer_b", "metric", "metric_kind", "estimate",
    "ci_low", "ci_high", "n_clips", "n_samples", "applicable", "reason",
]  # fmt: skip


def checkpoint_heatmap(rows: pd.DataFrame, metric: str, options: ChartOptions) -> go.Figure:
    """Checkpoint × checkpoint similarity for one metric at one layer."""
    problem = _missing(rows, _CROSS_COLUMNS)
    if problem:
        return empty_figure(problem, options)
    return _cross_heatmap(
        rows, "checkpoint_b", "checkpoint_a", options.capability_order, options.capability_order, metric, options,
        title=f"{metric_label(metric)} between checkpoints", x_title="Checkpoint", y_title="Checkpoint",
    )  # fmt: skip


def layer_heatmap(rows: pd.DataFrame, metric: str, options: ChartOptions, *, layer_order: Sequence[str] = ()) -> go.Figure:
    """Layer × layer similarity for one checkpoint pair."""
    problem = _missing(rows, _CROSS_COLUMNS)
    if problem:
        return empty_figure(problem, options)
    a = str(rows["checkpoint_a"].iloc[0])
    b = str(rows["checkpoint_b"].iloc[0])
    return _cross_heatmap(
        rows, "layer_b", "layer_a", layer_order, layer_order, metric, options,
        title=f"{metric_label(metric)} across layers: {a} × {b}", x_title=f"Layer ({b})", y_title=f"Layer ({a})",
    )  # fmt: skip


def trajectory_to_reference(rows: pd.DataFrame, metric: str, options: ChartOptions, *, reference: str = "strong") -> go.Figure:
    """Similarity of each checkpoint to the reference checkpoint versus training step."""
    problem = _missing(rows, [*_CROSS_COLUMNS, "training_step_a"])
    if problem:
        return empty_figure(problem, options)
    rows = rows[rows["applicable"].astype(bool)]
    if rows.empty:
        return empty_figure("Metric is not applicable to these layers.", options)
    fig = go.Figure()
    palette = ("#35495e", "#7b5ea7", "#17a2b8", "#b07c3b", "#c05299", "#5f6b7a", "#8a9a2b", "#d1495b")
    for index, (layer, group) in enumerate(rows.groupby("layer_a", sort=False)):
        group = group.sort_values("training_step_a")
        fig.add_trace(
            go.Scatter(
                x=group["training_step_a"], y=group["estimate"], mode="lines+markers",
                line={"color": palette[index % len(palette)], "width": 1.5},
                marker={
                    "size": 11, "symbol": [capability_symbol(c, options) for c in group["checkpoint_a"]],
                    "color": [capability_color(c, options) for c in group["checkpoint_a"]],
                    "line": {"color": palette[index % len(palette)], "width": 1},
                },
                error_y=_error_bars(group, options), name=str(layer),
                customdata=group[["checkpoint_a", "ci_low", "ci_high", "n_clips"]].to_numpy(),
                hovertemplate=f"{layer}<br>%{{customdata[0]}} at step %{{x}}<br>estimate %{{y:.4f}}"
                "<br>CI [%{customdata[1]:.4f}, %{customdata[2]:.4f}]<br>independent clips %{customdata[3]}<extra></extra>",
            )
        )  # fmt: skip
    kind = str(rows["metric_kind"].iloc[0])
    fig.update_xaxes(title_text="Training step of the compared checkpoint")
    fig.update_yaxes(title_text=f"{metric_label(metric)} to {reference} ({metric_direction(metric, kind)})")
    return _layout(fig, options, title=f"Trajectory toward the {reference} checkpoint (the {reference} point is a self-comparison)")


def metric_agreement(rows: pd.DataFrame, options: ChartOptions, *, metrics: Sequence[str], layer_order: Sequence[str] = ()) -> go.Figure:
    """Small multiples: one panel per geometry metric, checkpoint pairs across layers."""
    problem = _missing(rows, _CROSS_COLUMNS)
    if problem:
        return empty_figure(problem, options)
    present = [metric for metric in metrics if metric in set(rows["metric"])]
    if not present:
        return empty_figure("No geometry metrics are available.", options)
    fig = make_subplots(
        rows=1, cols=len(present), shared_yaxes=False,
        subplot_titles=[f"{metric_label(metric)}<br><sup>{metric_direction(metric, str(rows[rows['metric'] == metric]['metric_kind'].iloc[0]))}</sup>" for metric in present],
    )  # fmt: skip
    pair_colors = ("#35495e", "#d1495b", "#3a9d5d", "#7b5ea7", "#b07c3b", "#17a2b8")
    order = list(options.capability_order)
    pairs = rows[["checkpoint_a", "checkpoint_b"]].drop_duplicates()
    pair_names = sorted(
        {(a, b) for a, b in pairs.itertuples(index=False) if a != b and (order.index(a) if a in order else 99) < (order.index(b) if b in order else 99)},
        key=lambda pair: tuple(order.index(item) if item in order else 99 for item in pair),
    )
    for column, metric in enumerate(present, start=1):
        subset = rows[(rows["metric"] == metric) & rows["applicable"].astype(bool)]
        layers = _ordered(subset["layer_a"].tolist(), layer_order)
        for index, (a, b) in enumerate(pair_names):
            group = subset[(subset["checkpoint_a"] == a) & (subset["checkpoint_b"] == b)].copy()
            if group.empty:
                continue
            group["rank"] = group["layer_a"].map({name: rank for rank, name in enumerate(layers)})
            group = group.sort_values("rank")
            fig.add_trace(
                go.Scatter(
                    x=group["layer_a"], y=group["estimate"], mode="lines+markers",
                    line={"color": pair_colors[index % len(pair_colors)], "width": 1.5}, marker={"size": 7},
                    error_y=_error_bars(group, options), name=f"{a} × {b}", legendgroup=f"{a}-{b}",
                    showlegend=column == 1,
                    customdata=group[["ci_low", "ci_high", "n_clips"]].to_numpy(),
                    hovertemplate=f"{a} × {b}<br>%{{x}}<br>estimate %{{y:.4f}}"
                    "<br>CI [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>independent clips %{customdata[2]}<extra></extra>",
                ),
                row=1, col=column,
            )  # fmt: skip
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=layers, tickangle=45, row=1, col=column)
        fig.update_yaxes(range=[-0.02, 1.02], row=1, col=column)
    return _layout(fig, options, title="Metric agreement across layers (same layer, different checkpoints)", height=max(options.height, 420))


# --------------------------------------------------------------------------- scenario explorer and provenance


def clip_timeline(
    *, start_step: int, anchor_step: int, end_step: int, burn_in_steps: int, current_step: int,
    anchor_event: str, scenario: str, options: ChartOptions,
) -> go.Figure:  # fmt: skip
    """Source-episode timeline of one clip: burn-in, clip window, anchor, current frame."""
    fig = go.Figure()
    color = scenario_color(scenario, options)
    if burn_in_steps > 0:
        fig.add_shape(type="rect", x0=start_step - burn_in_steps - 0.5, x1=start_step - 0.5, y0=0.25, y1=0.75, fillcolor="#dddddd", line={"width": 0})
        fig.add_annotation(x=start_step - burn_in_steps / 2 - 0.5, y=0.9, text="burn-in", showarrow=False, font={"size": 10, "color": "#666666"})
    fig.add_shape(type="rect", x0=start_step - 0.5, x1=end_step + 0.5, y0=0.25, y1=0.75, fillcolor=color, opacity=0.35, line={"color": color, "width": 1})
    fig.add_annotation(x=(start_step + end_step) / 2, y=0.9, text=f"clip window ({end_step - start_step + 1} steps)", showarrow=False, font={"size": 10})
    fig.add_trace(
        go.Scatter(
            x=[anchor_step], y=[0.5], mode="markers", marker={"symbol": "x", "size": 13, "color": "#111111"},
            name=f"anchor: {anchor_event}", hovertemplate=f"anchor event {anchor_event}<br>episode step %{{x}}<extra></extra>",
        )
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=[current_step], y=[0.5], mode="markers",
            marker={"symbol": "line-ns", "size": 26, "line": {"color": "#d1495b", "width": 3}},
            name="current frame", hovertemplate="current frame<br>episode step %{x}<extra></extra>",
        )
    )  # fmt: skip
    low = min(start_step - burn_in_steps, anchor_step) - 1.5
    high = max(end_step, anchor_step) + 1.5
    fig.update_xaxes(title_text="Agent step in source episode", range=[low, high])
    fig.update_yaxes(visible=False, range=[0, 1.1])
    return _layout(fig, options, height=190)


def artifact_graph(nodes: Sequence[tuple[str, str, str]], edges: Sequence[tuple[str, str]], options: ChartOptions) -> go.Figure:
    """Left-to-right dependency graph. ``nodes`` are ``(id, label, status)`` in pipeline order."""
    if not nodes:
        return empty_figure("No artifacts to show.", options)
    colors = {"ok": "#3a9d5d", "warning": "#e09f3e", "error": "#d1495b", "info": "#4c78a8"}
    position = {node_id: (index, 0.0 if index % 2 == 0 else 0.35) for index, (node_id, _, _) in enumerate(nodes)}
    fig = go.Figure()
    for source, target in edges:
        if source in position and target in position:
            (x0, y0), (x1, y1) = position[source], position[target]
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowwidth=1.2, arrowcolor="#888888", standoff=16, startstandoff=16,
            )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=[position[node_id][0] for node_id, _, _ in nodes],
            y=[position[node_id][1] for node_id, _, _ in nodes],
            mode="markers+text", text=[label for _, label, _ in nodes], textposition="top center",
            marker={"size": 26, "color": [colors.get(status, NEUTRAL_COLOR) for _, _, status in nodes], "line": {"color": "#ffffff", "width": 2}},
            customdata=[[node_id, status] for node_id, _, status in nodes],
            hovertemplate="%{text}<br>%{customdata[0]}<br>status: %{customdata[1]}<extra></extra>",
            showlegend=False,
        )
    )  # fmt: skip
    fig.update_xaxes(visible=False, range=[-0.6, len(nodes) - 0.4])
    fig.update_yaxes(visible=False, range=[-0.4, 0.9])
    return _layout(fig, options, height=240)
