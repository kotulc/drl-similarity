"""Page 5 — representational geometry between checkpoints, without assuming aligned neurons."""

from __future__ import annotations

import streamlit as st

from drl_similarity.dashboard import charts
from drl_similarity.dashboard import components as ui
from drl_similarity.dashboard import data as dd
from drl_similarity.dashboard.data import DashboardData
from drl_similarity.tables import RAW_COSINE_DIAGNOSTIC


GEOMETRY_METRICS: tuple[str, ...] = ("linear_cka", "rsa_spearman", "svcca", "orthogonal_procrustes")


def _choice(label: str, options: list[str], key: str, default: str | None = None) -> str:
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]
    return str(st.selectbox(label, options, key=key))


def render(data: DashboardData) -> None:
    ui.header(
        data,
        "Cross-checkpoint geometry",
        "How much does representational geometry change between weak, medium, and strong checkpoints?",
    )
    cross = data.cross_checkpoint
    available = tuple(metric for metric in GEOMETRY_METRICS if len(cross) and metric in set(cross["metric"]))
    if not available:
        ui.sidebar_filters(data, show=())
        ui.empty_state("No cross-checkpoint geometry metrics were computed for this study.")
        return
    filters = ui.sidebar_filters(
        data, show=("layer", "ci"), metrics=available, metric_key="cross_metric", metric_title="Geometry metric"
    )
    metric = str(filters.metric)
    options = ui.chart_options(data)
    primary = cross[(cross["run_id_a"] == filters.run_id) & (cross["run_id_b"] == filters.run_id) & ~cross["diagnostic"].astype(bool)]
    rows = primary[primary["metric"] == metric]
    sample_units = sorted(rows["sample_unit"].unique()) if len(rows) else []
    hashes = rows["sample_ids_sha256"].nunique() if len(rows) else 0
    if hashes > 1:
        st.error(
            "Comparisons in this table use different ordered sample IDs. Alignment is invalid; "
            "regenerate with: drl-sim analyze --config <study.yaml> --force",
            icon="❌",
        )
        return
    st.caption(
        f"Samples: `{', '.join(sample_units)}` from {int(rows['n_clips'].iloc[0]) if len(rows) else 0} clips, "
        "identical and identically ordered for every checkpoint (verified by sample-ID checksum). "
        "Frames within a clip are not independent; intervals resample whole clips."
    )

    left, right = st.columns(2)
    with left:
        same_layer = rows[(rows["layer_a"] == filters.layer) & (rows["layer_b"] == filters.layer)]
        ui.plot(charts.checkpoint_heatmap(same_layer, metric, options), key="checkpoint_heatmap", data=data, frame=same_layer, name="checkpoint-heatmap")
        st.caption(f"Layer `{filters.layer}`. Diagonal cells compare a checkpoint with itself.")
    with right:
        capabilities = list(options.capability_order)
        pick = st.columns(2)
        with pick[0]:
            checkpoint_a = _choice("Checkpoint (rows)", capabilities, "cross_checkpoint_a", capabilities[0])
        with pick[1]:
            checkpoint_b = _choice("Checkpoint (columns)", capabilities, "cross_checkpoint_b", capabilities[-1])
        pair = rows[(rows["checkpoint_a"] == checkpoint_a) & (rows["checkpoint_b"] == checkpoint_b)]
        ui.plot(charts.layer_heatmap(pair, metric, options, layer_order=data.study.layers), key="layer_heatmap", data=data, frame=pair, name="layer-heatmap")
        not_applicable = pair[~pair["applicable"].astype(bool)] if len(pair) else pair
        if len(not_applicable):
            st.caption(f"`n/a`: {not_applicable['reason'].iloc[0]} ({len(not_applicable)} layer pairs).")

    ui.metric_note(data, metric)
    flagged = rows[rows["warnings"].fillna("").str.contains("estimate_outside_interval")] if len(rows) else rows
    if len(flagged):
        st.warning(
            f"{len(flagged)} of {len(rows)} {charts.metric_label(metric)} rows have a point estimate outside "
            "their own bootstrap interval. Resampling clips with replacement leaves fewer distinct samples, "
            "which inflates this metric; treat those intervals as indicative only.",
            icon="⚠️",
        )

    st.divider()
    reference = options.capability_order[-1]
    trajectory = rows[(rows["checkpoint_b"] == reference) & (rows["layer_a"] == rows["layer_b"])]
    ui.plot(charts.trajectory_to_reference(trajectory, metric, options, reference=reference), key="trajectory", data=data, frame=trajectory, name="trajectory")
    if len(trajectory) and trajectory["checkpoint_a"].nunique() <= len(options.capability_order):
        st.caption("Only the selected weak, medium, and strong checkpoints were extracted, so the trajectory has one point per capability.")

    st.divider()
    same = primary[primary["layer_a"] == primary["layer_b"]]
    ui.plot(
        charts.metric_agreement(same, options, metrics=GEOMETRY_METRICS, layer_order=data.study.layers),
        key="metric_agreement", data=data, frame=same, name="metric-agreement",
    )  # fmt: skip
    st.caption("CKA, RSA, and SVCCA are similarities; Procrustes is a disparity (lower = more similar). Agreement across metrics is stronger evidence than any one of them.")

    st.divider()
    st.markdown("#### Raw-coordinate cosine (diagnostic)")
    if st.toggle("Show raw-coordinate diagnostic", value=False, key="cross_show_raw_cosine"):
        st.warning(
            "Basis-dependent diagnostic. Networks can rotate or permute their latent basis during training, "
            "so a low raw-coordinate cosine does not mean the geometry differs. Do not use it as a result.",
            icon="⚠️",
        )
        diagnostic = cross[
            (cross["run_id_a"] == filters.run_id) & (cross["run_id_b"] == filters.run_id)
            & (cross["metric"] == RAW_COSINE_DIAGNOSTIC) & (cross["layer_a"] == filters.layer) & (cross["layer_b"] == filters.layer)
        ]  # fmt: skip
        ui.plot(charts.checkpoint_heatmap(diagnostic, RAW_COSINE_DIAGNOSTIC, options), key="raw_cosine", data=data, frame=diagnostic, name="raw-coordinate-cosine")

    ui.provenance_expander(
        data,
        [
            f"cross-checkpoint samples: {data.manifests['metrics']['metadata'].get('cross_checkpoint_samples')}",
            "intervals are computed for same-layer comparisons between different checkpoints; the layer-by-layer grid and self-comparisons are point estimates",
            "a metric that is not applicable to a layer pair is stored with its reason and shown as n/a",
            f"regenerate with `{dd.ANALYZE_COMMAND} --force`",
        ],
    )
