"""Page 4 — do scenarios separate in latent space, and does separation grow with capability?"""

from __future__ import annotations

import json

import streamlit as st

from drl_similarity.dashboard import charts
from drl_similarity.dashboard import components as ui
from drl_similarity.dashboard import data as dd
from drl_similarity.dashboard.data import DashboardData
from drl_similarity.tables import ALL_SCENARIOS


def render(data: DashboardData) -> None:
    ui.header(
        data,
        "Scenario separation",
        "Are clips from the same scenario closer to each other than to other scenarios, and does that grow with capability?",
    )
    filters = ui.sidebar_filters(data, show=("checkpoint", "layer", "pair_set", "ci", "raw"))
    options = ui.chart_options(data)
    layers = data.study.layers
    run = dd.select_rows(data.summaries, run_id=filters.run_id)
    overall = dd.select_rows(run, scope="separation", scenario=ALL_SCENARIOS)
    n_labels = max(len(data.study.labels), 1)

    st.markdown("#### Separation by capability")
    st.caption("One line per layer; x-axis ordered weak → medium → strong. Error bars are clip-bootstrap percentile intervals.")
    left, middle, right = st.columns(3)
    with left:
        gap = dd.select_rows(overall, metric="within_between_gap", pair_set=filters.pair_set)
        ui.plot(
            charts.metric_by_capability(gap, options, metric="within_between_gap", group_order=layers, title=f"Within − between gap ({filters.pair_set})"),
            key="gap_by_capability", data=data, frame=gap, name="within-between-gap",
        )  # fmt: skip
        st.caption("Mean within-scenario similarity minus mean between-scenario similarity, from matched pair definitions. Above 0 means scenarios separate.")
    with middle:
        silhouette = dd.select_rows(overall, metric="silhouette")
        ui.plot(
            charts.metric_by_capability(silhouette, options, metric="silhouette", group_order=layers),
            key="silhouette_by_capability", data=data, frame=silhouette, name="silhouette",
        )  # fmt: skip
        st.caption("Silhouette of standardized clip centroids, range −1 to 1. Resampled clips are never compared with their own duplicates.")
    with right:
        accuracy = dd.select_rows(overall, metric="nearest_centroid_accuracy")
        ui.plot(
            charts.metric_by_capability(accuracy, options, metric="nearest_centroid_accuracy", group_order=layers, chance=1.0 / n_labels),
            key="accuracy_by_capability", data=data, frame=accuracy, name="nearest-centroid-accuracy",
        )  # fmt: skip
        held_out = int(accuracy["n_clips"].iloc[0]) if len(accuracy) else 0
        st.caption(f"Centroids fitted on training clips, scored on {held_out} held-out clips. Dashed line: chance for {n_labels} balanced labels.")
    if held_out and held_out < 30:
        st.warning(
            f"Only {held_out} held-out clips: accuracy moves in steps of {1 / held_out:.2f} and its interval is wide.",
            icon="⚠️",
        )

    st.divider()
    st.markdown(f"#### {filters.checkpoint} checkpoint · layer {filters.layer}")
    selected = dd.select_rows(run, checkpoint=filters.checkpoint, layer=filters.layer)
    left, right = st.columns(2)
    with left:
        confusion = dd.select_rows(selected, metric="nearest_centroid_accuracy", scope="confusion")
        ui.plot(charts.confusion_matrix(confusion, options), key="confusion", data=data, frame=confusion, name="confusion")
    with right:
        metric = "normalized_euclidean" if "normalized_euclidean" in data.study.metrics else "cosine"
        distances = dd.select_rows(
            selected, metric=metric, pair_set="clip_centroids", scope=["within_scenario", "between_scenario"]
        )
        ui.plot(charts.centroid_distance_matrix(distances, options, metric=metric), key="centroid_distance", data=data, frame=distances, name="centroid-distance")
        if distances.empty:
            st.caption("Requires the `clip_centroids` pair set in the study configuration.")

    st.divider()
    embeddings = dd.select_rows(data.embeddings, run_id=filters.run_id, layer=filters.layer, level="clip_centroid")
    available = [name for name in ("pca", "umap") if name in set(embeddings["embedding"])] if len(embeddings) else []
    if not available:
        ui.empty_state("No embeddings were computed for this study.")
    for name in available:
        rows = dd.select_rows(embeddings, embedding=name, checkpoint=filters.checkpoint)
        st.markdown(f"#### {name.upper()} of clip centroids — {filters.checkpoint}")
        if name == "umap":
            st.warning(
                "UMAP is for visualisation only. Distances, cluster sizes, and gaps in this plot are not "
                "quantitative evidence; use the separation scores above.",
                icon="⚠️",
            )
        ui.plot(charts.embedding_scatter(rows, options, embedding=name), key=f"embedding_{name}", data=data, frame=rows, name=f"{name}-centroids")
        if len(rows):
            parameters = json.loads(str(rows["parameters"].iloc[0]))
            st.caption(
                f"Parameters: {parameters}. The projection is fitted on all frames of this checkpoint and layer; "
                "each point is the mean projected position of one clip. Projections of different checkpoints are "
                "fitted separately, so their axes are not comparable."
            )

    ui.provenance_expander(
        data,
        [
            "train/test split is by whole clip and fixed in the scenario manifest; it cannot be changed here",
            "standardization and nearest-centroid fitting use training clips only",
            f"split counts: {data.clips[data.clips['analysed']].groupby('split').size().to_dict()}",
        ],
    )
