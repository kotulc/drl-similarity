"""Page 3 — do the compared checkpoints differ in capability, and are the artifacts valid?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from drl_similarity.dashboard import charts
from drl_similarity.dashboard import components as ui
from drl_similarity.dashboard import data as dd
from drl_similarity.dashboard.data import DashboardData


def _cards(data: DashboardData, run: pd.Series) -> None:
    study = data.study
    cards = st.columns(6)
    cards[0].metric("Algorithm", str(run["adapter"]), help=str(run["algorithm_id"]))
    cards[1].metric("Policy", str(run["policy"]))
    cards[2].metric("Environment", str(run["env_id"]), help=f"observation fingerprint {run['observation_fingerprint']}")
    cards[3].metric("Clips per scenario", " / ".join(str(count) for _, count in study.clips_per_label), help=", ".join(label for label, _ in study.clips_per_label))
    cards[4].metric("Layers", len(study.layers), help=", ".join(study.layers))
    cards[5].metric("Generated", study.generated_at[:10], help=study.generated_at)


def _flow(data: DashboardData) -> None:
    study = data.study
    has_activations = study.activations_dir is not None
    nodes = [
        ("config", "study config", "ok"),
        ("training", "training run", "ok"),
        ("selection", "checkpoint selection", "ok"),
        ("scenarios", "scenario clips", "ok"),
        ("activations", "activations", "ok" if has_activations else "warning"),
        ("metrics", "metrics", "ok" if data.status != "error" else "error"),
        ("dashboard", "dashboard", data.status),
    ]
    edges = [(nodes[index][0], nodes[index + 1][0]) for index in range(len(nodes) - 1)]
    st.plotly_chart(
        charts.artifact_graph(nodes, edges, ui.chart_options(data)),
        use_container_width=True, key="flow_status", config={"displayModeBar": False},
    )  # fmt: skip
    st.caption(
        "Green stages were found and passed validation when this dashboard loaded. "
        "The dashboard cannot start with a missing or corrupt metrics artifact."
    )


def render(data: DashboardData) -> None:
    ui.header(
        data,
        "Training and quality overview",
        "Do the weak, medium, and strong checkpoints really differ in capability, and can the artifacts be trusted?",
    )
    filters = ui.sidebar_filters(data, show=("ci", "raw"))
    options = ui.chart_options(data)
    run_rows = data.runs[data.runs["run_id"] == filters.run_id]
    if run_rows.empty:
        ui.empty_state("No run metadata is available.")
        return
    run = run_rows.iloc[0]
    _cards(data, run)
    ui.page_link("similarity_matrix", "Back to the similarity matrix explorer", icon="🔷")

    evaluations = data.evaluations[data.evaluations["run_id"] == filters.run_id]
    selected = data.selected[data.selected["run_id"] == filters.run_id]
    left, right = st.columns([2, 1])
    with left:
        ui.plot(charts.learning_curve(evaluations, selected, options), key="learning_curve", data=data, frame=evaluations, name="evaluations")
    with right:
        ui.plot(charts.final_performance(evaluations, selected, options), key="final_performance", data=data, frame=evaluations[evaluations["suite"] == "final"], name="final-evaluations")
    st.caption(
        f"Selection seeds: {run['selection_seeds']} · final seeds: {run['final_seeds']} (disjoint). "
        "Shaded band: ±1 SD across evaluation episodes at each checkpoint — not variation across training seeds."
    )

    st.markdown("#### Checkpoint selection")
    rules = {"weak": run["weak_rule"], "medium": run["medium_rule"], "strong": run["strong_rule"]}
    table = selected.assign(rule=selected["capability"].map(rules))
    order = {name: rank for rank, name in enumerate(options.capability_order)}
    table = table.sort_values("capability", key=lambda column: column.map(order))
    st.dataframe(
        table[["capability", "checkpoint_id", "training_step", "score", "uncertainty", "rule"]].rename(
            columns={"score": "mean selection return", "uncertainty": "SD across episodes"}
        ),
        hide_index=True, use_container_width=True,
    )  # fmt: skip
    evidence = (data.selection.get(filters.run_id) or {}).get("evidence") or {}
    if evidence:
        st.caption(
            f"Evidence: {evidence.get('n_checkpoints')} evaluated checkpoints · strong − weak improvement "
            f"{float(evidence.get('improvement', float('nan'))):.2f} (minimum required "
            f"{evidence.get('minimum_strong_improvement')})."
        )

    st.markdown("#### Study flow")
    _flow(data)

    st.markdown("#### Data-quality checks")
    ui.quality_table(data)
    ui.provenance_expander(
        data,
        [
            f"run `{run['run_id']}` in `{run['run_dir']}` · training seed {int(run['training_seed'])}",
            f"runtime environment fingerprint `{run['runtime_fingerprint']}`",
            f"evaluation table: {dd.select_rows(data.evaluations, run_id=filters.run_id).shape[0]} episode rows",
        ],
    )
