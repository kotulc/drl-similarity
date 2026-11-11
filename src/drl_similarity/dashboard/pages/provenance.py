"""Page 6 — make every figure auditable: configuration, checksums, versions, and exports."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from drl_similarity.dashboard import charts
from drl_similarity.dashboard import components as ui
from drl_similarity.dashboard import data as dd
from drl_similarity.dashboard.data import DashboardData


TABLES: tuple[tuple[str, str], ...] = (
    ("summaries", "Summaries (within/between, separation, temporal)"),
    ("cross_checkpoint", "Cross-checkpoint geometry"),
    ("embeddings", "Embeddings"),
    ("frame_pairs", "Clip pairs and frame pairs"),
)


def _checksums(data: DashboardData) -> pd.DataFrame:
    rows = []
    for name, manifest in data.manifests.items():
        if "inputs" not in manifest:
            continue
        for kind in ("inputs", "outputs"):
            for item in manifest.get(kind, []):
                rows.append(
                    {
                        "artifact": name,
                        "direction": kind[:-1],
                        "role": item.get("role"),
                        "path": item.get("path"),
                        "sha256": item.get("sha256"),
                        "bytes": item.get("bytes") or None,
                    }
                )
    for row in data.layers.drop_duplicates(subset=["run_id", "capability"]).itertuples():
        rows.append(
            {
                "artifact": "checkpoint",
                "direction": "input",
                "role": f"{row.capability} checkpoint",
                "path": f"{row.run_id}/{row.checkpoint_id}",
                "sha256": row.checkpoint_sha256,
                "bytes": None,
            }
        )
    return pd.DataFrame(rows, columns=["artifact", "direction", "role", "path", "sha256", "bytes"])


def _versions(data: DashboardData) -> pd.DataFrame:
    rows = []
    for name, manifest in data.manifests.items():
        if "package_versions" not in manifest:
            continue
        rows.append({"artifact": name, "component": "git commit", "version": manifest.get("git_commit") or "not recorded"})
        rows.append({"artifact": name, "component": "created", "version": manifest.get("created_at")})
        for package, version in sorted((manifest.get("package_versions") or {}).items()):
            rows.append({"artifact": name, "component": package, "version": version})
    return pd.DataFrame(rows, columns=["artifact", "component", "version"])


def _filtered(data: DashboardData, table: str, filters: ui.Filters) -> pd.DataFrame:
    frame = getattr(data, table)
    if frame.empty:
        return frame
    if table == "cross_checkpoint":
        return frame[
            (frame["run_id_a"] == filters.run_id)
            & ((frame["layer_a"] == filters.layer) | (frame["layer_b"] == filters.layer))
            & ((frame["checkpoint_a"] == filters.checkpoint) | (frame["checkpoint_b"] == filters.checkpoint))
        ]
    return dd.select_rows(frame, run_id=filters.run_id, checkpoint=filters.checkpoint, layer=filters.layer)


def render(data: DashboardData) -> None:
    ui.header(
        data,
        "Provenance and exports",
        "Where does every number come from, and how can it be regenerated?",
    )
    filters = ui.sidebar_filters(data, show=("checkpoint", "layer"))
    study = data.study
    options = ui.chart_options(data)

    st.markdown("#### Artifact dependency graph")
    has_activations = study.activations_dir is not None
    nodes = [
        ("configs", "YAML configs", "ok"),
        ("run", "training run", "ok"),
        ("selected-checkpoints.yaml", "selected checkpoints", "ok"),
        ("scenario manifest + observations.npz", "scenario clips", "ok"),
        ("activations/", "activations", "ok" if has_activations else "warning"),
        (f"{study.metrics_version}/", "metrics", "ok"),
        ("this page", "dashboard", data.status),
    ]
    edges = [("configs", "run"), ("run", "selected-checkpoints.yaml"), ("selected-checkpoints.yaml", "scenario manifest + observations.npz"),
             ("scenario manifest + observations.npz", "activations/"), ("selected-checkpoints.yaml", "activations/"),
             ("activations/", f"{study.metrics_version}/"), (f"{study.metrics_version}/", "this page")]  # fmt: skip
    st.plotly_chart(charts.artifact_graph(nodes, edges, options), use_container_width=True, key="artifact_graph", config={"displayModeBar": False})

    st.markdown("#### Regeneration commands")
    config = study.config_path or "<study.yaml>"
    st.code(
        "\n".join(
            [
                "drl-sim train --config <experiment.yaml>",
                "drl-sim select-checkpoints --config <experiment.yaml>",
                f"drl-sim collect-scenarios --config {config}",
                f"drl-sim validate-scenarios --config {config}",
                f"drl-sim extract --config {config}",
                f"drl-sim analyze --config {config}",
                f"drl-sim dashboard --config {config}",
            ]
        ),
        language="bash",
    )
    if study.synthetic:
        st.info(
            "This study was produced by `python scripts/generate_fixture_study.py`. It has no trained "
            "checkpoints, so `train`, `collect-scenarios`, and `extract` would replace it with real data.",
            icon="🧪",
        )

    st.markdown("#### Checksums")
    checksums = _checksums(data)
    st.dataframe(checksums, hide_index=True, use_container_width=True)
    st.caption(
        f"config hash `{study.config_hash}` · metrics manifest sha256 `{study.metrics_manifest_sha256}`. "
        "All metrics inputs and outputs were re-hashed when the dashboard loaded."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Versions")
        st.dataframe(_versions(data), hide_index=True, use_container_width=True)
        st.caption("Versions are those recorded in each artifact manifest when it was generated.")
    with right:
        st.markdown("#### Scenario review and validation")
        review = data.clips.groupby(["scenario", "review_status"]).size().rename("clips").reset_index()
        st.dataframe(review, hide_index=True, use_container_width=True)
        ui.quality_table(data)

    st.markdown("#### Configuration")
    tabs = st.tabs(["Resolved study config", *[name for name, _ in data.source_configs]])
    with tabs[0]:
        st.code(data.resolved_yaml, language="yaml")
    for tab, (_, text) in zip(tabs[1:], data.source_configs, strict=True):
        with tab:
            st.code(text, language="yaml")
    st.caption("Absolute user paths are redacted to `<repo>` and `<home>`.")

    st.markdown("#### Downloads")
    if not data.dashboard.downloads.enabled:
        ui.empty_state("Downloads are disabled in the dashboard configuration.")
    else:
        table_key = st.selectbox("Table", [key for key, _ in TABLES], format_func=dict(TABLES).get, key="download_table")
        frame = _filtered(data, str(table_key), filters)
        st.caption(f"{len(frame):,} rows for run `{filters.run_id}`, checkpoint `{filters.checkpoint}`, layer `{filters.layer}`.")
        st.dataframe(frame.head(200), hide_index=True, use_container_width=True)
        buttons = st.columns(4)
        if "csv" in data.dashboard.downloads.formats:
            buttons[0].download_button(
                "Filtered rows (CSV)", dd.to_csv_bytes(frame), file_name=f"{study.study_id}-{table_key}.csv",
                mime="text/csv", key="download_rows", disabled=frame.empty,
            )  # fmt: skip
        buttons[1].download_button(
            "Resolved config (YAML)", data.resolved_yaml, file_name=f"{study.study_id}-resolved-config.yaml",
            mime="text/yaml", key="download_config",
        )  # fmt: skip
        buttons[2].download_button(
            "Metrics manifest (JSON)", dd.manifest_json(data.manifests["metrics"]),
            file_name=f"{study.study_id}-metrics-manifest.json", mime="application/json", key="download_manifest",
        )  # fmt: skip
        if "activations" in data.manifests:
            buttons[3].download_button(
                "Activations manifest (JSON)", dd.manifest_json(data.manifests["activations"]),
                file_name=f"{study.study_id}-activations-manifest.json", mime="application/json",
                key="download_activations_manifest",
            )  # fmt: skip
        st.caption("Checkpoints and activation arrays are not downloadable from the dashboard. Chart PNGs are available under each chart.")
