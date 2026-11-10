"""Streamlit entry point: page registration, study loading, and the fatal error boundary.

Launched by ``drl-sim dashboard --config <study.yaml>``, which validates the
artifacts and passes the study config path in ``DRL_SIM_STUDY_CONFIG``. The app
is read-only: it loads published artifacts and never trains, extracts, or
computes study metrics.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path

import streamlit as st


# ``streamlit run app.py`` executes this file as a script; make the package importable
# when it is run from a source checkout without an installed distribution.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drl_similarity.config import load_config  # noqa: E402
from drl_similarity.dashboard import components as ui  # noqa: E402
from drl_similarity.dashboard.data import (  # noqa: E402
    DashboardData,
    load_dashboard_data,
    metrics_cache_key,
)
from drl_similarity.dashboard.pages import PAGE_ORDER  # noqa: E402
from drl_similarity.errors import DrlSimilarityError  # noqa: E402


CONFIG_ENV = "DRL_SIM_STUDY_CONFIG"


@st.cache_data(show_spinner="Loading study artifacts…")
def _load(config_path: str, manifest_checksum: str) -> DashboardData:
    """Cached by config path *and* metrics-manifest checksum, never by path alone."""
    del manifest_checksum
    return load_dashboard_data(load_config(config_path, expected_kind="study"))


def _fatal(title: str, exc: Exception | None = None, hint: str | None = None) -> None:
    st.error(title, icon="❌")
    if exc is not None:
        st.code(str(exc), language="text")
        context = getattr(exc, "context", None)
        if context:
            st.json(context)
    if hint:
        st.info(hint)
    st.stop()


def _page(module_name: str, data: DashboardData) -> Callable[[], None]:
    def run() -> None:
        module = importlib.import_module(f"drl_similarity.dashboard.pages.{module_name}")
        try:
            module.render(data)
        except DrlSimilarityError as exc:
            st.error(f"This page could not be rendered: {exc}", icon="❌")
            context = getattr(exc, "context", None)
            if context:
                st.json(context)

    run.__name__ = f"page_{module_name}"
    return run


def main() -> None:
    st.set_page_config(page_title="DRL representation similarity", page_icon="🔷", layout="wide")
    config_path = os.environ.get(CONFIG_ENV)
    if not config_path:
        _fatal(
            f"{CONFIG_ENV} is not set.",
            hint="Launch the dashboard with: drl-sim dashboard --config <study.yaml>",
        )
    assert config_path is not None
    try:
        resolved = load_config(config_path, expected_kind="study")
        data = _load(str(Path(config_path).resolve()), metrics_cache_key(resolved))
    except DrlSimilarityError as exc:
        _fatal(
            "The study artifacts could not be loaded.",
            exc,
            hint="Regenerate the missing stage(s) with the command shown above, or run: "
            "drl-sim run-study --config <study.yaml>",
        )
        return

    ui.ensure_state(data)
    default_page = data.dashboard.default_page
    known = {page_id for page_id, _ in PAGE_ORDER}
    if default_page not in known:
        st.session_state.setdefault(ui.WARNINGS_KEY, [])
        message = f"default_page={default_page!r} is not a dashboard page; using {PAGE_ORDER[0][0]!r}."
        if message not in st.session_state[ui.WARNINGS_KEY]:
            st.session_state[ui.WARNINGS_KEY].append(message)
        default_page = PAGE_ORDER[0][0]
    pages = {
        page_id: st.Page(_page(page_id, data), title=title, url_path=page_id.replace("_", "-"), default=page_id == default_page)
        for page_id, title in PAGE_ORDER
    }
    st.session_state[ui.PAGES_KEY] = pages
    with st.sidebar:
        st.markdown(f"**{data.study.title}**")
        st.caption(f"read-only · `{data.study.study_id}`")
    st.navigation(list(pages.values())).run()


main()
