"""Dashboard selection state.

Pure helpers over a mutable mapping (``st.session_state`` in the app, a plain
``dict`` in tests). Nothing here imports Streamlit.

Matrix-explorer axis state is explicit:

``row_scenario`` / ``column_scenario``
    The case shown on each axis of the frame-pair matrix. They are independent
    and may be equal (within-case) or different (cross-case).
``row_frame`` / ``column_frame``
    0-based index of the selected frame on each axis.

Every setter validates against the study's configured labels before writing.
An invalid value leaves the state unchanged and returns a warning string.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any


ROW_SCENARIO = "row_scenario"
COLUMN_SCENARIO = "column_scenario"
ROW_FRAME = "row_frame"
COLUMN_FRAME = "column_frame"
MATRIX_KEYS: tuple[str, ...] = (ROW_SCENARIO, COLUMN_SCENARIO, ROW_FRAME, COLUMN_FRAME)

RUN = "filter_run"
CHECKPOINT = "filter_checkpoint"
LAYER = "filter_layer"
PAIR_SET = "filter_pair_set"
PRIMARY_METRIC = "filter_primary_metric"
SCENARIOS = "filter_scenarios"
FRAMES_PER_CASE = "filter_frames_per_case"
SHOW_CI = "filter_show_ci"
SHOW_RAW = "filter_show_raw"
SHOW_SIGNATURES = "filter_show_signatures"
EXPLORER_CLIP = "explorer_clip"
FILTER_KEYS: tuple[str, ...] = (
    RUN, CHECKPOINT, LAYER, PAIR_SET, PRIMARY_METRIC, SCENARIOS,
    FRAMES_PER_CASE, SHOW_CI, SHOW_RAW, SHOW_SIGNATURES,
)  # fmt: skip
#: Keys re-assigned on every rerun so widget-backed values survive page changes.
PERSISTENT_KEYS: tuple[str, ...] = (*MATRIX_KEYS, *FILTER_KEYS, EXPLORER_CLIP)

MAX_FRAMES_PER_CASE = 3

State = MutableMapping[str, Any]


@dataclass(frozen=True)
class MatrixSelection:
    """Snapshot of the matrix axes, read once per rerun."""

    row_scenario: str
    column_scenario: str
    row_frame: int
    column_frame: int

    @property
    def within_case(self) -> bool:
        return self.row_scenario == self.column_scenario

    @property
    def scope(self) -> str:
        return "within case" if self.within_case else "across cases"


@dataclass(frozen=True)
class FilterOptions:
    """Valid choices for the global filters, in display order."""

    runs: tuple[str, ...]
    checkpoints: tuple[str, ...]
    layers: tuple[str, ...]
    pair_sets: tuple[str, ...]
    metrics: tuple[str, ...]
    scenarios: tuple[str, ...]


@dataclass(frozen=True)
class FilterDefaults:
    """Defaults from the dashboard YAML."""

    checkpoint: str
    layer: str
    pair_set: str
    primary_metric: str
    row_scenario: str
    column_scenario: str
    frames_per_case: int = MAX_FRAMES_PER_CASE
    show_ci: bool = True
    show_raw: bool = True


def resolve_choice(value: Any, options: Sequence[Any], *, name: str) -> tuple[Any, str | None]:
    """Return ``value`` when valid, else the first option plus a warning."""
    if not options:
        return None, f"No options are available for {name}."
    if value in options:
        return value, None
    return options[0], f"{name}={value!r} is not available; using {options[0]!r}."


def clamp_frames_per_case(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return MAX_FRAMES_PER_CASE
    return min(max(number, 1), MAX_FRAMES_PER_CASE)


def center_frame(frames_per_case: int) -> int:
    """Default selected frame index: the centre frame."""
    return (max(int(frames_per_case), 1) - 1) // 2


def _frames(state: State, frames_per_case: int | None) -> int:
    if frames_per_case is not None:
        return clamp_frames_per_case(frames_per_case)
    return clamp_frames_per_case(state.get(FRAMES_PER_CASE, MAX_FRAMES_PER_CASE))


def _reset_frames(state: State, frames_per_case: int | None) -> None:
    center = center_frame(_frames(state, frames_per_case))
    state[ROW_FRAME] = center
    state[COLUMN_FRAME] = center


def _invalid(name: str, value: Any, labels: Sequence[str]) -> str:
    return f"{name}={value!r} is not a configured scenario label ({', '.join(labels)}); selection unchanged."


# --------------------------------------------------------------------------- matrix axes


def init_matrix_state(
    state: State,
    labels: Sequence[str],
    *,
    row_default: str,
    column_default: str,
    frames_per_case: int = MAX_FRAMES_PER_CASE,
) -> list[str]:
    """Fill missing or invalid matrix keys; returns warnings for invalid values."""
    warnings: list[str] = []
    for key, default in ((ROW_SCENARIO, row_default), (COLUMN_SCENARIO, column_default)):
        if key in state and state[key] in labels:
            continue
        candidate = state.get(key, default)
        choice, warning = resolve_choice(candidate, labels, name=key)
        if warning:
            warnings.append(warning)
        state[key] = choice
    count = clamp_frames_per_case(frames_per_case)
    for key in (ROW_FRAME, COLUMN_FRAME):
        value = state.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < count:
            state[key] = center_frame(count)
    return warnings


def read_matrix_state(state: State) -> MatrixSelection:
    return MatrixSelection(
        row_scenario=str(state[ROW_SCENARIO]),
        column_scenario=str(state[COLUMN_SCENARIO]),
        row_frame=int(state[ROW_FRAME]),
        column_frame=int(state[COLUMN_FRAME]),
    )


def set_case_pair(
    state: State,
    row: str,
    column: str,
    labels: Sequence[str],
    frames_per_case: int | None = None,
) -> str | None:
    """Case-cell click: set both axes and reset the frame selection."""
    if row not in labels:
        return _invalid(ROW_SCENARIO, row, labels)
    if column not in labels:
        return _invalid(COLUMN_SCENARIO, column, labels)
    state[ROW_SCENARIO] = row
    state[COLUMN_SCENARIO] = column
    _reset_frames(state, frames_per_case)
    return None


def set_row_scenario(
    state: State, row: str, labels: Sequence[str], frames_per_case: int | None = None
) -> str | None:
    """Change only the row axis; the column case is retained and frames reset."""
    if row not in labels:
        return _invalid(ROW_SCENARIO, row, labels)
    state[ROW_SCENARIO] = row
    _reset_frames(state, frames_per_case)
    return None


def set_column_scenario(
    state: State, column: str, labels: Sequence[str], frames_per_case: int | None = None
) -> str | None:
    """Change only the column axis; the row case is retained and frames reset."""
    if column not in labels:
        return _invalid(COLUMN_SCENARIO, column, labels)
    state[COLUMN_SCENARIO] = column
    _reset_frames(state, frames_per_case)
    return None


def set_frame_pair(
    state: State, row_frame: int, column_frame: int, frames_per_case: int | None = None
) -> str | None:
    """Frame-pair cell click: select one cell of the frame matrix."""
    count = _frames(state, frames_per_case)
    if not (0 <= int(row_frame) < count and 0 <= int(column_frame) < count):
        return f"Frame pair ({row_frame}, {column_frame}) is outside the {count}×{count} matrix; selection unchanged."
    state[ROW_FRAME] = int(row_frame)
    state[COLUMN_FRAME] = int(column_frame)
    return None


def swap_case_axes(state: State) -> None:
    """Exchange row/column cases and transpose the selected frame coordinates."""
    state[ROW_SCENARIO], state[COLUMN_SCENARIO] = state[COLUMN_SCENARIO], state[ROW_SCENARIO]
    state[ROW_FRAME], state[COLUMN_FRAME] = state[COLUMN_FRAME], state[ROW_FRAME]


def set_frames_per_case(state: State, frames_per_case: int) -> None:
    """Change the matrix size, keeping the selected frame inside it."""
    count = clamp_frames_per_case(frames_per_case)
    state[FRAMES_PER_CASE] = count
    for key in (ROW_FRAME, COLUMN_FRAME):
        if not 0 <= int(state.get(key, 0)) < count:
            state[key] = center_frame(count)


# --------------------------------------------------------------------------- global filters


def _filter_values(options: FilterOptions, defaults: FilterDefaults) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    values: dict[str, Any] = {}
    for key, value, choices, name in (
        (CHECKPOINT, defaults.checkpoint, options.checkpoints, "checkpoint"),
        (LAYER, defaults.layer, options.layers, "layer"),
        (PAIR_SET, defaults.pair_set, options.pair_sets, "pair_set"),
        (PRIMARY_METRIC, defaults.primary_metric, options.metrics, "primary_metric"),
    ):
        choice, warning = resolve_choice(value, choices, name=name)
        if warning:
            warnings.append(warning)
        values[key] = choice
    values[RUN] = options.runs[0] if options.runs else None
    values[SCENARIOS] = list(options.scenarios)
    values[FRAMES_PER_CASE] = clamp_frames_per_case(defaults.frames_per_case)
    if values[FRAMES_PER_CASE] != defaults.frames_per_case:
        warnings.append(
            f"frames_per_case={defaults.frames_per_case!r} is outside 1..{MAX_FRAMES_PER_CASE}; "
            f"using {values[FRAMES_PER_CASE]}."
        )
    values[SHOW_CI] = bool(defaults.show_ci)
    values[SHOW_RAW] = bool(defaults.show_raw)
    values[SHOW_SIGNATURES] = True
    return values, warnings


def _valid_filter(key: str, value: Any, options: FilterOptions) -> bool:
    choices: dict[str, Sequence[Any]] = {
        RUN: options.runs,
        CHECKPOINT: options.checkpoints,
        LAYER: options.layers,
        PAIR_SET: options.pair_sets,
        PRIMARY_METRIC: options.metrics,
    }
    if key in choices:
        return value in choices[key]
    if key == SCENARIOS:
        return isinstance(value, list | tuple) and all(item in options.scenarios for item in value)
    if key == FRAMES_PER_CASE:
        return isinstance(value, int) and 1 <= value <= MAX_FRAMES_PER_CASE
    return isinstance(value, bool)


def init_state(state: State, options: FilterOptions, defaults: FilterDefaults) -> list[str]:
    """Initialise missing/invalid filter and matrix keys from the dashboard defaults.

    Existing valid selections are kept. Returns warnings for defaults that are
    not available (they fall back to the first valid option).
    """
    values, warnings = _filter_values(options, defaults)
    for key, value in values.items():
        if key not in state or not _valid_filter(key, state[key], options):
            state[key] = value
    warnings.extend(
        init_matrix_state(
            state,
            options.scenarios,
            row_default=defaults.row_scenario,
            column_default=defaults.column_scenario,
            frames_per_case=state[FRAMES_PER_CASE],
        )
    )
    return warnings


def reset_state(state: State, options: FilterOptions, defaults: FilterDefaults) -> list[str]:
    """Restore every filter and matrix key to the dashboard YAML defaults."""
    for key in (*FILTER_KEYS, *MATRIX_KEYS):
        state.pop(key, None)
    return init_state(state, options, defaults)
