"""Strict YAML configuration loading, resolution, and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from drl_similarity.errors import ConfigError


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1


class StrictModel(BaseModel):
    """Base model that forbids unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ConfigReference(StrictModel):
    ref: Path


class WrapperSpec(StrictModel):
    id: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class EnvironmentExpected(StrictModel):
    observation_shape: list[int]
    observation_dtype: str
    action_space: Literal["discrete", "box"]
    effective_frame_skip: int = Field(ge=1)


class EnvironmentConfig(StrictModel):
    schema_version: Literal[1]
    kind: Literal["environment"]
    id: str
    backend: Literal["gymnasium"] = "gymnasium"
    registration_modules: list[str] = Field(default_factory=list)
    env_id: str
    make_kwargs: dict[str, Any] = Field(default_factory=dict)
    wrappers: list[WrapperSpec] = Field(default_factory=list)
    expected: EnvironmentExpected


class AlgorithmCapabilities(StrictModel):
    action_spaces: list[Literal["discrete", "box"]]
    recurrent: bool = False
    activation_layers: list[str]


class AlgorithmConfig(StrictModel):
    schema_version: Literal[1]
    kind: Literal["algorithm"]
    id: str
    adapter: str
    policy: str
    policy_kwargs: dict[str, Any] = Field(default_factory=dict)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    capabilities: AlgorithmCapabilities


class EvalSuiteConfig(StrictModel):
    seeds: list[int] = Field(min_length=1)
    episodes_per_seed: int = Field(default=1, ge=1)
    interval_steps: int | None = Field(default=None, ge=1)


class EvaluationConfig(StrictModel):
    deterministic_actions: bool = True
    max_agent_steps: int = Field(default=27000, ge=1)
    selection: EvalSuiteConfig
    final: EvalSuiteConfig


class TrainingConfig(StrictModel):
    total_timesteps: int = Field(ge=1)
    log_interval_steps: int = Field(default=10_000, ge=1)
    checkpoint_interval_steps: int = Field(default=100_000, ge=1)
    progress_bar: bool = True


class SelectionRuleConfig(StrictModel):
    rule: str
    smoothing_window: int = Field(default=1, ge=1)


class CheckpointSelectionConfig(StrictModel):
    weak: SelectionRuleConfig
    medium: SelectionRuleConfig
    strong: SelectionRuleConfig
    minimum_strong_improvement: float = 5.0


class ArtifactsPathConfig(StrictModel):
    root: Path
    collision: Literal["error", "version"] = "error"


class ReproducibilityConfig(StrictModel):
    torch_deterministic: bool = True
    cudnn_benchmark: bool = False
    capture_packages: bool = True
    capture_git: bool = True


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    kind: Literal["experiment"]
    id: str
    environment: EnvironmentConfig | ConfigReference
    algorithm: AlgorithmConfig | ConfigReference
    seed: int
    device: str = "auto"
    training: TrainingConfig
    evaluation: EvaluationConfig
    checkpoint_selection: CheckpointSelectionConfig
    artifacts: ArtifactsPathConfig
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)


class CandidateWindowConfig(StrictModel):
    anchor: Literal["life_start", "life_lost", "positive_goal_event"]
    offset_steps: int = 0
    min_reward: float = 0.0  # positive_goal_event fires when reward > min_reward


class ScenarioCollectionConfig(StrictModel):
    experiment_index: int = 0
    checkpoint: Literal["weak", "medium", "strong"] = "strong"
    seeds: list[int] = Field(min_length=1)
    episodes_per_seed: int = Field(default=2, ge=1)
    max_agent_steps: int = Field(default=27000, ge=1)
    retain_source_rgb: bool = True
    write_video_previews: bool = True
    candidate_windows: dict[str, CandidateWindowConfig]


class ScenarioReviewConfig(StrictModel):
    require_approved: bool = True
    minimum_reviewers: int = Field(default=1, ge=0)


class ScenarioSplitConfig(StrictModel):
    unit: Literal["clip"] = "clip"
    train_fraction: float = Field(default=0.7, gt=0.0, lt=1.0)
    seed: int = 3501


class ScenariosConfig(StrictModel):
    manifest: Path
    labels: list[str] = Field(min_length=2)
    minimum_clips_per_label: int = Field(default=20, ge=2)
    clip_steps: int = Field(default=32, ge=1)
    burn_in_steps: int = Field(default=16, ge=0)
    collection: ScenarioCollectionConfig
    review: ScenarioReviewConfig = Field(default_factory=ScenarioReviewConfig)
    split: ScenarioSplitConfig = Field(default_factory=ScenarioSplitConfig)


class LayerExtractionConfig(StrictModel):
    name: str
    source: str | None = None
    sources_by_adapter: dict[str, str] | None = None
    pooling: Literal["none", "flatten", "global_average"] = "none"
    required: bool = True

    @model_validator(mode="after")
    def _require_source(self) -> LayerExtractionConfig:
        if self.source is None and not self.sources_by_adapter:
            raise ValueError("layer extraction requires source or sources_by_adapter")
        return self


class ExtractionConfig(StrictModel):
    batch_size: int = Field(default=64, ge=1)
    device: str = "auto"
    dtype: Literal["float32"] = "float32"
    layers: list[LayerExtractionConfig] = Field(min_length=1)


class MetricSpec(StrictModel):
    id: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class EmbeddingSpec(StrictModel):
    id: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class BootstrapConfig(StrictModel):
    unit: Literal["clip"] = "clip"
    replicates: int = Field(default=2000, ge=1)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    seed: int = 4001


class AnalysisConfig(StrictModel):
    pair_sets: list[str]
    metrics: list[MetricSpec]
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    embeddings: list[EmbeddingSpec] = Field(default_factory=list)


class StudyExperimentRef(StrictModel):
    run_dir: Path
    checkpoint_set: Literal["selected"] = "selected"


class DashboardTheme(StrictModel):
    capability_order: list[str] = Field(default_factory=lambda: ["weak", "medium", "strong"])
    scenario_order: list[str] = Field(default_factory=lambda: ["baseline", "danger", "goal"])


class DashboardCharts(StrictModel):
    show_raw_points: bool = True
    confidence_intervals: bool = True
    max_heatmap_samples: int = 500
    clip_fps: int = 8


class DashboardDownloads(StrictModel):
    enabled: bool = True
    formats: list[Literal["csv", "png"]] = Field(default_factory=lambda: ["csv", "png"])


class DashboardDefaultFilters(StrictModel):
    checkpoint: str = "strong"
    layer: str = "policy_features"
    row_scenario: str = "danger"
    column_scenario: str = "danger"
    frames_per_case: int = 3
    primary_metric: str = "cosine"
    pair_set: str = "cross_clip_aligned_step"


class DashboardConfig(StrictModel):
    schema_version: Literal[1]
    kind: Literal["dashboard"]
    id: str
    title: str
    default_page: str = "similarity_matrix"
    default_filters: DashboardDefaultFilters = Field(default_factory=DashboardDefaultFilters)
    theme: DashboardTheme = Field(default_factory=DashboardTheme)
    charts: DashboardCharts = Field(default_factory=DashboardCharts)
    downloads: DashboardDownloads = Field(default_factory=DashboardDownloads)


class StudyConfig(StrictModel):
    schema_version: Literal[1]
    kind: Literal["study"]
    id: str
    experiments: list[StudyExperimentRef] = Field(min_length=1)
    scenarios: ScenariosConfig
    extraction: ExtractionConfig
    analysis: AnalysisConfig
    dashboard: DashboardConfig | ConfigReference
    artifacts: ArtifactsPathConfig


DocumentKind = Literal["environment", "algorithm", "experiment", "study", "dashboard"]
DocumentModel = EnvironmentConfig | AlgorithmConfig | ExperimentConfig | StudyConfig | DashboardConfig

KIND_MODELS: dict[DocumentKind, type[StrictModel]] = {
    "environment": EnvironmentConfig,
    "algorithm": AlgorithmConfig,
    "experiment": ExperimentConfig,
    "study": StudyConfig,
    "dashboard": DashboardConfig,
}


class ResolvedConfig(StrictModel):
    value: DocumentModel
    source_paths: tuple[Path, ...]
    config_hash: str
    canonical: dict[str, Any]


def read_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Unable to read config file: {path}", context={"path": str(path)}) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}", context={"path": str(path)}) from exc
    return data


def parse_document(data: Any, *, path: Path, expected_kind: DocumentKind | None = None) -> DocumentModel:
    if not isinstance(data, dict):
        raise ConfigError(
            "Config root must be a mapping with schema_version, kind, and id",
            context={"path": str(path), "received": type(data).__name__},
        )
    for key in ("schema_version", "kind", "id"):
        if key not in data:
            raise ConfigError(f"Missing required root field {key!r}", context={"path": str(path)})
    kind = data["kind"]
    if kind not in KIND_MODELS:
        raise ConfigError(
            f"Unsupported document kind {kind!r}",
            context={"path": str(path), "valid": sorted(KIND_MODELS)},
        )
    if expected_kind is not None and kind != expected_kind:
        raise ConfigError(
            f"Expected kind {expected_kind!r}, found {kind!r}",
            context={"path": str(path)},
        )
    if data["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(
            f"Unsupported schema_version {data['schema_version']!r}",
            context={"path": str(path)},
        )
    model_cls = KIND_MODELS[kind]  # type: ignore[index]
    try:
        return model_cls.model_validate(data)  # type: ignore[return-value]
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, path)) from exc


def _format_validation_error(exc: ValidationError, path: Path) -> str:
    parts: list[str] = [f"Config validation failed for {path}:"]
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ()))
        parts.append(f"  - {loc or '<root>'}: {err.get('msg')} (input={err.get('input')!r})")
    return "\n".join(parts)


def _resolve_ref(
    ref: ConfigReference,
    *,
    base_file: Path,
    expected_kind: DocumentKind,
    stack: list[Path],
    source_paths: list[Path],
) -> DocumentModel:
    target = (base_file.parent / ref.ref).resolve()
    if target in stack:
        cycle = " -> ".join(str(p) for p in [*stack, target])
        raise ConfigError(f"Config reference cycle detected: {cycle}")
    data = read_yaml(target)
    doc = parse_document(data, path=target, expected_kind=expected_kind)
    source_paths.append(target)
    return resolve_references(doc, base_file=target, stack=[*stack, target], source_paths=source_paths)


def resolve_references(
    doc: DocumentModel,
    *,
    base_file: Path,
    stack: list[Path] | None = None,
    source_paths: list[Path] | None = None,
) -> DocumentModel:
    stack = stack or [base_file.resolve()]
    source_paths = source_paths if source_paths is not None else []
    if isinstance(doc, ExperimentConfig):
        env = doc.environment
        algo = doc.algorithm
        if isinstance(env, ConfigReference):
            env = _resolve_ref(
                env,
                base_file=base_file,
                expected_kind="environment",
                stack=stack,
                source_paths=source_paths,
            )
        if isinstance(algo, ConfigReference):
            algo = _resolve_ref(
                algo,
                base_file=base_file,
                expected_kind="algorithm",
                stack=stack,
                source_paths=source_paths,
            )
        artifacts = ArtifactsPathConfig(
            root=_normalize_path(doc.artifacts.root, base_file),
            collision=doc.artifacts.collision,
        )
        return doc.model_copy(update={"environment": env, "algorithm": algo, "artifacts": artifacts})
    if isinstance(doc, StudyConfig):
        dashboard = doc.dashboard
        if isinstance(dashboard, ConfigReference):
            dashboard = _resolve_ref(
                dashboard,
                base_file=base_file,
                expected_kind="dashboard",
                stack=stack,
                source_paths=source_paths,
            )
        experiments = [
            StudyExperimentRef(
                run_dir=_normalize_path(item.run_dir, base_file),
                checkpoint_set=item.checkpoint_set,
            )
            for item in doc.experiments
        ]
        scenarios = doc.scenarios.model_copy(
            update={"manifest": _normalize_path(doc.scenarios.manifest, base_file)}
        )
        artifacts = ArtifactsPathConfig(
            root=_normalize_path(doc.artifacts.root, base_file),
            collision=doc.artifacts.collision,
        )
        return doc.model_copy(
            update={
                "dashboard": dashboard,
                "experiments": experiments,
                "scenarios": scenarios,
                "artifacts": artifacts,
            }
        )
    return doc


def _normalize_path(path: Path, base_file: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base_file.parent / path).resolve()


def validate_cross_document(doc: DocumentModel, *, allow_missing_scenario_manifest: bool = False) -> None:
    """Check registered IDs and cross-field rules after references are resolved."""
    if isinstance(doc, EnvironmentConfig):
        _validate_environment(doc)
    elif isinstance(doc, AlgorithmConfig):
        _require_registered(doc.adapter, "algorithm adapter", "adapter")
    elif isinstance(doc, ExperimentConfig):
        _validate_experiment(doc)
    elif isinstance(doc, StudyConfig):
        _validate_study(doc, allow_missing_scenario_manifest=allow_missing_scenario_manifest)


def _registered_ids(registry: str) -> list[str]:
    """Valid IDs of a named registry (imported lazily: registries depend on this module)."""
    if registry == "algorithm adapter":
        from drl_similarity.agents.registry import ALGORITHM_ADAPTERS

        return ALGORITHM_ADAPTERS.ids()
    if registry == "wrapper":
        from drl_similarity.environments.wrappers import WRAPPER_REGISTRY

        return WRAPPER_REGISTRY.ids()
    if registry == "selection rule":
        from drl_similarity.training.selection import SELECTION_RULES

        return SELECTION_RULES.ids()
    if registry == "metric":
        from drl_similarity.metrics.registry import list_metric_ids

        return list_metric_ids()
    if registry == "pair set":
        from drl_similarity.metrics.pairs import PAIR_SETS

        return sorted(PAIR_SETS)
    raise KeyError(registry)


def _require_registered(item_id: str, registry: str, field: str) -> None:
    valid = _registered_ids(registry)
    if item_id not in valid:
        raise ConfigError(f"Unknown {registry} id {item_id!r} in {field}; valid ids: {', '.join(valid)}")


def _validate_environment(doc: EnvironmentConfig) -> None:
    for index, wrapper in enumerate(doc.wrappers):
        _require_registered(wrapper.id, "wrapper", f"wrappers.{index}.id")


def _validate_experiment(doc: ExperimentConfig) -> None:
    assert isinstance(doc.environment, EnvironmentConfig)
    assert isinstance(doc.algorithm, AlgorithmConfig)
    _validate_environment(doc.environment)
    _require_registered(doc.algorithm.adapter, "algorithm adapter", "algorithm.adapter")
    for name in ("weak", "medium", "strong"):
        rule = getattr(doc.checkpoint_selection, name).rule
        _require_registered(rule, "selection rule", f"checkpoint_selection.{name}.rule")

    # Evaluation suites must not share seeds
    seeds = [*doc.evaluation.selection.seeds, *doc.evaluation.final.seeds]
    duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if duplicates:
        raise ConfigError(
            "Selection and final evaluation seeds must be unique and disjoint",
            context={"duplicates": duplicates},
        )

    action_space = doc.environment.expected.action_space
    if action_space not in doc.algorithm.capabilities.action_spaces:
        raise ConfigError(
            "Algorithm does not support the environment action space",
            context={"action_space": action_space, "supported": doc.algorithm.capabilities.action_spaces},
        )


def _validate_study(doc: StudyConfig, *, allow_missing_scenario_manifest: bool) -> None:
    scenarios = doc.scenarios
    if len(scenarios.labels) != len(set(scenarios.labels)):
        raise ConfigError("Scenario labels must be unique", context={"labels": scenarios.labels})
    if set(scenarios.collection.candidate_windows) != set(scenarios.labels):
        raise ConfigError(
            "scenarios.collection.candidate_windows must define exactly one window per label",
            context={"labels": scenarios.labels, "windows": sorted(scenarios.collection.candidate_windows)},
        )
    if len(scenarios.collection.seeds) != len(set(scenarios.collection.seeds)):
        raise ConfigError("Scenario collection seeds must be unique")
    if not allow_missing_scenario_manifest and not scenarios.manifest.exists():
        raise ConfigError(
            f"Scenario manifest not found: {scenarios.manifest}",
            context={"hint": "run drl-sim collect-scenarios, then curate the template"},
        )

    names = [layer.name for layer in doc.extraction.layers]
    if len(names) != len(set(names)):
        raise ConfigError("Extraction layer names must be unique", context={"layers": names})

    for index, pair_set in enumerate(doc.analysis.pair_sets):
        _require_registered(pair_set, "pair set", f"analysis.pair_sets.{index}")
    metric_ids = [metric.id for metric in doc.analysis.metrics]
    if len(metric_ids) != len(set(metric_ids)):
        raise ConfigError("Duplicate analysis metric ids", context={"metrics": metric_ids})
    for index, metric_id in enumerate(metric_ids):
        _require_registered(metric_id, "metric", f"analysis.metrics.{index}.id")

    # Dashboard defaults may only name things this study computes
    if isinstance(doc.dashboard, DashboardConfig):
        defaults = doc.dashboard.default_filters
        checks = {
            "layer": (defaults.layer, names),
            "primary_metric": (defaults.primary_metric, metric_ids),
            "pair_set": (defaults.pair_set, doc.analysis.pair_sets),
            "row_scenario": (defaults.row_scenario, scenarios.labels),
            "column_scenario": (defaults.column_scenario, scenarios.labels),
        }
        for field, (value, valid) in checks.items():
            if value not in valid:
                raise ConfigError(
                    f"dashboard.default_filters.{field}={value!r} is not part of this study",
                    context={"valid": list(valid)},
                )


def _path_to_repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def canonicalize_config(doc: DocumentModel) -> dict[str, Any]:
    """Serialize a resolved document with sorted keys and repo-relative paths."""
    raw = doc.model_dump(mode="python")
    return _canonicalize_value(raw)


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in value]
    if isinstance(value, Path):
        return _path_to_repo_relative(value)
    if isinstance(value, StrictModel):
        return _canonicalize_value(value.model_dump(mode="python"))
    return value


def hash_config(canonical: dict[str, Any]) -> str:
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_resolved_config(path: Path, canonical: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(canonical, sort_keys=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8")


def load_config(
    path: Path | str,
    expected_kind: DocumentKind | None = None,
    *,
    allow_missing_scenario_manifest: bool = False,
) -> ResolvedConfig:
    """Load, resolve, validate, canonicalize, and hash a YAML config document."""
    config_path = Path(path).resolve()
    data = read_yaml(config_path)
    doc = parse_document(data, path=config_path, expected_kind=expected_kind)
    source_paths = [config_path]
    resolved = resolve_references(doc, base_file=config_path, source_paths=source_paths)
    validate_cross_document(
        resolved,
        allow_missing_scenario_manifest=allow_missing_scenario_manifest,
    )
    canonical = canonicalize_config(resolved)
    config_hash = hash_config(canonical)
    return ResolvedConfig(
        value=resolved,
        source_paths=tuple(dict.fromkeys(source_paths)),
        config_hash=config_hash,
        canonical=canonical,
    )

