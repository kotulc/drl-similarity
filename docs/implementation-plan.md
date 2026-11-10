# Implementation plan

This plan turns the Frogger representation study described in the
[README](../README.md) into a small, reproducible Python package. The first
implementation is intentionally offline: training, scenario curation,
activation extraction, and analysis produce artifacts that a read-only
dashboard loads.

## 1. Success criteria

The MVP is complete when a fresh checkout can:

1. Install on Python 3.11 with one documented `uv sync`.
2. train either built-in CNN DQN or CNN PPO from YAML on the pinned Frogger
   environment while saving evaluation results and resumable checkpoints;
3. select weak, medium, and strong checkpoints from held-out evaluation return;
4. load a versioned dataset containing at least 20 curated clips per scenario;
5. extract aligned activations from at least the final convolutional map and
   NatureCNN feature vector for all three checkpoints;
6. compute within-scenario, between-scenario, temporal, and cross-checkpoint
   metrics with clip-level confidence intervals;
7. reproduce every result from saved configuration and artifact manifests; and
8. display the results and source clips in a Streamlit dashboard without
   retraining or recomputing activations.

The initial experiment is exploratory and uses one training seed. The package
and schemas must accept multiple seeds without changing the analysis API.

## 2. Pinned design choices

### Environment

- Environment ID: `ALE/Frogger-v5`.
- Observation source: RGB pixels, resized to 84×84 grayscale.
- Observation history: four frames.
- Effective action repeat: four.
- Sticky-action probability: `0.0` for exact replay in the first study.
- Episodic life: disabled; a lost life is recorded as an event but does not
  become an artificial episode boundary.
- No-op reset count, maximum episode length, reward clipping, and terminal
  handling are explicit configuration fields.

Only one component may perform frame skipping. If SB3's `AtariWrapper` repeats
actions four times, the base ALE environment is created with `frameskip=1`.
This avoids the common accidental 16-frame action repeat.

Training and final reporting use the same environment fingerprint. A fingerprint
is a hash of the environment ID, ALE version, wrapper order, wrapper arguments,
observation shape/dtype, and action-space description.

### Agent

- Initial algorithm: Stable-Baselines3 DQN.
- Initial policy: `CnnPolicy` with the standard NatureCNN extractor.
- Training budget: configurable, with 10 million environment steps as the
  initial full-run target.
- Evaluation: deterministic greedy actions on a fixed selection seed set at
  regular intervals.
- Checkpoint cadence: at every evaluation, plus final and best-observed.

The algorithm is selected by `algorithm.adapter` in YAML; no command or
pipeline module imports DQN directly. The initial implementation should use SB3
rather than copy a DQN training loop. A thin callback records checkpoints and
evaluations. PyTorch forward hooks extract activations; SB3 itself is not
forked.

The same adapter contract supports SB3 PPO and A2C, sb3-contrib QR-DQN and
RecurrentPPO, and later non-SB3 algorithms. DQN produces the first Frogger
result; DQN and PPO are both required for the completed framework so the
algorithm-neutral registry, configuration, artifacts, extraction, and
dashboard are exercised by an off-policy and an on-policy implementation.

### Capability selection

Checkpoint selection consumes only the selection evaluation suite:

1. **Strong:** checkpoint with the highest mean return; ties prefer the earlier
   checkpoint.
2. **Weak:** earliest post-learning-start checkpoint with a complete evaluation.
3. **Medium:** checkpoint whose smoothed mean return is closest to the midpoint
   between weak and strong, constrained to occur between them in training.

The selector fails with an explanation if strong does not materially outperform
weak or no valid medium checkpoint exists. Thresholds for “materially” and
smoothing are configuration, not hidden constants. Selection results record
step, score, uncertainty, and rule version.

A disjoint final seed set measures the selected checkpoints. Scenario clips are
also disjoint from both evaluation suites.

### Scenario clips

The first dataset contains three balanced labels:

- `baseline`: a window near the beginning of a life, before a salient outcome;
- `danger`: a window ending immediately before loss of life or game over;
- `goal`: a window ending immediately before or at a successful goal/reward
  event.

Candidate episodes are collected from the strong checkpoint and, if needed for
coverage, a manual or scripted policy. A curator chooses fixed-length windows
and records the anchor event and label. A second review pass checks labels
without looking at model activations.

The default target is 20 clips per label, 32 model steps per clip, plus up to 16
burn-in steps. These values are configurable. Splits occur by clip, never by
frame. The stored preprocessed observations are the canonical stimuli used by
all checkpoints.

## 3. Package layout

```text
pyproject.toml
README.md
configs/
  environments/
    atari-frogger.yaml
  algorithms/
    dqn-nature-cnn.yaml
    ppo-nature-cnn.example.yaml
  experiments/
    frogger-dqn.yaml
  studies/
    frogger-v1.yaml
  dashboards/
    default.yaml
src/drl_similarity/
  cli.py
  config.py
  errors.py
  logging.py
  pipeline.py
  registry.py
  environments/
    base.py
    atari.py
    wrappers.py
  agents/
    base.py
    registry.py
    sb3/
      base.py
      dqn.py
      ppo.py
  training/
    callbacks.py
    evaluate.py
    runner.py
    selection.py
  scenarios/
    schema.py
    collect.py
    validate.py
  activations/
    extract.py
    pooling.py
  metrics/
    pairs.py
    registry.py
    vector.py
    geometry.py
    bootstrap.py
  analysis.py
  artifacts/
    manifest.py
    store.py
  dashboard/
    app.py
    data.py
    state.py
    charts.py
    components.py
    pages/
tests/
  unit/
  integration/
```

Modules remain ordinary functions unless polymorphism is required. The adapter
interfaces exist to isolate environment and model differences, not to create a
general plugin framework in the MVP.

## 4. Core interfaces

Use typed protocols and plain dataclasses or Pydantic models for boundaries.

### `EnvironmentAdapter`

```python
class EnvironmentAdapter(Protocol):
    def make(self, *, seed: int, render_mode: str | None = None) -> gym.Env: ...
    def fingerprint(self) -> str: ...
    def describe(self) -> dict[str, object]: ...
```

The Atari implementation owns wrapper order and asserts the final observation
shape, dtype, frame stack, and effective frame skip.

### `AlgorithmAdapter` and `LoadedAgent`

`AlgorithmAdapter` owns training-framework differences. `LoadedAgent` is the
small inference/extraction surface returned after loading a checkpoint.

```python
class AlgorithmAdapter(Protocol):
    adapter_id: str

    def validate(
        self,
        algorithm: AlgorithmConfig,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> None: ...
    def build(
        self,
        algorithm: AlgorithmConfig,
        env: gym.Env,
        *,
        seed: int,
        device: str,
    ) -> TrainableAgent: ...
    def load(
        self,
        checkpoint: Path,
        algorithm: AlgorithmConfig,
        env: gym.Env | None,
        *,
        device: str,
    ) -> LoadedAgent: ...


class LoadedAgent(Protocol):
    def predict(self, observations: np.ndarray) -> np.ndarray: ...
    def layer_names(self) -> tuple[str, ...]: ...
    def extract(
        self,
        observations: np.ndarray,
        *,
        layers: tuple[str, ...],
        episode_starts: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]: ...
```

`TrainableAgent` adds `learn`, `save`, and callback registration. Keep this
protocol internal until a second backend proves which methods are genuinely
shared.

The registry is an explicit dictionary from stable YAML IDs to factories:

```python
ALGORITHM_ADAPTERS = {
    "sb3.dqn": make_sb3_dqn_adapter,
    "sb3.ppo": make_sb3_ppo_adapter,
}
```

`get_algorithm_adapter(id)` raises a configuration error listing valid IDs.
Pipeline code asks the registry for an adapter and never branches on the ID.
Do not accept arbitrary import strings from YAML. Third-party Python entry
points can be considered after the built-in adapters are stable.

The DQN adapter exposes stable logical layer names rather than leaking PyTorch
module paths into study configuration. Example names are `conv1`, `conv2`,
`conv3`, `conv_flat`, `policy_features`, and `q_values`. In standard SB3 Atari
DQN, NatureCNN's 512-dimensional `policy_features` output is the final hidden
representation before Q-values; the plan does not invent a separate Q-network
hidden layer. PPO can expose the shared feature extractor, policy/value latent
vectors, action logits, and value estimate under equally stable logical names.

For a future RNN adapter, `episode_starts`, ordered batches, burn-in, and hidden
state are mandatory. Feed-forward adapters ignore `episode_starts`.

### Adapter contract rules

Every algorithm adapter must:

1. validate observation and action spaces before training;
2. apply the seed and device supplied by the runner;
3. save a framework checkpoint plus framework-neutral metadata;
4. load checkpoints without requiring a training environment for extraction;
5. return deterministic actions when requested;
6. enumerate logical layers and reject unknown requested layers;
7. return activations in the original sample order;
8. disclose whether the policy is feed-forward or recurrent; and
9. pass shared save/load, prediction-shape, extraction-shape, and no-mutation
   contract tests.

### Metric registry

Each metric receives a validated sample-by-feature matrix plus aligned sample
metadata and returns tidy records:

```python
MetricResult(
    metric="cosine",
    scope="within_scenario",
    checkpoint="strong",
    layer="policy_features",
    scenario="danger",
    estimate=0.81,
    ci_low=0.76,
    ci_high=0.85,
    n_clips=20,
)
```

Registry metadata declares whether a metric is a similarity or distance, its
range, required alignment, normalization, and supported tensor rank.

## 5. Configuration

All scientific and presentation choices are YAML-driven. The CLI accepts a
config path and operational flags only. `--resume`, `--force`, `--dry-run`, and
`--log-level` may alter execution behavior; they must not alter the experiment.
There are no flags such as `--learning-rate`, `--metric`, or `--env-id`.

Use PyYAML for parsing and strict Pydantic models for validation. Set
`extra="forbid"` on every model. Paths are resolved relative to the YAML file
that declares them. YAML references are loaded, validated by their own model,
and inserted into a resolved root model. Do not implement arbitrary recursive
merging, environment-variable substitution, executable YAML tags, or Hydra-like
override syntax in the MVP.

### Configuration document types

Every YAML document starts with:

```yaml
schema_version: 1
kind: environment  # environment | algorithm | experiment | study | dashboard
id: atari-frogger-v1
```

`schema_version` controls migrations. `kind` selects the Pydantic root model.
`id` is a stable human-readable identifier. Duplicate IDs with different
content are an error within one resolved study.

### Environment YAML

`configs/environments/atari-frogger.yaml`:

```yaml
schema_version: 1
kind: environment
id: atari-frogger-v1

backend: gymnasium
registration_modules:
  - ale_py
env_id: ALE/Frogger-v5
make_kwargs:
  obs_type: rgb
  frameskip: 1
  repeat_action_probability: 0.0
  full_action_space: false

wrappers:
  - id: record_episode_statistics
    kwargs: {}
  - id: sb3_atari
    kwargs:
      noop_max: 30
      frame_skip: 4
      screen_size: 84
      terminal_on_life_loss: false
      clip_reward: false
  - id: frame_stack
    kwargs:
      num_stack: 4

expected:
  observation_shape: [4, 84, 84]
  observation_dtype: uint8
  action_space: discrete
  effective_frame_skip: 4
```

Use **Gymnasium**, not legacy Gym. Import each module in
`registration_modules` before `gymnasium.make`; this registers ALE names in
current Gymnasium/ALE releases. Wrapper IDs resolve through an explicit wrapper
registry. The adapter asserts `expected` values after construction. It emits an
observation fingerprint and a full runtime fingerprint so reward-only wrapper
changes can be distinguished from changes to model input.

### Algorithm YAML

`configs/algorithms/dqn-nature-cnn.yaml`:

```yaml
schema_version: 1
kind: algorithm
id: sb3-dqn-nature-cnn-v1

adapter: sb3.dqn
policy: CnnPolicy
policy_kwargs:
  normalize_images: true
  features_extractor: nature_cnn
model_kwargs:
  learning_rate: 0.0001
  buffer_size: 100000
  learning_starts: 80000
  batch_size: 32
  tau: 1.0
  gamma: 0.99
  train_freq: 4
  gradient_steps: 1
  target_update_interval: 1000
  exploration_fraction: 0.1
  exploration_initial_eps: 1.0
  exploration_final_eps: 0.01
capabilities:
  action_spaces: [discrete]
  recurrent: false
  activation_layers:
    - conv1
    - conv2
    - conv3
    - conv_flat
    - policy_features
    - q_values
```

Adapter-specific `model_kwargs` remain a JSON-compatible mapping, but each
built-in adapter validates allowed keys and types before calling its backend.
Friendly policy/extractor IDs map to Python classes inside the adapter; YAML
does not contain import paths or Python expressions.

A PPO config changes `adapter`, policy, and model kwargs only:

```yaml
schema_version: 1
kind: algorithm
id: sb3-ppo-nature-cnn-v1
adapter: sb3.ppo
policy: CnnPolicy
policy_kwargs:
  normalize_images: true
  features_extractor: nature_cnn
model_kwargs:
  learning_rate: 0.00025
  n_steps: 128
  batch_size: 256
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.1
capabilities:
  action_spaces: [discrete, box]
  recurrent: false
  activation_layers:
    - conv1
    - conv2
    - conv3
    - policy_features
    - policy_latent
    - value_latent
    - action_logits
    - value
```

The PPO file may use the `.example.yaml` suffix during development. Rename it
to a normal config only after `sb3.ppo` passes the adapter contract suite. The
runner must fail early with “adapter not installed” rather than silently
falling back to DQN.

### Experiment YAML

`configs/experiments/frogger-dqn.yaml`:

```yaml
schema_version: 1
kind: experiment
id: frogger-dqn-seed-1

environment:
  ref: ../environments/atari-frogger.yaml
algorithm:
  ref: ../algorithms/dqn-nature-cnn.yaml

seed: 1
device: auto
training:
  total_timesteps: 10000000
  log_interval_steps: 10000
  checkpoint_interval_steps: 100000
  progress_bar: true
evaluation:
  deterministic_actions: true
  max_agent_steps: 27000  # 108,000 emulator frames at action repeat 4
  selection:
    seeds: [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    episodes_per_seed: 2
    interval_steps: 100000
  final:
    seeds: [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]
    episodes_per_seed: 3
checkpoint_selection:
  weak:
    rule: earliest_complete_after_learning_starts
  medium:
    rule: nearest_performance_midpoint
    smoothing_window: 3
  strong:
    rule: highest_mean_selection_return
  minimum_strong_improvement: 5.0
artifacts:
  root: ../../runs
  collision: error
reproducibility:
  torch_deterministic: true
  cudnn_benchmark: false
  capture_packages: true
  capture_git: true
```

The experiment resolver inserts complete environment and algorithm documents,
normalizes all paths, fills defaults, and writes
`runs/<run-id>/resolved-config.yaml`. Its SHA-256 hash is the `config_hash` used
by all downstream artifacts.

### Study YAML

`configs/studies/frogger-v1.yaml`:

```yaml
schema_version: 1
kind: study
id: frogger-representation-v1

experiments:
  - run_dir: ../../runs/frogger-dqn-seed-1
    checkpoint_set: selected

scenarios:
  manifest: ../../data/scenarios/frogger-v1.yaml
  labels: [baseline, danger, goal]
  minimum_clips_per_label: 20
  clip_steps: 32
  burn_in_steps: 16
  collection:
    experiment_index: 0
    checkpoint: strong
    seeds: [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010]
    episodes_per_seed: 2
    max_agent_steps: 27000
    retain_source_rgb: true
    write_video_previews: true
    candidate_windows:
      baseline:
        anchor: life_start
        offset_steps: 0
      danger:
        anchor: life_lost
        offset_steps: -31
      goal:
        anchor: positive_goal_event
        offset_steps: -31
  review:
    require_approved: true
    minimum_reviewers: 1
  split:
    unit: clip
    train_fraction: 0.7
    seed: 3501

extraction:
  batch_size: 64
  device: auto
  dtype: float32
  layers:
    - name: conv1
      source: conv1
      pooling: global_average
    - name: conv2
      source: conv2
      pooling: global_average
    - name: conv3
      source: conv3
      pooling: global_average
    - name: conv_flat
      source: conv_flat
      pooling: none
    - name: policy_features
      source: policy_features
      pooling: none
    - name: q_values
      source: q_values
      pooling: none

analysis:
  pair_sets:
    - cross_clip_all
    - cross_clip_aligned_step
    - clip_centroids
  metrics:
    - id: cosine
    - id: correlation
    - id: normalized_euclidean
      kwargs: {standardize_on: train_clips}
    - id: centroid_dispersion
    - id: temporal_cosine
      kwargs: {lags: [1, 2, 4, 8]}
    - id: within_between_gap
    - id: silhouette
    - id: nearest_centroid_accuracy
    - id: linear_cka
    - id: rsa_spearman
      kwargs: {first_order_distance: correlation}
    - id: svcca
      kwargs: {variance_retained: 0.99}
    - id: orthogonal_procrustes
  bootstrap:
    unit: clip
    replicates: 2000
    confidence_level: 0.95
    seed: 4001
  embeddings:
    - id: pca
      kwargs: {components: 2}
    - id: umap
      enabled: false
      kwargs: {components: 2, neighbors: 15, min_dist: 0.1, seed: 5001}

dashboard:
  ref: ../dashboards/default.yaml
artifacts:
  root: ../../artifacts
  collision: error
```

An experiment run is referenced by its immutable directory rather than by
copying model settings. Analysis verifies the stored resolved config and
checksums before loading it. `collect-scenarios` permits the declared manifest
to be absent because it creates a template and candidate data; all later study
stages require the manifest to exist, be approved, and pass validation. This is
the only stage-specific relaxation in reference validation.

For a multi-algorithm study, each extraction entry has a semantic `name` used
in result tables and either one `source` logical layer or
`sources_by_adapter`. For example:

```yaml
- name: action_output
  sources_by_adapter:
    sb3.dqn: q_values
    sb3.ppo: action_logits
  pooling: none
```

An optional entry may set `required: false`; otherwise a missing source layer
fails validation. Cross-algorithm geometry is computed only for semantic layer
names present in both runs. CKA and RSA permit different feature counts. SVCCA
uses its configured common retained subspace. Orthogonal Procrustes is marked
inapplicable, with an explicit reason row, when matrix shapes are incompatible.

### Dashboard YAML

`configs/dashboards/default.yaml`:

```yaml
schema_version: 1
kind: dashboard
id: default-dashboard-v1

title: Frogger representation study
default_page: overview
default_filters:
  checkpoint: strong
  layer: policy_features
  scenario: all
  pair_set: cross_clip_aligned_step
theme:
  capability_order: [weak, medium, strong]
  scenario_order: [baseline, danger, goal]
charts:
  show_raw_points: true
  confidence_intervals: true
  max_heatmap_samples: 500
  clip_fps: 8
downloads:
  enabled: true
  formats: [csv, png]
```

The dashboard configuration controls defaults and presentation only. It cannot
enable metrics that were not computed.

### Resolution and validation algorithm

Implement `load_config(path, expected_kind)` in this exact order:

1. Read UTF-8 YAML with `yaml.safe_load`.
2. Verify the root is a mapping with `schema_version`, `kind`, and `id`.
3. Reject unsupported schema versions and an unexpected root kind.
4. Parse the root with its strict Pydantic model.
5. Resolve each `ref` relative to the declaring file and recursively load the
   expected document kind.
6. Detect reference cycles using canonical absolute paths.
7. Convert paths to normalized absolute paths in memory.
8. Run cross-document validation: registered adapters/wrappers/metrics,
   algorithm/action-space compatibility, recurrent extraction requirements,
   unique seeds, and required artifact availability.
9. Serialize a canonical resolved mapping with sorted keys and paths relative
   to the repository where possible.
10. Hash the canonical serialization and return
    `ResolvedConfig(value, source_paths, config_hash)`.

Write the resolved YAML beside every top-level artifact. Never mutate source
YAML. Configuration validation rejects overlapping evaluation/scenario seeds,
incompatible observation fingerprints, missing layers, unknown algorithm
adapters, invalid backend kwargs, unbalanced or undersized scenario sets, and
dashboard defaults that do not exist in the result set.

## 6. Artifact schemas

### Run artifacts

`evaluations.parquet` has one row per checkpoint and evaluation episode:

- `run_id`, `training_seed`, `checkpoint_id`, `training_step`;
- `algorithm_id`, `adapter_id`, `policy_id`, and `backend_version`;
- `suite` (`selection` or `final`), `eval_seed`, `episode_index`;
- `return`, `length`, `lives_lost`, `goals`, and `wall_time`; and
- `environment_fingerprint`, `code_commit`, and `config_hash`.

Each SB3 checkpoint is accompanied by a small JSON sidecar with its step,
evaluation status, SHA-256 checksum, environment fingerprint, and versions.

### Scenario manifest

The versioned YAML manifest contains dataset-level provenance and one record per
clip:

- `dataset_id`, `environment_fingerprint`, collector policy/checkpoint;
- `clip_id`, `scenario`, `split`, `source_episode_id`;
- `collection_seed`, action prefix or source trajectory reference;
- `start_step`, `anchor_step`, `end_step`, and `burn_in_steps`;
- `anchor_event`, rewards, actions, lives, terminal flags;
- observation array key and checksum; and
- curator, review status, notes, and creation timestamp.

`observations.npz` stores `uint8` preprocessed frame stacks keyed by clip ID.
The manifest records shape and checksum. A validator confirms fixed sequence
lengths, legal labels, unique IDs, balanced counts, non-overlapping source
windows, split isolation, and matching checksums.

### Activations

Activation files are partitioned by checkpoint and layer. Each array has shape
`(clips, steps, features)` after declared pooling. Metadata includes:

- source observation checksum;
- checkpoint checksum;
- logical and physical layer name;
- original tensor shape and pooling method;
- dtype, extraction batch size, device, and code/config hashes; and
- clip ordering and valid-step mask.

Float32 NPZ is sufficient for the MVP. If artifacts become too large, migrate
behind the artifact store to Zarr without changing metric interfaces.

### Metric outputs

- `frame_pairs.parquet`: optional diagnostic pair-level values with both clip
  and step IDs.
- `summaries.parquet`: within-checkpoint estimates and clip-bootstrap
  confidence intervals.
- `cross_checkpoint.parquet`: CKA, RSA, SVCCA, and Procrustes results for
  aligned samples.
- `embeddings.parquet`: PCA/UMAP coordinates joined to clip and step metadata.
- `manifest.json`: checksums and lineage for every input and output.

## 7. Analysis rules

### Preprocessing

Metric-specific preprocessing is explicit and fitted only where needed:

- cosine uses vectors as extracted, with zero-vector handling reported;
- correlation centers each vector;
- normalized Euclidean standardizes features using the analysis training clips,
  then applies those parameters to held-out clips;
- CKA centers sample Gram matrices;
- SVCCA chooses retained variance through configuration;
- Procrustes centers matrices and reports normalized residual; and
- RSA uses a configured first-order distance and Spearman correlation between
  upper triangles of the resulting dissimilarity matrices.

All outputs retain the preprocessing identifier.

### Pair construction

The primary within-scenario estimate compares frames from different clips with
the same label. It excludes pairs from the same clip to prevent adjacent-frame
autocorrelation from dominating the result. Report:

- all cross-clip pairs;
- frame-aligned pairs at the same relative offset; and
- clip-centroid pairs.

Between-scenario estimates use balanced clip pairs and the same relative-offset
rule where applicable. The within/between gap is calculated from matched pair
definitions.

Temporal smoothness is a separate analysis at configured lags within each clip.
It is plotted against input-space frame change so static clips cannot be
mistaken for learned stability.

### Uncertainty and leakage

- Bootstrap whole clips, stratified by scenario, with at least 1,000 replicates.
- Report point estimates and 95% percentile intervals.
- Fit nearest-centroid classification and normalization on training clips and
  evaluate on held-out clips.
- Do not interpret frame count as the independent sample count; always report
  clip count.
- For the future multi-seed study, aggregate within each seed first, then report
  variation across seeds.

Multiple metrics are descriptive in the MVP. If formal hypothesis tests are
added later, preregister primary metrics and correct for multiple comparisons.

## 8. Metric implementations

Prefer maintained package implementations:

- SciPy and scikit-learn for cosine, correlation, Euclidean, silhouette,
  nearest-centroid, PCA, and Procrustes primitives;
- `rsatoolbox` for representational dissimilarity matrices and RSA;
- scikit-learn CCA/SVD primitives for SVCCA; and
- a small, unit-tested linear CKA function if no stable dependency meets the
  array and metadata requirements.

The registry should initially expose only:

```text
cosine
correlation
normalized_euclidean
centroid_dispersion
temporal_cosine
within_between_gap
silhouette
nearest_centroid_accuracy
linear_cka
rsa_spearman
svcca
orthogonal_procrustes
```

Adding many near-duplicate distances is not an MVP goal. A new metric requires
documented invariances, a synthetic test, and a dashboard interpretation.

## 9. CLI contracts

The Typer application exposes these commands. Every command that can affect a
result requires a YAML config:

```text
drl-sim validate-config --config PATH
drl-sim doctor --config PATH
drl-sim train --config EXPERIMENT_YAML [--resume CHECKPOINT] [--dry-run]
drl-sim evaluate --config EXPERIMENT_YAML [--dry-run]
drl-sim select-checkpoints --config EXPERIMENT_YAML [--dry-run]
drl-sim collect-scenarios --config STUDY_YAML [--dry-run]
drl-sim validate-scenarios --config STUDY_YAML
drl-sim extract --config STUDY_YAML [--force] [--dry-run]
drl-sim analyze --config STUDY_YAML [--force] [--dry-run]
drl-sim dashboard --config STUDY_YAML
drl-sim run-study --config STUDY_YAML [--dry-run]
```

`run-study` is an idempotent orchestrator: validate references, verify completed
experiment runs, validate scenarios, extract missing activation artifacts,
analyze missing metric artifacts, then print the dashboard command. It does not
start expensive training implicitly. A missing experiment run is a clear error
with the exact `train` command required.

Commands print resolved config hash and artifact path and return nonzero on
validation failure. `--dry-run` resolves and validates configuration, prints
the planned inputs/outputs, and writes nothing. `--force` never overwrites in
place; it creates a new artifact version. `doctor` runs package, Atari ROM,
adapter registration, wrapper registration, CUDA, environment-shape, and short
reset/step checks selected by the supplied environment or experiment config.

Implement each Typer command as a thin function:

1. call `load_config`;
2. create a run logger with config hash;
3. call one application service;
4. catch only expected domain/configuration exceptions;
5. print a concise success or actionable failure message; and
6. return an exit code.

Commands do not contain Gymnasium, SB3, metric, or Streamlit business logic.

## 10. Dashboard implementation

The dashboard is a read-only view over completed artifacts. It never imports
SB3, creates a Gymnasium environment, executes a model, computes metrics, or
writes into an experiment directory. `drl-sim dashboard` resolves the study,
validates result manifests, sets the config path in the process environment,
and launches Streamlit.

The visual target is captured in
[the dashboard overview mockup](dashboard-mockup.svg). It is a layout contract,
not fabricated study output; displayed numbers are illustrative.

### Visualization framework and packages

Use **Streamlit + Plotly**.

- `streamlit`: Python-native multipage application shell, layout, controls,
  session state, caching, media display, downloads, and app testing. It keeps
  the dashboard in the same language and environment as the experiments and
  requires no custom web backend.
- `plotly`: interactive scientific charts, linked hover metadata, faceting,
  confidence bands, heatmaps, scatter selection, deterministic category/color
  ordering, and PNG/SVG export.
- `pandas` + `pyarrow`: typed Parquet loading, filtering, grouping, and CSV
  export.
- `Pillow` + `imageio`: frame-stack rendering, contact sheets, and MP4/GIF clip
  previews prepared outside the chart layer.
- `kaleido`: server-side Plotly PNG/SVG export for download buttons.
- `numpy`: matrix/image conversion only; dashboard pages do not recompute study
  metrics.

Do not use Matplotlib or Seaborn for dashboard charts: they produce static
figures and duplicate Plotly styling logic. They remain acceptable for
headless paper exports later. Do not introduce Dash, Panel, React, custom
JavaScript, or a database in the MVP. Streamlit is sufficient for this
read-only, artifact-backed research dashboard and minimizes custom code.

Use these Streamlit components:

- `st.navigation` and `st.Page` for the six-page structure;
- `st.sidebar`, `st.selectbox`, `st.multiselect`, `st.segmented_control`,
  `st.toggle`, and `st.slider` for global/local filters;
- `st.columns`, `st.container`, `st.metric`, `st.status`, and `st.warning` for
  page hierarchy and validation state;
- `st.plotly_chart` for every quantitative visualization;
- `st.image` and `st.video` for scenario inspection;
- `st.dataframe` for sortable provenance and filtered raw summaries;
- `st.tabs` and `st.expander` for secondary detail;
- `st.download_button` for filtered data, manifests, and rendered figures; and
- `st.cache_data` for checksum-keyed artifact reads and transformations.

Use these Plotly chart primitives:

- `plotly.graph_objects.Scatter` for learning curves, temporal lag curves,
  points with confidence intervals, and confidence bands;
- `plotly.express.scatter` for PCA/UMAP projections with hover metadata;
- `plotly.graph_objects.Heatmap` for similarity, confusion, centroid-distance,
  and cross-checkpoint matrices;
- `plotly.graph_objects.Violin` and `Box` for clip-level distributions;
- `make_subplots` for scenario/layer facets where independent axes are needed;
  and
- shapes/annotations for checkpoint markers, chance baselines, scenario
  boundaries, and warnings.

No extra component library is required initially. Add a third-party Streamlit
component only when a documented interaction cannot be built with core
Streamlit, it is actively maintained, and it has an automated fallback.

### Dashboard module responsibilities

- `dashboard/app.py`: page registration, global layout, study loading, fatal
  error boundary.
- `dashboard/data.py`: typed artifact readers, schema checks, joins, cached
  filtering, and download serialization.
- `dashboard/state.py`: query/session state, defaults from dashboard YAML, and
  valid filter options.
- `dashboard/charts.py`: pure functions returning Plotly figures.
- `dashboard/components.py`: reusable metric cards, warnings, provenance
  expander, filter controls, clip player, and empty states.
- `dashboard/pages/*.py`: page composition only; no direct file reads.

`data.py` returns immutable data bundles:

```python
DashboardData(
    study=StudySummary(...),
    runs=pd.DataFrame(...),
    evaluations=pd.DataFrame(...),
    clips=pd.DataFrame(...),
    summaries=pd.DataFrame(...),
    cross_checkpoint=pd.DataFrame(...),
    embeddings=pd.DataFrame(...),
    quality=tuple[QualityIssue, ...],
)
```

Use `st.cache_data` on readers with the manifest checksum as an explicit cache
key. Do not cache by path alone. Validate required columns and enums before a
page renders. One corrupt artifact produces a page-level error naming the file,
expected schema version, and regeneration command.

### Global shell and filters

The page uses wide layout. The header shows study title, config hash prefix,
artifact version, and a green/yellow/red validation badge. The sidebar contains:

1. study/run selector when multiple runs or seeds are present;
2. algorithm selector when a study compares algorithms;
3. checkpoint capability selector ordered weak, medium, strong;
4. layer selector populated from available activation metadata;
5. scenario multiselect ordered baseline, danger, goal;
6. pair-set selector;
7. aggregation selector (`frame`, `aligned_step`, `clip_centroid`);
8. metric selector restricted to the current page;
9. confidence-interval and raw-point toggles; and
10. a reset button restoring dashboard YAML defaults.

Filters with no effect on the current page are hidden, not disabled. Invalid
saved defaults fall back to the first valid option and generate a warning.
Selections persist in `st.session_state`; do not require URL routing in the
MVP.

Use fixed colors throughout:

- capability colors are keyed by `weak`, `medium`, and `strong`;
- scenario colors are keyed by `baseline`, `danger`, and `goal`;
- missing/invalid values use neutral gray; and
- color assignment never depends on current filtering order.

### Page 1 — Overview

Purpose: establish that the compared checkpoints differ in capability and that
the study artifacts are valid.

Top row metric cards:

- number of training seeds;
- algorithm and policy;
- environment ID and observation fingerprint prefix;
- clips per scenario;
- selected layers; and
- latest artifact generation timestamp.

Main charts and interactions:

1. Evaluation-return line chart: x is training step, y is mean selection return,
   shaded 95% episode interval, one line per seed, and vertical markers for
   weak/medium/strong checkpoints. Hover includes checkpoint ID, episode count,
   mean, standard deviation, and config hash.
2. Final-performance point/interval chart: selected capabilities on x, final
   return on y, individual episode points optionally visible.
3. Checkpoint-selection explanation: a compact record of each rule, selected
   step, selection score, and minimum-improvement check.
4. Study flow status: configuration → runs → scenarios → activations → metrics,
   with each stage linked to provenance details.
5. Data-quality panel: seed count warning, class balance, missing values,
   checksum status, clip overlap checks, and any metric warnings.

The page must visibly warn that a one-seed study is exploratory.

### Page 2 — Scenario explorer

Purpose: let a reviewer verify what baseline, danger, and goal mean before
looking at latent results.

Controls:

- scenario, split, source policy, review status, and clip ID;
- relative frame slider; and
- play/pause with FPS from dashboard YAML.

Display:

1. current preprocessed four-frame stack, with the newest frame emphasized;
2. optional source RGB frame if it was retained;
3. clip timeline with current step, burn-in boundary, anchor event, rewards,
   life-loss, goal, and terminal markers;
4. metadata: source episode, collection seed, collector checkpoint, action,
   reward, lives, start/anchor/end steps, curator, and checksum;
5. contact-sheet view for comparing all clips in one scenario at their anchor
   frame; and
6. per-scenario counts by source and split.

The player reads frozen observations; it does not replay the emulator.

### Page 3 — Within-scenario consistency

Purpose: test H1 and H3.

Views:

1. Capability comparison: point and 95% clip-bootstrap interval for cosine,
   correlation, normalized Euclidean, or centroid dispersion. Facet by
   scenario; optionally facet or select layer.
2. Distribution view: violin/box plot of clip-level values, with raw clip
   points keyed by clip ID. Do not plot millions of frame-pair points.
3. Similarity heatmap: a deterministic, capped sample of clip centroids or
   aligned frames, with scenario boundary annotations.
4. Temporal-lag chart: lag on x and temporal cosine on y, one line per
   capability, interval by clips.
5. Visual-change control: latent change versus input pixel change scatter and
   binned trend. Hover identifies clip and step.

Every chart states whether higher or lower means “more consistent,” pair-set
definition, normalization, number of independent clips, and bootstrap seed.

### Page 4 — Scenario separation

Purpose: test H2.

Views:

1. Within/between similarity gap by capability, layer, and pair-set.
2. Silhouette score with clip-bootstrap intervals.
3. Held-out nearest-centroid accuracy with chance line at one-third.
4. Confusion matrix normalized by true class, with raw counts available in
   hover.
5. PCA scatter of clip centroids, colored by scenario and symbolized by
   capability; axis labels include explained variance.
6. Optional UMAP scatter with a permanent “visualization only” warning and all
   configured parameters shown.
7. Per-scenario centroid-distance matrix to reveal which labels separate.

Train/test split and fit scope appear in a provenance expander. A user cannot
switch to frame-random splitting from the dashboard.

### Page 5 — Cross-checkpoint geometry

Purpose: test H4 without assuming neuron-wise alignment.

Views:

1. Checkpoint-by-checkpoint heatmap for one metric/layer.
2. Layer-by-layer heatmap for one checkpoint pair.
3. Learning-trajectory line: similarity to strong checkpoint versus training
   step when non-selected checkpoints were extracted.
4. Metric agreement small multiples for linear CKA, RSA, SVCCA, and normalized
   Procrustes.
5. Raw-coordinate cosine diagnostic in a separately labeled section, disabled
   by default and accompanied by a basis-drift warning.

Only comparisons with identical ordered sample IDs are rendered. Alignment
validation failures appear as errors, never as missing heatmap cells.

### Page 6 — Provenance and exports

Purpose: make every figure auditable.

Display:

- source and resolved YAML with secrets/absolute user paths redacted;
- config, checkpoint, observation, activation, and metric checksums;
- Python, package, ALE, Gymnasium, SB3, CUDA, OS, and git versions;
- artifact dependency graph;
- scenario review and validation summaries; and
- exact regeneration commands.

Downloads:

- currently filtered summary rows as CSV;
- chart as PNG through Plotly's supported export path;
- resolved config and artifact manifest; and
- no raw checkpoints or large activation arrays from the web UI.

### Dashboard chart contract

Each `charts.py` function:

1. accepts a filtered DataFrame plus an immutable chart-options dataclass;
2. validates required columns;
3. returns a Plotly `Figure` without calling Streamlit;
4. uses fixed category/color order;
5. labels metric direction and independent sample count;
6. includes provenance IDs in `customdata`; and
7. returns an annotated empty figure when no rows match.

Unit-test figure traces, category order, axis labels, and empty states. Use
Streamlit's app testing support for page smoke tests and filter interactions.

## 11. Implementation phases

### Phase 1 — Project and environment

- Add `pyproject.toml`, locked dependencies, package skeleton, CLI, strict
  configuration models, logging, and artifact manifest utilities.
- Add the five YAML document models, reference resolver, canonical serializer,
  config hash, and `validate-config`.
- Add explicit wrapper and algorithm registries with unknown-ID errors.
- Implement the Atari environment adapter and `doctor`.
- Add a 100-step environment smoke test and verify no double frame skip.

**Exit:** all example YAML resolves without mutation, unknown fields fail,
installation and `doctor` pass on CPU, and the environment fingerprint is
stable across two equivalent constructions.

### Phase 2 — Training and capability selection

- Implement the algorithm-neutral runner and the `sb3.dqn` and `sb3.ppo`
  adapters.
- Wrap SB3 training with generic evaluation/checkpoint callbacks used unchanged
  by DQN and PPO.
- Persist resolved config, versions, evaluations, and checkpoint checksums.
- Implement deterministic selection/final evaluation suites.
- Implement and plot weak/medium/strong selection.
- Run the shared adapter contract suite against DQN and PPO; keep the PPO config
  marked as an example until it passes.

**Exit:** a short training smoke run can resume, produces valid artifacts, and
selects three ordered checkpoints or gives a clear insufficient-learning error
for both built-in adapters.

### Phase 3 — Scenario dataset

- Save complete candidate trajectories from the strong policy.
- Add a simple local curation command that exports contact sheets or videos and
  a YAML manifest template; avoid building a custom labeling web app.
- Freeze canonical preprocessed clips and implement full validation.
- Curate and independently review Frogger v1.

**Exit:** at least 20 validated, balanced clips per scenario load identically in
two processes and have no overlap with evaluation seeds.

### Phase 4 — Activation extraction

- Implement logical PyTorch hook mappings for SB3 DQN.
- Extract batched activation tensors with pooling and provenance.
- Verify extraction does not change predicted Q-values or model parameters.
- Add recurrent fields to the interface and schemas without implementing an
  RNN adapter yet.

**Exit:** all selected checkpoints produce aligned, finite arrays for every
configured layer, and rerunning yields matching checksums within the declared
determinism tolerance.

### Phase 5 — Metrics

- Implement pair construction, feature preprocessing, metric registry, and
  clip-level bootstrap.
- Produce tidy summary, cross-checkpoint, pair diagnostic, and embedding files.
- Add input-space temporal-change controls.

**Exit:** synthetic invariance tests pass, all hypotheses map to at least one
saved result, and no analysis split leaks clips.

### Phase 6 — Dashboard and reproducibility

- Implement typed dashboard data loading, state, pure Plotly chart functions,
  and the six dashboard pages against artifact files.
- Add empty/error states and one-seed warnings.
- Add filtered CSV/chart export and the provenance dependency view.
- Run a clean end-to-end smoke study and document exact reproduction commands.

**Exit:** the dashboard opens from saved artifacts on CPU with no model or
environment loaded, and all displayed aggregates trace back to manifest
checksums.

## 12. File-by-file implementation blueprint

Implement in the order below. Do not begin a later group until the listed tests
for the current group pass.

### 12.1 Packaging and shared errors

`pyproject.toml`

- Define package `drl-similarity` with `src` layout and console script
  `drl-sim = drl_similarity.cli:app`.
- Core dependencies: Gymnasium Atari extra, ale-py, PyTorch,
  Stable-Baselines3, NumPy, Pandas, PyArrow, SciPy, scikit-learn, Pydantic,
  PyYAML, rsatoolbox, Plotly, Kaleido, Streamlit, Typer, Pillow, and imageio.
- Optional dependency groups: `dev` for pytest/ruff/mypy and `umap` for
  umap-learn.
- Configure Ruff, mypy, and pytest. Pin compatible major/minor ranges and commit
  `uv.lock`.

`src/drl_similarity/errors.py`

- Define `DrlSimilarityError` and subclasses `ConfigError`,
  `RegistryError`, `ArtifactError`, `EnvironmentError`,
  `CheckpointError`, `ScenarioError`, `ExtractionError`, and `AnalysisError`.
- Store an actionable message and optional context mapping.
- CLI catches `DrlSimilarityError`; programming errors retain tracebacks.

`src/drl_similarity/logging.py`

- `configure_logging(level: str, log_file: Path | None)`.
- Include timestamp, level, stage, and config hash in structured text.
- Never log complete observation or activation arrays.

### 12.2 Configuration system

`src/drl_similarity/config.py`

- Define strict Pydantic models for all YAML shown in section 5.
- Use discriminated unions for document `kind`, wrapper specs, pooling specs,
  metric specs, and embedding specs.
- Define `ConfigReference(ref: Path)`.
- Implement `read_yaml`, `parse_document`, `resolve_references`,
  `validate_cross_document`, `canonicalize_config`, `hash_config`,
  `write_resolved_config`, and public `load_config`.
- Preserve source file paths separately from resolved values.
- Error messages include dotted field path, received value, and expected type.

`tests/unit/test_config.py`

- Valid document for every kind.
- Unknown root and nested field.
- Wrong kind and schema version.
- Relative reference resolution.
- Missing reference and cycle.
- Stable hash despite mapping key order.
- Different hash after one scientific setting changes.
- Rejection of arbitrary Python tags/import strings.

`configs/**/*.yaml`

- Add the concrete files from section 5.
- Add comments explaining values but ensure comments are not part of the hash.
- Keep the PPO file named `.example.yaml` until its adapter contract tests pass,
  then rename it to `ppo-nature-cnn.yaml`.

### 12.3 Registries

`src/drl_similarity/registry.py`

- Generic `Registry[T]` with `register(id, factory)`, `get(id)`, and `ids()`.
- Reject duplicate IDs at import time.
- Unknown-ID errors list sorted valid IDs and the config field that requested
  the ID.

`src/drl_similarity/environments/wrappers.py`

- Register `record_episode_statistics`, `sb3_atari`, and `frame_stack`.
- Each factory accepts an environment and validated kwargs.
- Record an ordered `WrapperDescriptor` after application.
- Compute effective frame skip from base and wrapper settings and reject double
  skip.

`src/drl_similarity/agents/registry.py`

- Register `sb3.dqn` immediately.
- Register `sb3.ppo` when implemented; both are required before the full
  framework milestone is complete.
- Expose `get_algorithm_adapter`.

`src/drl_similarity/metrics/registry.py`

- Register exactly the metric IDs listed in section 8.
- Store direction, range, scopes, alignment requirement, and factory.

### 12.4 Environment construction

`src/drl_similarity/environments/base.py`

- Define `EnvironmentDescription`, `EnvironmentFingerprints`, and the protocol
  from section 4.
- Separate observation fingerprint from full runtime fingerprint.

`src/drl_similarity/environments/atari.py`

- `register_environment_modules(names)`.
- `make_environment(config, seed, render_mode=None)`.
- Seed environment reset and action space.
- Apply wrappers in YAML order.
- `inspect_environment` by one reset and a legal step.
- `assert_expected_environment`.
- `compute_environment_fingerprints` from canonical descriptors, not Python
  object reprs.
- Always close temporary environments in `finally`.

`tests/unit/test_environment.py`

- Use a fake registered pixel environment for wrapper/fingerprint tests.
- Verify shape, dtype, action space, deterministic seeded reset, and frame skip.
- Mark real ALE smoke tests `atari` so CI may skip them when ROM support is not
  installed.

### 12.5 Artifact and provenance layer

`src/drl_similarity/artifacts/manifest.py`

- Define versioned `ArtifactManifest`, `ArtifactInput`, and `ArtifactOutput`.
- Store checksums, byte sizes, schema versions, config hash, git commit,
  package versions, creation time, and stage.
- `sha256_file` streams files rather than reading them all into memory.

`src/drl_similarity/artifacts/store.py`

- Resolve run and study artifact IDs from config ID/hash.
- Create output in a temporary sibling directory.
- Write all files, manifest last, fsync where available, then atomically rename.
- On collision follow YAML: `error` by default; `--force` creates a new version
  suffix and never deletes old output.
- `verify_artifact(path)` checks every declared input/output checksum.

### 12.6 Algorithm adapter and training runner

`src/drl_similarity/agents/base.py`

- Define `AlgorithmAdapter`, `LoadedAgent`, capability metadata, layer metadata,
  and prediction/extraction result dataclasses.
- Include `recurrent`, supported action-space kinds, and backend version.

`src/drl_similarity/agents/sb3/base.py`

- Shared conversion between NumPy observations and SB3 policy tensors.
- Shared deterministic prediction and PyTorch hook manager.
- Hook manager registers in a context manager and always removes hooks.
- Validate one activation per requested logical layer and preserve batch order.

`src/drl_similarity/agents/sb3/dqn.py`

- Validate discrete action space and DQN kwargs.
- Map friendly policy/extractor IDs to SB3 objects.
- Build/load/save SB3 DQN.
- Map logical layers to actual modules after model construction; assert mapping
  by a dry forward pass.
- Return Q-values separately as functional output.

`src/drl_similarity/agents/sb3/ppo.py`

- Validate PPO kwargs and any Gymnasium action space supported by SB3 PPO.
- Build/load/save SB3 PPO through the same adapter surface.
- Expose shared CNN features, policy latent, value latent, action logits or
  distribution parameters, and value estimate under stable logical names.
- Keep policy/value branches distinct even if the configured extractor shares
  parameters.
- Pass the same contract tests as DQN before registration is enabled.

`src/drl_similarity/training/callbacks.py`

- `CheckpointEvaluationCallback` receives generic save and evaluation callables.
- At configured steps: save checkpoint atomically, run selection suite, append
  episode rows, and flush Parquet.
- Handle interruption by leaving the latest complete checkpoint resumable.

`src/drl_similarity/training/evaluate.py`

- `evaluate_checkpoint(agent, env_factory, suite_config)`.
- Recreate environment per seed/episode, collect raw return and declared game
  events, enforce max steps, and always close.
- Never use clipped training reward for reported return.

`src/drl_similarity/training/runner.py`

- Resolve experiment, environment, adapter, artifact store, and callbacks.
- Validate before allocating a replay buffer/GPU.
- Apply all random seeds.
- Build or resume, run training, evaluate final checkpoint if configured,
  finalize manifest, and print run directory.

`src/drl_similarity/training/selection.py`

- Aggregate selection episodes by checkpoint.
- Implement each named selection rule as a registered pure function.
- Validate weak step < medium step < strong step and minimum improvement.
- Write `selected-checkpoints.yaml` containing rules and evidence.

`tests/contract/test_algorithm_adapter.py`

- Parameterize over every registered adapter.
- Fake image/discrete environment.
- Build, short learn, save, load, deterministic predict.
- Enumerate/extract each logical layer.
- Compare batched and unbatched outputs.
- Confirm extraction does not alter parameters or predictions.

### 12.7 Scenario collection and curation

`src/drl_similarity/scenarios/schema.py`

- Implement dataset/clip/event manifest models from section 6.
- Store labels as configured strings rather than a hard-coded enum, while the
  first study config restricts them to baseline/danger/goal.

`src/drl_similarity/scenarios/collect.py`

- Load the collector checkpoint through the configured algorithm adapter.
- Capture full candidate episodes: source RGB if enabled, canonical
  preprocessed observations, actions, raw rewards, lives, terminal flags, and
  event candidates.
- Write candidate trajectory NPZ plus metadata, contact sheets, and optional
  MP4 previews.
- Generate a manifest template; never auto-approve scenario labels.

`src/drl_similarity/scenarios/validate.py`

- Check schema, checksums, shapes, fixed lengths, event bounds, unique IDs,
  source overlap, split isolation, class balance, minimum counts, review status,
  seed separation, and observation fingerprint.
- Print all failures in one pass so curation can fix them together.

### 12.8 Activation extraction

`src/drl_similarity/activations/pooling.py`

- `none` accepts rank two only after batch flatten semantics are declared.
- `flatten` preserves deterministic channel/spatial order.
- `global_average` averages spatial axes and retains channels.
- Reject ambiguous ranks with checkpoint/layer/shape in the error.

`src/drl_similarity/activations/extract.py`

- Verify study, run, scenario, and checkpoint manifests first.
- Load canonical clips in manifest order.
- For feed-forward models, flatten clip/step to batches and restore shape.
- For recurrent models, process complete ordered clips with burn-in and reset
  masks; never mix hidden state across clips.
- Apply configured pooling, verify finite values, and save arrays/metadata.
- Re-run one small batch and compare predictions to prove hooks are passive.
- Skip only when an existing artifact manifest exactly matches every input.

### 12.9 Analysis

`src/drl_similarity/metrics/vector.py`

- Implement vector and classification metrics as small pure wrappers around
  SciPy/scikit-learn.
- Return values plus warnings such as zero vectors or undefined silhouette.

`src/drl_similarity/metrics/geometry.py`

- Wrap rsatoolbox RSA.
- Implement SVCCA from SVD variance reduction plus CCA.
- Implement normalized orthogonal Procrustes.
- Implement linear CKA in one documented function with epsilon handling.
- Validate identical sample-ID ordering before every cross-checkpoint metric.

`src/drl_similarity/metrics/pairs.py`

- Generate the three configured pair sets.
- Exclude same-clip pairs from cross-clip sets.
- Balance between-scenario pairs with a deterministic RNG.
- Emit pair IDs so estimates can be audited.

`src/drl_similarity/metrics/bootstrap.py`

- Resample clip IDs stratified by scenario.
- Rebuild dependent pair sets inside each replicate.
- Return estimate, percentile interval, replicate count, successful replicate
  count, and warnings.

`src/drl_similarity/analysis.py`

- Orchestrate preprocessing fit on train clips, metric scopes, bootstraps,
  embeddings, tidy Parquet writes, and final manifest.
- Never silently drop NaN values; count and report them.
- Sort output rows by stable identifier columns before writing.

### 12.10 Dashboard

Implement the module and page responsibilities from section 10 in this order:

1. typed readers and fixture artifacts;
2. overview page;
3. scenario player;
4. consistency page;
5. separation page;
6. geometry page;
7. provenance/export page; and
8. global filters, caching, and final visual consistency pass.

Create one small fixture study under `tests/fixtures/study/` containing all
schemas but no proprietary ROM or checkpoint. Dashboard tests use only this
fixture.

### 12.11 CLI and orchestration

`src/drl_similarity/cli.py`

- Implement the thin commands from section 9.
- Print paths in a form terminals can open.
- `validate-config` prints the resolved kind, ID, references, and hash.
- `doctor` reports each check and a final pass/fail count.
- `dashboard` checks artifacts before launching Streamlit.

`src/drl_similarity/pipeline.py`

- Implement stage DAG and idempotent `run-study`.
- A stage is complete only if its manifest exists and verifies.
- Print skipped, planned, executed, and blocked stages.
- Do not catch keyboard interrupt during training; allow runner cleanup and
  return the standard interrupt exit code.

## 13. Test strategy

### Unit tests

- Strict YAML parsing, reference cycles, canonical hashes, unknown registry IDs,
  and cross-document compatibility.
- Environment wrapper order, effective frame skip, shape, dtype, and fingerprint.
- Shared algorithm-adapter contract tests for DQN and PPO.
- Checkpoint selector on improving, flat, noisy, tied, and insufficient runs.
- Scenario schema, checksums, split isolation, and overlap detection.
- Hook resolution, pooling shapes, batching equivalence, and no model mutation.
- Pair construction excludes same-clip pairs and balances scenario pairs.
- Bootstrap resamples clip IDs rather than frames.
- Metric behavior on identical, noisy, scaled, permuted, and orthogonally
  rotated synthetic matrices.
- CKA/RSA/SVCCA/Procrustes alignment failures on shuffled sample IDs.
- Artifact cache invalidation when any input checksum changes.

Expected invariances are asserted explicitly. For example, linear CKA should be
near one under orthogonal rotation and isotropic scaling, while raw coordinate
cosine need not be.

### Integration tests

- A tiny deterministic fake image environment runs the complete pipeline
  quickly in CI without Atari.
- An optional local Atari test resets, steps, saves a short clip, reloads it,
  and checks observation equality.
- A short SB3 run saves/resumes and extracts activations from two checkpoints.
- The same tiny experiment config runs with DQN and PPO algorithm references
  without changing pipeline code.
- The dashboard's data-loading functions open a fixture experiment and produce
  expected filtered frames.
- Streamlit app smoke tests open all six pages, change global filters, and
  exercise empty/error states.

Do not put a multi-million-step Frogger run in CI. Full experiment manifests
record the commit and test status used to generate them.

## 14. Scientific acceptance checklist

- Checkpoint labels derive from held-out return, not training step alone.
- Strong materially outperforms weak on selection and disjoint final suites.
- Every checkpoint sees byte-identical scenario observations.
- Switching a study from DQN to PPO requires changing only an algorithm YAML
  reference; pipeline and dashboard code contain no algorithm-name branches.
- Each scenario has enough independent clips and balanced clip lengths/sources.
- Scenario review occurs without activation or metric results visible.
- Pairwise analyses exclude same-clip pairs unless explicitly labeled temporal.
- Confidence intervals resample clips.
- Cross-checkpoint claims use representation-aware metrics.
- Input visual change is reported with temporal smoothness.
- Dashboard plots identify checkpoint, layer, metric direction, normalization,
  independent sample count, and uncertainty.
- Conclusions state that one seed is exploratory and avoid causal or
  state-of-the-art claims.

## 15. Explicit non-goals

The MVP will not implement distributed RL, a generic plugin marketplace,
hyperparameter optimization, cloud experiment tracking, automated semantic
scenario detection, learned probes beyond nearest centroid, model stitching,
causal interventions, every available similarity metric, or RNN training.

The next model-family study should add FF/CNN/RNN adapters only after the
Frogger CNN pipeline satisfies the acceptance checklist.
