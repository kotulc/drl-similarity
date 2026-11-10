# User guide

How to run a study, what every YAML option means, and how to extend each
component. Design rationale lives in the [implementation plan](implementation-plan.md);
module internals in [modules.md](modules.md).

## 1. Install and check

```bash
uv sync                                                          # Python 3.11, locked deps
uv run drl-sim doctor --config configs/experiments/frogger-dqn-demo.yaml
```

`doctor` checks packages, registries, CUDA, and (for environment/experiment
configs) builds the environment and takes one step.

## 2. Run the demo end to end (CPU, ~20 minutes)

```bash
E=configs/experiments/frogger-dqn-demo.yaml
S=configs/studies/frogger-demo.yaml

uv run drl-sim train --config $E                 # 120k steps, checkpoint + eval every 10k
uv run drl-sim select-checkpoints --config $E    # weak / medium / strong
uv run drl-sim evaluate --config $E              # final suite on the selected checkpoints
uv run drl-sim collect-scenarios --config $S     # candidate clips + manifest template
uv run python scripts/approve_demo_clips.py --config $S   # DEMO ONLY: skips human review
uv run drl-sim run-study --config $S             # validate, extract, analyze (idempotent)
uv run drl-sim dashboard --config $S             # prints a local URL
```

For the real study use `frogger-dqn.yaml` / `frogger-v1.yaml` and replace the
approval script with manual review (section 4).

## 3. Commands

| Command | Config | Writes |
|---|---|---|
| `validate-config` | any | nothing |
| `doctor` | any | nothing |
| `train [--resume CKPT.zip]` | experiment | `runs/<id>/` checkpoints, `evaluations.parquet`, manifest |
| `select-checkpoints` | experiment | `runs/<id>/selected-checkpoints.yaml` |
| `evaluate` | experiment | final-suite rows in `evaluations.parquet` |
| `collect-scenarios` | study | manifest template + `data/scenarios/<name>/` |
| `validate-scenarios` | study | nothing (lists every issue in one pass) |
| `extract [--force]` | study | `artifacts/<study>/activations[-vN]/` |
| `analyze [--force]` | study | `artifacts/<study>/metrics[-vN]/` |
| `run-study` | study | whatever of extract/analyze is missing or stale |
| `dashboard` | study | nothing (read-only) |

- `--dry-run` (train, select, evaluate, collect, extract, analyze, run-study) validates and prints the plan.
- `--force` never overwrites; it publishes a new `-vN` version. Readers use the latest.
- `--log-level` goes before the command: `drl-sim --log-level DEBUG train ...`.
- A run directory is immutable: re-running `train` fails unless `--resume` points at one of its checkpoints with an unchanged config.

## 4. Curating scenarios

1. `collect-scenarios` writes every candidate as `review_status: pending`.
2. Inspect `data/scenarios/<name>/candidates/contact_sheets/*.png` (and `previews/*.mp4`).
3. In the manifest set `review_status: approved | rejected` and `curator: <name>[, <name>]`. Do this before looking at any metric.
4. `validate-scenarios` must pass: balanced counts, minimum per label, both splits per label, no overlapping source windows, matching checksums and fingerprints, seeds disjoint from evaluation seeds.

## 5. Configuration reference

Every document starts with `schema_version: 1`, `kind`, and `id`. Unknown
fields are errors. Paths and `ref`s resolve relative to the declaring file.

### Environment (`kind: environment`)

| Field | Values |
|---|---|
| `registration_modules` | modules imported before `gymnasium.make`, e.g. `[ale_py]` |
| `env_id`, `make_kwargs` | passed to `gymnasium.make` |
| `wrappers[].id` | `record_episode_statistics`, `sb3_atari`, `frame_stack` (applied in order) |
| `sb3_atari` kwargs | `noop_max`, `frame_skip`, `screen_size`, `terminal_on_life_loss`, `clip_reward`, `action_repeat_probability` |
| `frame_stack` kwargs | `num_stack` |
| `expected` | `observation_shape`, `observation_dtype`, `action_space` (`discrete`/`box`), `effective_frame_skip`; asserted after construction |

Only one component may skip frames: `make_kwargs.frameskip > 1` together with `sb3_atari.frame_skip > 1` is rejected.

### Algorithm (`kind: algorithm`)

| Field | Values |
|---|---|
| `adapter` | `sb3.dqn`, `sb3.ppo` |
| `policy` | `CnnPolicy` |
| `policy_kwargs.features_extractor` | `nature_cnn`; other keys pass to the SB3 policy |
| `model_kwargs` | allow-listed per adapter (below); unknown keys fail before training |
| `capabilities` | `action_spaces`, `recurrent`, `activation_layers` (documentation + compatibility check) |

| Adapter | `model_kwargs` | Logical layers |
|---|---|---|
| `sb3.dqn` | `learning_rate`, `buffer_size`, `learning_starts`, `batch_size`, `tau`, `gamma`, `train_freq`, `gradient_steps`, `target_update_interval`, `exploration_fraction`, `exploration_initial_eps`, `exploration_final_eps`, `max_grad_norm`, `optimize_memory_usage` | `conv1`, `conv2`, `conv3`, `conv_flat`, `policy_features`, `q_values` |
| `sb3.ppo` | `learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `clip_range_vf`, `normalize_advantage`, `ent_coef`, `vf_coef`, `max_grad_norm`, `target_kl` | `conv1`, `conv2`, `conv3`, `conv_flat`, `policy_features`, `policy_latent`, `value_latent`, `action_logits`, `value` |

`conv1-3` are post-ReLU maps; `policy_features` is the 512-d NatureCNN output.

### Experiment (`kind: experiment`)

| Field | Meaning |
|---|---|
| `environment.ref`, `algorithm.ref` | referenced documents |
| `seed`, `device` | `device`: `auto`, `cpu`, `cuda` |
| `training.total_timesteps` | budget (PPO may overshoot by one rollout) |
| `training.log_interval_steps`, `progress_bar` | progress output |
| `evaluation.selection.interval_steps` | checkpoint + selection-suite cadence (falls back to `training.checkpoint_interval_steps`) |
| `evaluation.{selection,final}.seeds`, `episodes_per_seed` | disjoint seed sets |
| `evaluation.max_agent_steps`, `deterministic_actions` | episode cap, greedy actions |
| `checkpoint_selection.{weak,medium,strong}.rule` | `earliest_complete_after_learning_starts`, `nearest_performance_midpoint`, `highest_mean_selection_return` |
| `checkpoint_selection.medium.smoothing_window` | rolling-mean window over checkpoints |
| `checkpoint_selection.minimum_strong_improvement` | required strong − weak return |
| `artifacts.root` | parent of the run directory `<root>/<id>` |
| `reproducibility` | `torch_deterministic`, `cudnn_benchmark`, `capture_packages`, `capture_git` |

### Study (`kind: study`)

| Field | Meaning |
|---|---|
| `experiments[].run_dir` | completed run directories |
| `scenarios.manifest` | curated manifest path (data lives in the sibling `<stem>/` folder) |
| `scenarios.labels`, `minimum_clips_per_label`, `clip_steps`, `burn_in_steps` | dataset shape |
| `scenarios.collection` | `experiment_index`, `checkpoint`, `seeds`, `episodes_per_seed`, `max_agent_steps`, `retain_source_rgb`, `write_video_previews` |
| `scenarios.collection.candidate_windows.<label>` | `anchor`: `life_start` / `life_lost` / `positive_goal_event`; `offset_steps` (window start relative to the anchor); `min_reward` (goal fires when reward > value) |
| `scenarios.review` | `require_approved`, `minimum_reviewers` |
| `scenarios.split` | `train_fraction`, `seed` (split is per clip, stratified by label) |
| `extraction` | `batch_size`, `device`, `layers[]` |
| `extraction.layers[]` | `name`, `source` or `sources_by_adapter`, `pooling` (`none`, `flatten`, `global_average`), `required` |
| `analysis.pair_sets` | `cross_clip_all`, `cross_clip_aligned_step`, `clip_centroids` |
| `analysis.metrics[]` | `id`, `kwargs`, `enabled` (table below) |
| `analysis.bootstrap` | `replicates`, `confidence_level`, `seed` (whole clips, stratified by scenario) |
| `analysis.embeddings[]` | `pca` (`components`), `umap` (`components`, `neighbors`, `min_dist`, `seed`; needs the `umap` extra) |
| `dashboard.ref`, `artifacts.root` | dashboard document, artifact root |

| Metric | Kind | kwargs |
|---|---|---|
| `cosine`, `correlation` | similarity | none |
| `normalized_euclidean` | distance | `standardize_on`: `train_clips` / `all_clips` |
| `centroid_dispersion` | distance | none |
| `temporal_cosine` | similarity | `lags: [int]` |
| `within_between_gap` | score | `base_metric`: `cosine` / `correlation` |
| `silhouette`, `nearest_centroid_accuracy` | score | none |
| `linear_cka`, `orthogonal_procrustes` | similarity / distance | `sample_unit`: `frame` / `clip_centroid`; `max_steps_per_clip` |
| `rsa_spearman` | similarity | above + `first_order_distance`: `correlation` / `euclidean` / `cosine` |
| `svcca` | similarity | above + `variance_retained` |

### Dashboard (`kind: dashboard`)

| Field | Meaning |
|---|---|
| `title`, `default_page` | page ids: `similarity_matrix`, `scenario_explorer`, `training_overview`, `scenario_separation`, `cross_checkpoint`, `provenance` |
| `default_filters` | `checkpoint`, `layer`, `row_scenario`, `column_scenario`, `frames_per_case` (1-3), `primary_metric`, `pair_set`; must exist in the study |
| `theme` | `capability_order`, `scenario_order` |
| `charts` | `show_raw_points`, `confidence_intervals`, `max_heatmap_samples`, `clip_fps` |
| `downloads` | `enabled`, `formats` (`csv`, `png`) |

## 6. Customizing and extending

### Change hyperparameters, budget, or seeds

Copy a YAML, change values, give it a new `id`. Example: a second seed.

```yaml
# configs/experiments/frogger-dqn-seed-2.yaml  (copy of frogger-dqn.yaml)
id: frogger-dqn-seed-2
seed: 2
```

Add it to the study under `experiments:`; the dashboard gains a run selector.

### Switch or compare algorithms

Point the experiment at another algorithm document; nothing else changes.

```yaml
algorithm:
  ref: ../algorithms/ppo-nature-cnn.yaml
```

For a DQN-vs-PPO study, name layers semantically:

```yaml
extraction:
  layers:
    - name: action_output
      sources_by_adapter: {sb3.dqn: q_values, sb3.ppo: action_logits}
    - name: value_head
      source: value
      required: false      # skipped for adapters without it
```

### Add an environment

Usually YAML only: set `registration_modules`, `env_id`, `wrappers`, `expected`.
A new wrapper is one registered factory in [wrappers.py](../src/drl_similarity/environments/wrappers.py):

```python
def _grayscale(env: gym.Env, kwargs: dict[str, Any]) -> gym.Env:
    """Convert RGB observations to grayscale."""
    return GrayscaleObservation(env, **kwargs)

WRAPPER_REGISTRY.register("grayscale", _grayscale)
```

A wrapper that repeats actions must also be listed in `FRAME_SKIP_KWARGS`.

### Add an algorithm adapter

Subclass `Sb3Adapter` and register it ([agents/sb3/](../src/drl_similarity/agents/sb3/), [registry.py](../src/drl_similarity/agents/registry.py)):

```python
class Sb3A2cAdapter(Sb3PpoAdapter):
    """A2C shares PPO's actor-critic policy and logical layers."""

    adapter_id = "sb3.a2c"
    model_class = A2C
    allowed_kwargs = frozenset({"learning_rate", "n_steps", "gamma", "gae_lambda", "ent_coef", "vf_coef"})
```

```python
ALGORITHM_ADAPTERS.register("sb3.a2c", make_sb3_a2c_adapter)
```

Add its kwargs/layers to `tests/conftest.py`; the contract suite and the
pipeline test then run against it automatically. A non-SB3 backend implements
the `AlgorithmAdapter` protocol in [agents/base.py](../src/drl_similarity/agents/base.py) directly.

### Add a checkpoint selection rule

Register a function `(table, context) -> row` in [selection.py](../src/drl_similarity/training/selection.py):

```python
def last_checkpoint(table: pd.DataFrame, context: dict[str, Any]) -> pd.Series:
    """Strong: the final checkpoint regardless of score."""
    return table.iloc[-1]

SELECTION_RULES.register("last_checkpoint", last_checkpoint)
```

`table` has `checkpoint_id`, `training_step`, `mean_return`, `std_return`,
`n_episodes`, `smoothed_return`; the medium rule also receives `weak` and `strong`.

### Add scenario labels or anchors

- New label from existing anchors: YAML only (`labels` + one `candidate_windows` entry).
- New anchor event: add the detection in `_Collector._rollout` and the name to `ANCHOR_EVENTS` ([collect.py](../src/drl_similarity/scenarios/collect.py)) and to `CandidateWindowConfig.anchor` ([config.py](../src/drl_similarity/config.py)).
- Dashboard colors for new labels are assigned from a fixed fallback palette; set `theme.scenario_order`.

### Add a metric

1. Implement a pure function in [metrics/vector.py](../src/drl_similarity/metrics/vector.py) or [geometry.py](../src/drl_similarity/metrics/geometry.py).
2. Register a `MetricDefinition` (kind, range, scopes, kwargs) in [metrics/registry.py](../src/drl_similarity/metrics/registry.py).
3. Wire it into the orchestrator: `PAIRWISE_METRICS` / `GEOMETRY_METRICS` and the matching branch in [analysis.py](../src/drl_similarity/analysis.py) (dispatch is by ID, not automatic).
4. Add a synthetic invariance test in `tests/unit/test_metrics.py`.

### Add a pooling method or embedding

- Pooling: a branch in `apply_pooling` plus the name in `POOLING_METHODS` and `LayerExtractionConfig.pooling`.
- Embedding: a branch in `_embedding_frames` plus the id in `EMBEDDING_IDS` ([analysis.py](../src/drl_similarity/analysis.py)).

### Add a dashboard page or chart

- Chart: a pure function in [charts.py](../src/drl_similarity/dashboard/charts.py) that takes a DataFrame + `ChartOptions` and returns a Plotly figure (no Streamlit calls).
- Page: a module in `dashboard/pages/` exposing `render(data)`, listed in `PAGE_ORDER`. Pages read only from `DashboardData`.

## 7. Outputs

| Path | Content |
|---|---|
| `runs/<experiment>/` | `resolved-config.yaml`, `environment_fingerprint.json`, `versions.json`, `evaluations.parquet`, `checkpoints/checkpoint_<step>.zip|.json`, `selected-checkpoints.yaml`, `manifest.json` |
| `data/scenarios/<name>.yaml` | curated manifest (commit this) |
| `data/scenarios/<name>/` | `observations.npz`, `candidates/` (contact sheets, previews, episodes, RGB) |
| `artifacts/<study>/activations/` | `<run>/<capability>/<layer>.npz` shaped `(clips, steps, features)`, `clips.parquet`, manifest |
| `artifacts/<study>/metrics/` | `summaries`, `cross_checkpoint`, `frame_pairs`, `embeddings` parquet tables, manifest |

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Run directory already exists` | pass `--resume <checkpoint.zip>` or change the experiment `id` |
| `Insufficient learning` | strong does not beat an earlier weak checkpoint by `minimum_strong_improvement`; train longer or lower the threshold |
| `No checkpoint lies between weak and strong` | checkpoint more often (`selection.interval_steps`) |
| many `duplicate_observations` skipped | greedy policy in a deterministic emulator replays identical frames; add collection seeds or enable sticky actions |
| `Existing ... artifact is stale` | config changed since extraction/analysis; rerun with `--force` or use `run-study` |
| resumed DQN dips briefly | the replay buffer is not checkpointed and refills after resume |
| "Page not found" dialog on a pasted deep link | Streamlit quirk; the page still renders, use the sidebar |
