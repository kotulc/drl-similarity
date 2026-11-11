# User guide

This framework trains a reinforcement-learning agent on Atari Frogger, then asks
how its internal representations differ between situations (a calm start, an
imminent death, reaching a goal) and between a weak, medium, and strong version
of the same agent. Everything is driven by YAML files; each command reads one
YAML file and writes files to disk that the next command uses.

- Sections 0-2: what the pieces are, installing, and a full walkthrough.
- Sections 3-8: reference (commands, every YAML option, extending, outputs, troubleshooting).
- Design rationale: [implementation plan](implementation-plan.md). Code: [modules.md](modules.md).

## 0. How it fits together

| Term | Meaning | Lives in |
|---|---|---|
| **Experiment** | one training run: environment + algorithm + seed + budget | `configs/experiments/*.yaml` → `runs/<id>/` |
| **Checkpoint** | saved model weights at a training step | `runs/<id>/checkpoints/` |
| **Weak / medium / strong** | three checkpoints picked by evaluation score, not by step | `runs/<id>/selected-checkpoints.yaml` |
| **Scenario clip** | a short, fixed sequence of game frames labelled `baseline`, `danger`, or `goal` | `data/scenarios/<name>.yaml` + `data/scenarios/<name>/` |
| **Study** | which runs and clips to compare, which layers to read, which metrics to compute | `configs/studies/*.yaml` → `artifacts/<id>/` |
| **Activations** | each checkpoint's internal layer outputs for every clip frame | `artifacts/<id>/activations/` |
| **Metrics** | similarity numbers (with confidence intervals) computed from activations | `artifacts/<id>/metrics/` |

The pipeline runs in this order; each stage refuses to start until the previous one's files exist and verify:

```text
train → select-checkpoints → evaluate → collect-scenarios → (you review clips) → run-study → dashboard
└──────────── experiment YAML ────────┘ └──────────────────── study YAML ──────────────────────────┘
```

Two configurations ship ready to use:

| Config pair | Purpose | Training | Clips per label |
|---|---|---|---|
| `frogger-dqn-demo.yaml` + `frogger-demo.yaml` | learn the workflow on a CPU | 120k steps (~15 min CPU) | 8 |
| `frogger-dqn.yaml` + `frogger-v1.yaml` | the actual study | 10M steps (GPU, many hours) | 20 |

## 1. Install and check

Requires Python 3.11 (the pinned torch/numpy versions have no 3.13 builds).
If `py -3.11 --version` fails, install 3.11 from python.org; it can sit beside
other versions. Run everything from the repository root.

The `.venv` already in this checkout works; recreate it only if it is missing or broken.

**Create the environment and install** (once):

```powershell
# PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

```bash
# bash (Linux/macOS: replace Scripts with bin)
python3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

**Activate it** in every new terminal so `drl-sim` and `python` refer to the project:

| Shell | Command |
|---|---|
| PowerShell | `.\.venv\Scripts\Activate.ps1` |
| cmd | `.venv\Scripts\activate.bat` |
| bash (Windows) | `source .venv/Scripts/activate` |
| bash (Linux/macOS) | `source .venv/bin/activate` |

PowerShell blocks `Activate.ps1` by default ("running scripts is disabled").
Either allow local scripts once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or skip activation and
call the tools by path: `.\.venv\Scripts\drl-sim.exe ...`.

**Check the install**:

```text
drl-sim doctor --config configs/experiments/frogger-dqn-demo.yaml
```

This checks packages, the registries, CUDA, and builds the Frogger environment
and takes one step. Expect `doctor: 18/18 checks passed`.

Install options:

- Exact versions used for the demo: `pip install -r requirements-lock.txt`, then `pip install -e . --no-deps` (frozen on Windows with CPU PyTorch).
- GPU: install the CUDA build of torch from pytorch.org first, then `pip install -e ".[dev]"`.
- UMAP embeddings: `pip install -e ".[dev,umap]"`.

## 2. Walkthrough

After activation, every `drl-sim` line below runs unchanged in PowerShell,
cmd, and bash (forward slashes work everywhere). Only copying and opening files
differ by shell; both versions are given.

### 2a. Look at the existing demo results (2 minutes)

This checkout already contains a finished demo run, so you can see the end
result before producing your own.

```text
drl-sim dashboard --config configs/studies/frogger-demo.yaml
```

- **What it does:** checks the demo's artifacts, then starts a local web server. It never trains or computes anything.
- **Output:** a line like `Local URL: http://localhost:8501`. Open it in a browser. Stop the server with `Ctrl+C`.
- **What to look at:**
  - **Similarity matrix** (the start page): each cell compares two scenario types; click **Open** on a cell to see individual frame pairs below it, and **Inspect** on a frame pair to show it in the side panel.
  - **Scenario explorer:** the actual clips behind each label.
  - **Training and quality:** the learning curve and where weak, medium, and strong were taken.

### 2b. Create your own experiment and study

Running `train` on the demo config again fails with `Run directory already exists`:
run directories are never overwritten. Make your own copies with new IDs instead.

```powershell
# PowerShell
Copy-Item configs/experiments/frogger-dqn-demo.yaml configs/experiments/my-frogger.yaml
Copy-Item configs/studies/frogger-demo.yaml configs/studies/my-study.yaml
code configs/experiments/my-frogger.yaml configs/studies/my-study.yaml   # or any editor
```

```bash
# bash
cp configs/experiments/frogger-dqn-demo.yaml configs/experiments/my-frogger.yaml
cp configs/studies/frogger-demo.yaml configs/studies/my-study.yaml
```

Edit these lines (the IDs name the output folders, so they must be new):

| File | Line | Change to |
|---|---|---|
| `my-frogger.yaml` | `id: frogger-dqn-demo` | `id: my-frogger` |
| `my-study.yaml` | `id: frogger-demo-v1` | `id: my-study-v1` |
| `my-study.yaml` | `run_dir: ../../runs/frogger-dqn-demo` | `run_dir: ../../runs/my-frogger` |
| `my-study.yaml` | `manifest: ../../data/scenarios/frogger-demo.yaml` | `manifest: ../../data/scenarios/my-study.yaml` |

Check both files before running anything:

```text
drl-sim validate-config --config configs/experiments/my-frogger.yaml
drl-sim validate-config --config configs/studies/my-study.yaml
```

Expect `ok: configuration is valid`. A typo or unknown field fails here with the field name, not hours into training.
Paths inside a YAML file are relative to that file, which is why they start with `../../`.

### 2c. Train

```text
drl-sim train --config configs/experiments/my-frogger.yaml
```

- **What it does:** trains a DQN agent for `training.total_timesteps` (120,000 here). Every `evaluation.selection.interval_steps` (10,000) it saves a checkpoint and plays a few evaluation games.
- **Takes:** about 15 minutes on a CPU.
- **Output:** one line per checkpoint, e.g. `checkpoint checkpoint_40000 mean selection return 14.00`. Return is points scored; it should generally rise.
- **Produces:** `runs/my-frogger/`: checkpoints, `evaluations.parquet` (every evaluation game), the resolved config, version and fingerprint records.
- **If interrupted:** resume from the last checkpoint: `drl-sim train --config configs/experiments/my-frogger.yaml --resume runs/my-frogger/checkpoints/checkpoint_60000.zip`.
- Add `--dry-run` to any command to check it without writing anything.

### 2d. Pick weak, medium, and strong checkpoints

```text
drl-sim select-checkpoints --config configs/experiments/my-frogger.yaml
```

- **What it does:** applies the rules in `checkpoint_selection`: weak = first evaluated checkpoint after learning starts; strong = highest mean return; medium = the checkpoint between them closest to the midpoint score.
- **Produces:** `runs/my-frogger/selected-checkpoints.yaml` with each choice's step, score, and the evidence.
- **If it fails** with `Insufficient learning`: strong did not beat weak by `minimum_strong_improvement`. Train longer or lower that threshold; the framework will not compare checkpoints that are not meaningfully different.

### 2e. Measure the chosen checkpoints on fresh games

```text
drl-sim evaluate --config configs/experiments/my-frogger.yaml
```

- **What it does:** plays the `evaluation.final.seeds` games (never used for selection) with each of the three chosen checkpoints, so their scores are not inflated by having been selected.
- **Produces:** rows with `suite = final` in `runs/my-frogger/evaluations.parquet`; shown on the dashboard's Training page.

### 2f. Collect candidate clips

```text
drl-sim collect-scenarios --config configs/studies/my-study.yaml
```

- **What it does:** plays games with the strong checkpoint and cuts a fixed-length window (`clip_steps`, 16 here) around each event: a new life starting (`baseline`), a life being lost (`danger`), a point being scored (`goal`). See `scenarios.collection.candidate_windows`.
- **Takes:** about 10 seconds for the demo settings.
- **Output:** e.g. `wrote 72 candidate clip(s) and manifest template`. A warning appears if a label has fewer candidates than required.
- **Produces:**
  - `data/scenarios/my-study.yaml`: the manifest, one entry per clip, all `review_status: pending`.
  - `data/scenarios/my-study/observations.npz`: the frozen frames every checkpoint will see.
  - `data/scenarios/my-study/candidates/contact_sheets/*.png`: one image per clip for review.

### 2g. Review the clips (the one manual step)

The labels are only as good as this review, and it must be done before looking at any results.

1. Open the contact sheets. Each PNG shows the clip's frames left to right, top to bottom.
   - PowerShell: `ii data/scenarios/my-study/candidates/contact_sheets`
   - bash: `explorer.exe data/scenarios/my-study/candidates/contact_sheets` (Linux: `xdg-open`)
2. Open `data/scenarios/my-study.yaml`. For each clip, compare its sheet with its `scenario` label and edit two fields:

   ```yaml
   - clip_id: danger-0003
     scenario: danger
     ...
     review_status: approved   # was: pending. Use rejected for wrong or ambiguous clips
     curator: clay             # was: null. Your name; several reviewers: "clay, sam"
   ```

3. Keep the labels balanced: approve the same number per label (at least `minimum_clips_per_label`, 8 here), and reject the rest.
4. Check your work:

   ```text
   drl-sim validate-scenarios --config configs/studies/my-study.yaml
   ```

   It lists every problem at once, e.g.:

   ```text
   error: Scenario validation failed with 3 issue(s):
     - 72 clip(s) still pending review; set review_status to approved or rejected: ...
     - fewer than 8 usable clips for label(s): {'baseline': 0, 'danger': 0, 'goal': 0}
   ```

   Repeat until it prints `ok: scenarios valid`.

**Practice-only shortcut:** `python scripts/approve_demo_clips.py --config configs/studies/my-study.yaml`
approves the first non-overlapping clips per label without looking at them and
marks the curator as `demo-auto-approved`. Never use it for real results.

### 2h. Extract activations and compute metrics

```text
drl-sim run-study --config configs/studies/my-study.yaml
```

- **What it does:** verifies the run and the reviewed clips, feeds every clip frame through the weak, medium, and strong checkpoints, records the configured layers (`extraction.layers`), then computes every metric in `analysis.metrics` with bootstrap confidence intervals.
- **Takes:** about 1 minute for the demo.
- **Output:** one line per stage:

  ```text
  [skipped] experiment[0]: verified run ...runs\my-frogger
  [skipped] scenarios: valid ...data\scenarios\my-study.yaml
  [executed] extract: ...artifacts\my-study-v1\activations
  [executed] analyze: ...artifacts\my-study-v1\metrics
  ok: study ready. Next: drl-sim dashboard --config configs/studies/my-study.yaml
  ```

  `[blocked]` names the missing step and the command to run. Running it again skips finished stages; after editing the study YAML it recomputes only what the edit affects.
- **Produces:** `artifacts/my-study-v1/`. `extract` and `analyze` can also be run individually.

### 2i. Explore the results

```text
drl-sim dashboard --config configs/studies/my-study.yaml
```

Same as 2a, now showing your study. The yellow banner (one training seed) and
orange "valid with warnings" badge are expected for a single run.

### 2j. Moving to the real study

The real configs are `frogger-dqn.yaml` and `frogger-v1.yaml`; the steps are identical. Differences:

- **Training:** 10M steps. Use a GPU (`device: auto` picks it up) and expect hours; `--resume` makes interruptions cheap.
- **Clips:** 20 per label, 32 steps each, reviewed by hand.
- **Clip variety:** a deterministic agent in a deterministic game repeats itself, and identical clips are skipped (`duplicate_observations` in `candidates/candidates.json`). Add collection seeds if a label is short.
- **Changing settings:** edit the YAML (section 5), never pass settings on the command line. A changed experiment needs a new `id`; a changed study re-runs with `run-study`.

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

Checklist for the review walked through in 2g.

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
