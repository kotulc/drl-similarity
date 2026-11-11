# DRL Representation Similarity

An experimental framework for studying how a deep reinforcement learning
agent's internal representations change as its task performance improves.

The first study uses a CNN DQN agent in Atari Frogger. Three checkpoints from
one training run—weak, medium, and strong—see the same curated clips from three
types of situation:

- **Baseline:** ordinary states near the beginning of play.
- **Danger:** states immediately preceding a likely loss of life.
- **Goal:** states near a successful crossing or other positive outcome.

The experiment asks whether a more capable checkpoint represents similar
situations more consistently, and whether it separates meaningfully different
situations more clearly.

The framework is entirely YAML-driven. Environment, algorithm, policy,
hyperparameters, evaluation, scenario collection, activation layers, metrics,
artifacts, and dashboard defaults are declared in configuration rather than
hard-coded into commands. The first study uses DQN, but the execution pipeline
selects an algorithm adapter from configuration so PPO, A2C, QR-DQN, recurrent
policies, and future implementations can be added without changing study code.

> This repository is in the design stage. The commands below describe the
> intended interface and will become runnable as the implementation milestones
> are completed.

## Research questions

1. Does within-scenario representation consistency increase with agent
   capability?
2. Does separation between baseline, danger, and goal representations increase
   with capability?
3. How does the overall representational geometry change during training?
4. Are any observed changes stable across clips, layers, evaluation episodes,
   and eventually independent training seeds?

The primary claim is deliberately narrower than “better agents always have more
similar activations.” Useful representations may be sparse, nonlinear, or
distributed, and raw latent coordinates may rotate during training. The study
therefore reports several complementary metrics rather than treating one
similarity score as ground truth.

## Hypotheses

- **H1 — Within-scenario consistency:** the strong checkpoint has higher
  pairwise cosine and correlation similarity, lower normalized Euclidean
  distance, and lower centroid dispersion than the medium and weak checkpoints
  for clips with the same scenario label.
- **H2 — Scenario separation:** the strong checkpoint has a larger
  within-scenario versus between-scenario similarity gap, a higher silhouette
  score, and better held-out nearest-centroid classification.
- **H3 — Temporal stability:** adjacent steps within a clip change more
  smoothly in the medium and strong checkpoints than in the weak checkpoint,
  after controlling for visual change between frames.
- **H4 — Geometry changes with learning:** CKA, RSA, SVCCA, and Procrustes
  comparisons reveal systematic changes from weak to medium to strong rather
  than only coordinate-level drift.

H1–H3 are evaluated separately for each checkpoint and layer. H4 compares
aligned stimulus sets across checkpoints. Results that do not support these
hypotheses are still informative; the dashboard must display uncertainty and
the full metric distributions rather than only favorable summaries.

## Study design

### Capability levels

The three models are checkpoints from the same initialized training run and
share an architecture. They are selected from held-out evaluation returns:

- **Weak:** a low-performing checkpoint after learning has begun.
- **Medium:** the checkpoint nearest the midpoint between weak and strong
  smoothed evaluation performance.
- **Strong:** the best held-out checkpoint observed in the run.

Selection by measured performance prevents training step from being used as a
proxy for capability. The final checkpoint is called **strong** or **best
observed**, not “state of the art,” unless it is evaluated under a published
Frogger protocol and actually meets the corresponding benchmark.

The initial study uses one training seed to keep the experiment inexpensive.
It is exploratory, not sufficient evidence for a general learning claim. The
artifact format includes `training_seed` so the same analysis can later be
repeated across at least three independent runs.

### Frozen scenario clips

Each scenario contains multiple short, ordered clips. A clip stores the exact
preprocessed observation sequence and enough provenance to reconstruct its
source episode. Every checkpoint receives the same observations in the same
order.

This frozen-stimulus design is important: if each model generated its own
states, representation differences would be confounded by different policies
visiting different parts of the game. Clips may be collected with a strong
policy, a scripted/manual policy, or both, but collection source is recorded
and balanced across scenarios where practical.

Scenario labels are curated and versioned in a manifest. They describe the
state visible in the clip, not the checkpoint's chosen action. For recurrent
models, clips also preserve sequence boundaries and burn-in observations so
hidden states can be initialized consistently.

### Latent layers

The first study extracts:

1. The output of each convolutional block.
2. The flattened final convolutional map.
3. NatureCNN's 512-dimensional feature vector, which is the final hidden
   representation before Q-values in the standard SB3 Atari DQN.
4. Q-values as a functional output, reported separately from latent features.

Spatial convolutional activations are summarized with a declared aggregation
method, initially global average pooling. Raw tensors remain available so a
later study can compare flattening and spatial-sample approaches.

## Metrics

No single metric answers every question. The framework groups metrics by the
claim they can support.

### Within a checkpoint

- **Cosine similarity:** directional agreement between latent vectors.
- **Correlation similarity:** agreement after removing each vector's mean.
- **Normalized Euclidean distance:** magnitude-sensitive distance after a
  declared feature normalization.
- **Centroid dispersion:** average distance from each clip or scenario
  centroid.
- **Temporal smoothness:** similarity at fixed lags within an ordered clip.
- **Within/between gap:** average same-scenario similarity minus average
  different-scenario similarity.
- **Silhouette score:** scenario compactness relative to scenario separation.
- **Nearest-centroid accuracy:** scenario prediction on held-out clips, never
  random individual frames from the same clip.

Frames from one clip are correlated. Confidence intervals therefore bootstrap
whole clips, not individual frames.

### Across checkpoints

Directly comparing latent dimension 17 in two neural networks can be
misleading: equivalent representations can differ by a rotation, permutation,
or scale. Cross-checkpoint analysis therefore emphasizes:

- **Linear CKA:** similarity of sample geometry, invariant to orthogonal
  transformations and isotropic scaling.
- **RSA:** correlation between representational dissimilarity matrices.
- **SVCCA:** similarity between dominant representational subspaces.
- **Orthogonal Procrustes:** residual distance after the best orthogonal
  alignment.

These metrics require the same aligned observations in both activation
matrices. Raw cosine similarity across checkpoints may be shown as a diagnostic
because checkpoints share initialization, but it is not a primary result.

### Visual embeddings

PCA and optional UMAP projections help inspect clusters and outliers. They are
visual aids, not statistical evidence. All dashboard plots retain links to the
checkpoint, layer, scenario, clip, and frame that generated each point.

## Planned workflow

The target command-line workflow accepts only a YAML file plus operational
flags such as `--resume` or `--force`. Scientific choices live in versioned
configuration:

```bash
# Install the project and Atari dependencies (Python 3.11).
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

# Train one agent, evaluate regularly, and save checkpoints.
drl-sim train --config configs/experiments/frogger-dqn.yaml

# Select weak/medium/strong checkpoints from held-out evaluation results.
drl-sim select-checkpoints --config configs/experiments/frogger-dqn.yaml

# Collect candidate clips, then curate labels in the generated manifest.
drl-sim collect-scenarios --config configs/studies/frogger-v1.yaml
drl-sim validate-scenarios --config configs/studies/frogger-v1.yaml

# Extract activations and compute analysis artifacts.
drl-sim extract --config configs/studies/frogger-v1.yaml
drl-sim analyze --config configs/studies/frogger-v1.yaml

# Explore performance, clips, representations, and uncertainty.
drl-sim dashboard --config configs/studies/frogger-v1.yaml
```

Each command writes immutable, configuration-addressed artifacts. Expensive
training and extraction never run inside Streamlit; the dashboard reads
completed Parquet and NPZ files.

The configuration is split into reusable YAML documents:

```text
configs/
  environments/atari-frogger.yaml
  algorithms/dqn-nature-cnn.yaml
  experiments/frogger-dqn.yaml
  studies/frogger-v1.yaml
  dashboards/default.yaml
```

An experiment config references an environment and algorithm config. A study
config references one or more completed experiment runs plus scenario,
activation, metric, and dashboard settings. References are resolved into one
self-contained `resolved-config.yaml` before execution. Unknown fields,
incompatible algorithm/environment combinations, and command-line overrides of
scientific settings are rejected.

## Dashboard

The dashboard is planned as six focused views:

1. **Study overview:** environment and model configuration, checkpoint
   selection, evaluation return curves, and data-quality warnings.
2. **Scenario explorer:** replay clips and inspect labels, clip sources, rewards,
   actions, terminal events, and frame metadata.
3. **Similarity matrix explorer:** a 3×3 case matrix and a drill-down 3×3
   frame-pair matrix for the selected cases. Every cell shows its paired frames,
   latent signatures, cosine, Pearson correlation, CKA, and normalized
   Euclidean distance, with distributions and confidence intervals available
   as secondary views.
4. **Scenario separation:** within/between gaps, silhouette scores,
   nearest-centroid results, PCA, and optional UMAP.
5. **Cross-checkpoint geometry:** layer-by-layer CKA, RSA, SVCCA, and
   Procrustes comparisons.
6. **Provenance and exports:** resolved YAML, checksums, package versions,
   artifact lineage, regeneration commands, and filtered CSV/chart downloads.

Filters apply to checkpoint, layer, scenario, clip source, and seed. Every
aggregate displays its sample count and confidence interval where applicable.
The implementation plan specifies the controls, charts, interactions, cache
rules, empty/error states, and data contract for every page.

The primary interaction is matrix-first rather than KPI-first: select a model
checkpoint, layer, metric, row-axis case, and column-axis case; then inspect the
nine comparisons formed by three frames from each selected case. Clicking a
case-matrix cell updates both axis selectors, while each selector can also be
changed independently and the axes can be swapped. The
implementation uses Streamlit for the Python-native application shell and
Plotly for interactive, publication-quality charts. See the
[dashboard implementation specification](docs/implementation-plan.md#10-dashboard-implementation)
for package choices and component-level details.

## Proposed stack

- Python 3.11 with a standard `venv` and `pip`; `requirements-lock.txt` pins
  exact versions for reproduction.
- **Gymnasium**, the maintained successor to Gym, and ALE via
  `gymnasium[atari]` for `ALE/Frogger-v5`. Legacy `gym` is not used.
- Stable-Baselines3 and PyTorch for DQN training and activation hooks.
- NumPy, Pandas, SciPy, and scikit-learn for numerical analysis.
- `rsatoolbox` for representational dissimilarity analysis.
- Streamlit for the dashboard shell, Plotly/Kaleido for interactive charts and
  exports, and Pillow/imageio for scenario media.
- Typer for a small command-line interface.
- Pydantic and PyYAML for strict, resolved configuration.
- PyArrow/Parquet for tabular results and NPZ for observations and activation
  tensors.

The environment configuration, including frame skip and sticky-action
probability, is pinned in each experiment. Training, checkpoint evaluation, and
scenario collection use explicit seeds.

## Planned artifacts

```text
configs/
  environments/atari-frogger.yaml
  algorithms/dqn-nature-cnn.yaml
  experiments/frogger-dqn.yaml
  studies/frogger-v1.yaml
  dashboards/default.yaml
data/
  scenarios/
    frogger-v1.yaml
    frogger-v1/
      observations.npz
runs/
  <run-id>/
    config.yaml
    evaluations.parquet
    checkpoints/
artifacts/
  <experiment-id>/
    manifest.json
    activations/
    metrics/
      frame_pairs.parquet
      summaries.parquet
      cross_checkpoint.parquet
    embeddings/
```

Large generated data, checkpoints, and artifacts are local by default and will
be excluded from version control. Small manifests and configurations are
versioned.

## Extending the framework

The execution and analysis pipelines depend on narrow adapters rather than a
particular SB3 policy or learning algorithm:

- `EnvironmentAdapter` creates and fingerprints an environment.
- `AlgorithmAdapter` validates algorithm-specific YAML, constructs training,
  saves/loads checkpoints, predicts actions, and exposes named latent layers.
- `LoadedAgent` is the inference-facing object returned by an algorithm
  adapter.
- `ActivationExtractor` handles feed-forward or recurrent sequence semantics.
- The scenario manifest supplies aligned observations and labels.
- The metric registry consumes ordinary sample-by-feature matrices.

A new algorithm is added by implementing one adapter and registering its YAML
identifier; no training, scenario, analysis, or dashboard branch should check
for `dqn`, `ppo`, or another algorithm name. The initial framework implements
SB3 DQN and PPO end to end and runs both through the same adapter contract
tests. SB3 A2C and sb3-contrib QR-DQN/RecurrentPPO are subsequent adapters.

A follow-up FF/CNN/RNN study can add or extend adapters while reusing scenario
datasets, metrics, artifact schemas, and dashboard views. Recurrent extraction
must use ordered clips, reset masks, and a declared burn-in period; treating RNN
frames as independent samples would invalidate the comparison.

Other environments can be added when they provide:

1. A deterministic, fingerprinted preprocessing configuration.
2. A versioned scenario manifest with enough clips per label.
3. An agent adapter with named extraction points.
4. A capability measure and held-out checkpoint evaluation protocol.

## Limitations and validity risks

- A single seed cannot distinguish a learning effect from run-specific noise.
- Curated labels can encode researcher judgment; the manifest must preserve
  provenance and support blind review.
- Adjacent frames are not independent samples.
- Apparent consistency may reflect lower visual diversity rather than learned
  abstraction. Analyses should report input-space similarity or visual-change
  controls alongside latent temporal smoothness.
- Checkpoint selection and hypothesis testing on the same episodes would leak
  information. Selection episodes, scenario clips, and final evaluation
  episodes must be disjoint.
- DQN return in one Frogger configuration is not directly comparable with
  scores produced under different frame skip, sticky-action, no-op, life-loss,
  or time-limit settings.
- Similarity does not establish that a representation causes better behavior.
  Linear probes or intervention/stitching studies would be follow-up work.

## Scope of the first implementation

The MVP intentionally includes one environment, one DQN architecture, one
training run, three capability checkpoints, three curated scenarios, a small
metric registry, reproducible artifacts, and a read-only dashboard.

It does not initially include distributed training, hyperparameter sweeps,
online experiment tracking services, automatic semantic scenario detection,
causal interventions, or claims of state-of-the-art performance.

See [the implementation plan](docs/implementation-plan.md) for milestones,
interfaces, schemas, tests, and acceptance criteria.
