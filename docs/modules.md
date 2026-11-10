# Module reference

One section per primary module of `src/drl_similarity/`. Usage and YAML options
are in the [user guide](user-guide.md).

## Data flow

```text
YAML ─ config.py ─► training/ ─► runs/<id>/            (checkpoints, evaluations, selection)
                    scenarios/ ─► data/scenarios/       (frozen clips + reviewed manifest)
                    activations/ ─► artifacts/<study>/activations/
                    analysis.py + metrics/ ─► artifacts/<study>/metrics/
                    dashboard/  (read-only view of all of the above)
```

Import rule: `config`, `study_io`, `tables`, `artifacts`, `metrics`, and
`dashboard` never import a training backend. SB3/PyTorch load only inside
`agents/`, `training/`, `scenarios/collect.py`, and `activations/extract.py`.

## `config.py` — strict YAML loading

| Item | Purpose |
|---|---|
| `load_config(path, expected_kind)` | read → parse → resolve `ref`s → cross-validate → canonicalize → hash |
| `ResolvedConfig` | `value` (model), `source_paths`, `config_hash`, `canonical` (mapping written beside artifacts) |
| `*Config` models | one Pydantic model per YAML block, all `extra="forbid"` and frozen |
| `validate_cross_document` | registered adapter/wrapper/rule/metric/pair-set IDs, seed disjointness, action-space compatibility, dashboard defaults |

- Hash = SHA-256 of the sorted canonical mapping with repo-relative paths; comments and key order do not affect it.
- No merging, env-var substitution, YAML tags, or import strings.

## `registry.py` — ID registries

`Registry[T]`: `register(id, item)` (duplicates raise), `get(id)` (unknown IDs list valid ones), `ids()`.

| Registry | Location | IDs |
|---|---|---|
| `WRAPPER_REGISTRY` | `environments/wrappers.py` | `record_episode_statistics`, `sb3_atari`, `frame_stack` |
| `ALGORITHM_ADAPTERS` | `agents/registry.py` | `sb3.dqn`, `sb3.ppo` |
| `SELECTION_RULES` | `training/selection.py` | three selection rules |
| `METRICS` | `metrics/registry.py` | twelve metrics |

## `environments/` — construction and fingerprints

| Function | Purpose |
|---|---|
| `make_environment(config, seed, render_mode)` | `gymnasium.make`, wrappers in YAML order, frame-skip and `expected` assertions, seeding |
| `apply_wrappers` | applies registered factories; rejects a second frame-skipping component |
| `compute_environment_fingerprints` | `observation` hash (what the model sees) and `runtime` hash (everything, incl. versions) |
| `inspect_environment` | reset + one step report used by `doctor` |

`sb3_atari` is SB3's `AtariWrapper` with the trailing channel axis removed, so
`frame_stack` yields `(4, 84, 84)` uint8.

## `agents/` — algorithm adapters

| Item | Purpose |
|---|---|
| `AlgorithmAdapter` (protocol) | `validate`, `build(..., resume=)`, `load`, `capabilities` |
| `LoadedAgent` (protocol) | `predict`, `layer_names`, `extract(observations, layers=, episode_starts=)` |
| `Sb3Adapter` | shared SB3 build/load/validate; subclasses declare `model_class`, `allowed_kwargs`, `action_spaces`, `output_names`, `_forward` |
| `Sb3LoadedAgent` | forward hooks (always removed) + functional outputs, returned in sample order |
| `nature_cnn_modules` | logical names → modules: post-ReLU `conv1-3`, `conv_flat`, `policy_features` |

- DQN hooks the online Q-network; `q_values` is the network output.
- PPO returns `policy_latent`, `value_latent`, `action_logits`, `value` from one forward pass.
- Recurrent support is interface-only (`episode_starts`, `recurrent`); extraction refuses recurrent adapters.

## `training/` — runner, evaluation, selection

| Module | Purpose |
|---|---|
| `runner.run_training` | validates, seeds, builds or resumes, writes provenance, trains, writes the run manifest |
| `callbacks.CheckpointEvaluationCallback` | algorithm-neutral: calls `on_checkpoint(step)` on the interval and at the end |
| `evaluate.evaluate_agent` | per seed: `episodes_per_seed` episodes; raw return (via `record_episode_statistics`), length, lives lost, positive-reward count |
| `evaluate.run_final_evaluation` | final suite on the three selected checkpoints |
| `selection.select_checkpoints` | applies the configured rules; writes `selected-checkpoints.yaml` with evidence |

- Each checkpoint is saved to a temporary name then renamed; evaluations are flushed after every checkpoint, so an interrupt leaves a resumable run.
- Selection uses only `suite == "selection"` rows.

## `scenarios/` — clips and curation

| Module | Purpose |
|---|---|
| `schema.py` | `ScenarioManifest`, `ClipRecord`, `assign_splits`, `observation_sha256`, `load_clip_arrays` |
| `collect.py` | rolls out the collector checkpoint, detects anchors, cuts windows (+ burn-in), writes observations, contact sheets, previews, and a `pending` manifest template |
| `validate.py` | every check in one pass: IDs, labels, review, lengths, counts/balance, overlap, split isolation, fingerprints, seeds, shapes, checksums |

- Stored preprocessed observations are the canonical stimuli for every checkpoint.
- Byte-identical candidate clips are skipped at collection.
- `positive_goal_event` is any reward above `min_reward`; in Frogger each forward hop scores, so set `min_reward` to isolate reaching home.

## `activations/` — extraction and storage

| Module | Purpose |
|---|---|
| `pooling.apply_pooling` | `none` (rank 2 only), `flatten`, `global_average`; ambiguous ranks are errors |
| `extract.extract_activations` | validates scenarios, loads each selected checkpoint, extracts in batches, pools, checks finiteness and that predictions are unchanged |
| `store.py` | `ActivationEntry`, `publish_activations`, `load_activation_bundle` (checksum-verified) |

Reuse: an existing artifact is returned only if it verifies and its extraction
config hash matches; otherwise a new version is required.

## `metrics/` — metric library

| Module | Purpose |
|---|---|
| `registry.py` | `MetricDefinition`: kind/direction, range, scopes, alignment, preprocessing, allowed kwargs |
| `vector.py` | cosine, correlation, normalized Euclidean, centroids, dispersion, temporal cosine, silhouette, nearest centroid |
| `geometry.py` | linear CKA, RSA, SVCCA, orthogonal Procrustes; all validate identical ordered sample IDs |
| `pairs.py` | within/between clip pairs (never same-clip), the three pair sets, representative frames |
| `bootstrap.py` | stratified whole-clip resampling, percentile intervals |

Invariances (tested): CKA, SVCCA, and Procrustes are invariant to orthogonal
rotation and isotropic scaling; RSA is rotation invariant only with a
rotation-invariant first-order distance (`euclidean`), not the default `correlation`.

## `analysis.py` — metric orchestration

- Fits standardization and the nearest-centroid classifier on train clips only.
- Produces `summaries`, `cross_checkpoint`, `frame_pairs`, `embeddings` tables with the column contracts in `tables.py`.
- Counts NaNs and flags estimates outside their bootstrap interval instead of dropping them.
- Dispatches metrics by ID (`PAIRWISE_METRICS`, `GEOMETRY_METRICS`).

## `artifacts/` and `study_io.py` — provenance

| Item | Purpose |
|---|---|
| `ArtifactManifest` | stage, config hash, git commit, package versions, input/output checksums, metadata |
| `ArtifactStore.publish` | write to a temporary sibling, manifest last, atomic rename; collisions error or create `-vN` |
| `verify_artifact_dir` | re-hashes every declared file (paths relative to the artifact directory) |
| `study_io.load_run` | reads a run directory into `RunInfo` (config, selection, fingerprints) |
| `study_io.*_stage_hash` | hash of only the config that determines a stage, so unrelated edits do not invalidate it |

## `pipeline.py` and `cli.py`

- `run_study_pipeline`: runs → scenarios → extract → analyze; each stage is `skipped`, `planned`, `executed`, or `blocked`. Never trains.
- `cli.py`: one thin Typer command per stage; catches only `DrlSimilarityError` (exit code 1); programming errors keep their traceback.
- `doctor.py`: package, registry, CUDA, and environment checks.

## `dashboard/` — read-only Streamlit app

| Module | Purpose |
|---|---|
| `app.py` | page registration, cached loading keyed by the metrics-manifest checksum, fatal error boundary |
| `data.py` | typed readers, schema checks, quality issues, filter helpers; returns `DashboardData` |
| `state.py` | pure session-state logic: matrix axes, frame selection, defaults, reset (no Streamlit import) |
| `charts.py` | pure Plotly figure functions with fixed category/color order and annotated empty states |
| `components.py` | case cell, frame-pair cell, metric bars, inspector helpers, clip player, downloads |
| `pages/` | six pages composing the above; no file reads |

Matrix page layout (matches the Canvas mockup): control bar → case-to-case
matrix with the selected-frame-pair inspector beside it → frame matrix → pair
details → secondary tabs.

## Tests

| Path | Covers |
|---|---|
| `tests/conftest.py` | fake pixel environment and tiny configs (no Atari) |
| `tests/contract/` | adapter contract, parameterized over every registered adapter |
| `tests/unit/` | config, environment, selection, metrics, artifacts/pooling/schema |
| `tests/integration/test_pipeline.py` | train → resume → select → evaluate → collect → run-study → dashboard data, for DQN and PPO |
| `tests/integration/test_dashboard_fixture.py` | dashboard readers, all six pages, matrix interactions on the synthetic fixture |
| `scripts/generate_fixture_study.py` | regenerates `tests/fixtures/study/` after a schema change |
