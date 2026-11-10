"""Configuration loading and hashing tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from drl_similarity.config import hash_config, load_config
from drl_similarity.errors import ConfigError


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


def test_load_environment() -> None:
    resolved = load_config(CONFIGS / "environments" / "atari-frogger.yaml", expected_kind="environment")
    assert resolved.value.kind == "environment"
    assert resolved.value.id == "atari-frogger-v1"
    assert len(resolved.config_hash) == 64


def test_load_algorithm() -> None:
    resolved = load_config(CONFIGS / "algorithms" / "dqn-nature-cnn.yaml", expected_kind="algorithm")
    assert resolved.value.adapter == "sb3.dqn"  # type: ignore[attr-defined]


def test_load_experiment_resolves_refs() -> None:
    resolved = load_config(CONFIGS / "experiments" / "frogger-dqn.yaml", expected_kind="experiment")
    value = resolved.value
    assert value.kind == "experiment"
    assert value.environment.kind == "environment"  # type: ignore[union-attr]
    assert value.algorithm.kind == "algorithm"  # type: ignore[union-attr]
    assert value.environment.id == "atari-frogger-v1"  # type: ignore[union-attr]


def test_load_dashboard() -> None:
    resolved = load_config(CONFIGS / "dashboards" / "default.yaml", expected_kind="dashboard")
    assert resolved.value.title.startswith("Frogger")  # type: ignore[attr-defined]


def test_unknown_field_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "dashboard",
                "id": "x",
                "title": "t",
                "not_a_field": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(path, expected_kind="dashboard")


def test_wrong_kind_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(CONFIGS / "environments" / "atari-frogger.yaml", expected_kind="algorithm")


def test_stable_hash_key_order(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    payload = {
        "schema_version": 1,
        "kind": "dashboard",
        "id": "hash-test",
        "title": "t",
        "default_filters": {"checkpoint": "strong", "layer": "policy_features"},
    }
    a.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    # Different key insertion order in nested mapping
    payload2 = {
        "title": "t",
        "kind": "dashboard",
        "id": "hash-test",
        "schema_version": 1,
        "default_filters": {"layer": "policy_features", "checkpoint": "strong"},
    }
    b.write_text(yaml.safe_dump(payload2, sort_keys=False), encoding="utf-8")
    ha = load_config(a).config_hash
    hb = load_config(b).config_hash
    assert ha == hb


def test_hash_changes_with_scientific_setting(tmp_path: Path) -> None:
    base = {
        "schema_version": 1,
        "kind": "algorithm",
        "id": "x",
        "adapter": "sb3.dqn",
        "policy": "CnnPolicy",
        "policy_kwargs": {},
        "model_kwargs": {"learning_rate": 0.0001},
        "capabilities": {
            "action_spaces": ["discrete"],
            "recurrent": False,
            "activation_layers": ["q_values"],
        },
    }
    p1 = tmp_path / "a.yaml"
    p2 = tmp_path / "b.yaml"
    p1.write_text(yaml.safe_dump(base), encoding="utf-8")
    base["model_kwargs"] = {"learning_rate": 0.0002}
    p2.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert load_config(p1).config_hash != load_config(p2).config_hash


def test_missing_reference(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "experiment",
                "id": "x",
                "environment": {"ref": "missing-env.yaml"},
                "algorithm": {"ref": "missing-algo.yaml"},
                "seed": 1,
                "training": {"total_timesteps": 10},
                "evaluation": {
                    "selection": {"seeds": [1]},
                    "final": {"seeds": [2]},
                },
                "checkpoint_selection": {
                    "weak": {"rule": "earliest_complete_after_learning_starts"},
                    "medium": {"rule": "nearest_performance_midpoint"},
                    "strong": {"rule": "highest_mean_selection_return"},
                },
                "artifacts": {"root": str(tmp_path / "runs")},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(path)


def test_selection_final_seed_overlap_rejected(tmp_path: Path) -> None:
    env = tmp_path / "env.yaml"
    algo = tmp_path / "algo.yaml"
    env.write_text(
        (CONFIGS / "environments" / "atari-frogger.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    algo.write_text(
        (CONFIGS / "algorithms" / "dqn-nature-cnn.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "experiment",
                "id": "overlap",
                "environment": {"ref": "env.yaml"},
                "algorithm": {"ref": "algo.yaml"},
                "seed": 1,
                "training": {"total_timesteps": 10},
                "evaluation": {
                    "selection": {"seeds": [1, 2]},
                    "final": {"seeds": [2, 3]},
                },
                "checkpoint_selection": {
                    "weak": {"rule": "earliest_complete_after_learning_starts"},
                    "medium": {"rule": "nearest_performance_midpoint"},
                    "strong": {"rule": "highest_mean_selection_return"},
                },
                "artifacts": {"root": str(tmp_path / "runs")},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="disjoint"):
        load_config(exp)


def test_canonicalize_hash_helper() -> None:
    assert len(hash_config({"a": 1, "b": [2, 3]})) == 64


def test_python_tags_rejected(tmp_path: Path) -> None:
    """YAML is parsed with safe_load: executable tags never construct objects."""
    path = tmp_path / "evil.yaml"
    path.write_text("schema_version: 1\nkind: dashboard\nid: !!python/object/apply:os.getcwd []\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


@pytest.mark.parametrize(
    ("source", "old", "new", "message"),
    [
        ("algorithms/dqn-nature-cnn.yaml", "adapter: sb3.dqn", "adapter: sb3.nope", "algorithm adapter"),
        ("environments/atari-frogger.yaml", "id: frame_stack", "id: nope", "wrapper"),
        ("dashboards/default.yaml", "schema_version: 1", "schema_version: 2", "schema_version"),
    ],
)
def test_unknown_ids_and_versions_rejected(tmp_path: Path, source: str, old: str, new: str, message: str) -> None:
    text = (CONFIGS / source).read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "config.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_load_config_does_not_mutate_source() -> None:
    path = CONFIGS / "studies" / "frogger-v1.yaml"
    before = path.read_bytes()

    load_config(path, "study", allow_missing_scenario_manifest=True)

    assert path.read_bytes() == before
