"""Installation and environment health checks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from drl_similarity.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    ExperimentConfig,
    ResolvedConfig,
    StudyConfig,
)


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


def run_doctor(resolved: ResolvedConfig) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    checks.extend(_package_checks())
    checks.append(_cuda_check())
    checks.extend(_registry_checks())

    env_cfg = _environment_from_resolved(resolved)
    if env_cfg is not None:
        checks.extend(_environment_checks(env_cfg))
    return checks


def _package_checks() -> list[DoctorCheck]:
    required = [
        "numpy",
        "pandas",
        "yaml",
        "pydantic",
        "gymnasium",
        "ale_py",
        "torch",
        "stable_baselines3",
        "streamlit",
        "plotly",
        "sklearn",
        "scipy",
    ]
    out: list[DoctorCheck] = []
    for name in required:
        try:
            mod = import_module(name)
            version = getattr(mod, "__version__", "unknown")
            out.append(DoctorCheck(name=f"package:{name}", ok=True, detail=f"version={version}"))
        except Exception as exc:  # noqa: BLE001
            out.append(DoctorCheck(name=f"package:{name}", ok=False, detail=str(exc)))
    return out


def _cuda_check() -> DoctorCheck:
    try:
        import torch

        if torch.cuda.is_available():
            return DoctorCheck(
                name="cuda",
                ok=True,
                detail=f"available device_count={torch.cuda.device_count()}",
            )
        return DoctorCheck(name="cuda", ok=True, detail="not available (CPU ok)")
    except Exception as exc:  # noqa: BLE001
        return DoctorCheck(name="cuda", ok=False, detail=str(exc))


def _registry_checks() -> list[DoctorCheck]:
    out: list[DoctorCheck] = []
    try:
        from drl_similarity.agents.registry import ALGORITHM_ADAPTERS

        ids = ALGORITHM_ADAPTERS.ids()
        out.append(DoctorCheck(name="adapters", ok=bool(ids), detail=f"ids={ids}"))
    except Exception as exc:  # noqa: BLE001
        out.append(DoctorCheck(name="adapters", ok=False, detail=str(exc)))
    try:
        from drl_similarity.environments.wrappers import WRAPPER_REGISTRY

        ids = WRAPPER_REGISTRY.ids()
        out.append(DoctorCheck(name="wrappers", ok=bool(ids), detail=f"ids={ids}"))
    except Exception as exc:  # noqa: BLE001
        out.append(DoctorCheck(name="wrappers", ok=False, detail=str(exc)))
    try:
        from drl_similarity.metrics.registry import list_metric_ids

        ids = list_metric_ids()
        out.append(DoctorCheck(name="metrics", ok=bool(ids), detail=f"ids={ids}"))
    except Exception as exc:  # noqa: BLE001
        out.append(DoctorCheck(name="metrics", ok=False, detail=str(exc)))
    return out


def _environment_from_resolved(resolved: ResolvedConfig) -> EnvironmentConfig | None:
    value = resolved.value
    if isinstance(value, EnvironmentConfig):
        return value
    if isinstance(value, ExperimentConfig) and isinstance(value.environment, EnvironmentConfig):
        return value.environment
    if isinstance(value, StudyConfig):
        return None
    if isinstance(value, AlgorithmConfig):
        return None
    return None


def _environment_checks(env_cfg: EnvironmentConfig) -> list[DoctorCheck]:
    out: list[DoctorCheck] = []
    try:
        from drl_similarity.environments.atari import (
            assert_expected_environment,
            inspect_environment,
            make_environment,
        )

        env = make_environment(env_cfg, seed=0)
        try:
            assert_expected_environment(env, env_cfg)
            info = inspect_environment(env)
            out.append(
                DoctorCheck(
                    name="environment_shape",
                    ok=True,
                    detail=f"obs={info['observation_shape']} dtype={info['observation_dtype']}",
                )
            )
            out.append(
                DoctorCheck(
                    name="environment_step",
                    ok=True,
                    detail=f"action_n={info.get('action_n')} stepped_ok=True",
                )
            )
        finally:
            env.close()
    except Exception as exc:  # noqa: BLE001
        out.append(DoctorCheck(name="environment", ok=False, detail=str(exc)))
    return out

