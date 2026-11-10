"""Structured logging helpers for pipeline stages."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


class StageFormatter(logging.Formatter):
    """Format records with optional stage and config hash fields."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "stage"):
            record.stage = "-"  # type: ignore[attr-defined]
        if not hasattr(record, "config_hash"):
            record.config_hash = "-"  # type: ignore[attr-defined]
        return super().format(record)


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure root package logging once and return the package logger."""
    global _CONFIGURED
    logger = logging.getLogger("drl_similarity")
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown log level: {level!r}")

    if _CONFIGURED:
        logger.setLevel(numeric)
        return logger

    logger.setLevel(numeric)
    logger.propagate = False
    formatter = StageFormatter(
        fmt="%(asctime)s %(levelname)s stage=%(stage)s config=%(config_hash)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the package root."""
    if name:
        return logging.getLogger(f"drl_similarity.{name}")
    return logging.getLogger("drl_similarity")


def log_extra(*, stage: str, config_hash: str | None = None) -> dict[str, str]:
    """Build LoggerAdapter-compatible extras for stage-aware records."""
    return {
        "stage": stage,
        "config_hash": (config_hash or "-")[:12],
    }
