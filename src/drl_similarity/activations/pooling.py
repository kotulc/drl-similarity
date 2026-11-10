"""Declared pooling of raw activation tensors into sample-by-feature matrices."""

from __future__ import annotations

import numpy as np

from drl_similarity.errors import ExtractionError


POOLING_METHODS: tuple[str, ...] = ("none", "flatten", "global_average")


def apply_pooling(array: np.ndarray, method: str, *, context: str = "") -> np.ndarray:
    """Reduce a batched activation tensor to shape ``(batch, features)``.

    ``none``
        Accepts only rank-2 ``(batch, features)`` input and returns it unchanged.
    ``flatten``
        Flattens every non-batch axis in C order, so a ``(batch, channels,
        height, width)`` map becomes channel-major, then row, then column.
    ``global_average``
        Averages the spatial axes of ``(batch, channels, *spatial)`` input and
        keeps one feature per channel.

    Ambiguous ranks are rejected instead of guessed. ``context`` (for example
    ``"checkpoint=strong layer=conv1"``) is included in error messages.
    """
    arr = np.asarray(array)
    where = f" [{context}]" if context else ""
    if method not in POOLING_METHODS:
        raise ExtractionError(
            f"Unknown pooling method {method!r}{where}; valid methods: {', '.join(POOLING_METHODS)}"
        )
    if arr.ndim < 2:
        raise ExtractionError(
            f"Activation must have a batch axis and at least one feature axis{where}",
            context={"shape": tuple(arr.shape), "pooling": method},
        )
    if method == "none":
        if arr.ndim != 2:
            raise ExtractionError(
                f"Pooling 'none' requires (batch, features) activations{where}; "
                "declare 'flatten' or 'global_average' for spatial layers",
                context={"shape": tuple(arr.shape)},
            )
        return arr
    if method == "flatten":
        return arr.reshape(arr.shape[0], -1)
    # global_average
    if arr.ndim < 3 or arr.ndim > 5:
        raise ExtractionError(
            f"Pooling 'global_average' requires (batch, channels, *spatial) activations{where}",
            context={"shape": tuple(arr.shape)},
        )
    return arr.mean(axis=tuple(range(2, arr.ndim)))
