"""Activation pooling, storage, and extraction.

``extract`` loads a training backend and is imported explicitly by the CLI.
"""

from drl_similarity.activations.pooling import POOLING_METHODS, apply_pooling


__all__ = ["POOLING_METHODS", "apply_pooling"]
