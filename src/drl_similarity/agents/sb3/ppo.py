"""Stable-Baselines3 PPO adapter (``sb3.ppo``)."""

from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3 import PPO

from drl_similarity.agents.sb3.base import Outputs, Sb3Adapter


class Sb3PpoAdapter(Sb3Adapter):
    """PPO with shared CNN layers plus distinct policy/value latents and heads."""

    adapter_id = "sb3.ppo"
    model_class = PPO
    action_spaces = (gym.spaces.Discrete, gym.spaces.Box)
    output_names = ("policy_latent", "value_latent", "action_logits", "value")
    allowed_kwargs = frozenset(
        {
            "batch_size",
            "clip_range",
            "clip_range_vf",
            "ent_coef",
            "gae_lambda",
            "gamma",
            "learning_rate",
            "max_grad_norm",
            "n_epochs",
            "n_steps",
            "normalize_advantage",
            "target_kl",
            "vf_coef",
        }
    )

    def _forward(self, model: PPO, observations: th.Tensor) -> Outputs:
        policy = model.policy
        policy_latent, value_latent = policy.mlp_extractor(policy.extract_features(observations))

        # action_logits holds distribution parameters (logits, or the mean for Box actions)
        return {
            "policy_latent": policy_latent,
            "value_latent": value_latent,
            "action_logits": policy.action_net(policy_latent),
            "value": policy.value_net(value_latent),
        }
