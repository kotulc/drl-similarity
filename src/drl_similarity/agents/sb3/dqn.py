"""Stable-Baselines3 DQN adapter (``sb3.dqn``)."""

from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3 import DQN

from drl_similarity.agents.sb3.base import Outputs, Sb3Adapter, Sb3LoadedAgent, nature_cnn_modules


class Sb3DqnAdapter(Sb3Adapter):
    """DQN with logical layers conv1-3, conv_flat, policy_features, and q_values."""

    adapter_id = "sb3.dqn"
    model_class = DQN
    action_spaces = (gym.spaces.Discrete,)
    output_names = ("q_values",)
    allowed_kwargs = frozenset(
        {
            "batch_size",
            "buffer_size",
            "exploration_final_eps",
            "exploration_fraction",
            "exploration_initial_eps",
            "gamma",
            "gradient_steps",
            "learning_rate",
            "learning_starts",
            "max_grad_norm",
            "optimize_memory_usage",
            "target_update_interval",
            "tau",
            "train_freq",
        }
    )

    def _forward(self, model: DQN, observations: th.Tensor) -> Outputs:
        return {"q_values": model.policy.q_net(observations)}

    def _wrap(self, model: DQN) -> Sb3LoadedAgent:
        # Hook the online Q-network's extractor, not the target network's
        modules = nature_cnn_modules(model.policy.q_net.features_extractor)
        return Sb3LoadedAgent(model, modules, self._forward, self.output_names)
