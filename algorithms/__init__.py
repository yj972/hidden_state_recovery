"""
RL algorithms: PPO with dual rewards (Dual-Step Think->Act), GRPO (optional).
"""

from .ppo_dual import (
    PPODualReward,
    DualStepPPOTrainer,
    make_experience,
    compute_advantages,
    RolloutBuffer,
    get_log_probs_for_response,
)

from .grpo import GRPO

__all__ = [
    "PPODualReward",
    "DualStepPPOTrainer",
    "make_experience",
    "compute_advantages",
    "RolloutBuffer",
    "get_log_probs_for_response",
    "GRPO",
]
