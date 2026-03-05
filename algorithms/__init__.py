"""RL: Dual-Step PPO (Think->Act), GAE, RolloutBuffer; optional GRPO stub."""

from .ppo_dual import (
    DualStepPPOTrainer,
    make_experience,
    compute_advantages,
    RolloutBuffer,
    get_log_probs_for_response,
)
from .grpo import GRPO

__all__ = [
    "DualStepPPOTrainer",
    "make_experience",
    "compute_advantages",
    "RolloutBuffer",
    "get_log_probs_for_response",
    "GRPO",
]
