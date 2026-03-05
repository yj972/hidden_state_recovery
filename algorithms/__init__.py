"""
RL algorithms: PPO with dual rewards, GRPO (optional).
"""

from .ppo_dual import PPODualReward
from .grpo import GRPO

__all__ = ["PPODualReward", "GRPO"]
