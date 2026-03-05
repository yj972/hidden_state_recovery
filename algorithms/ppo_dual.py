"""
PPODualReward: PPO modified for dual rewards (r_PRM + R_task).

- Process reward r_PRM: intrinsic (e.g. entropy reduction).
- Task reward R_task: sparse outcome (correct y*).
Combined reward for advantage computation: r = w_PRM * r_PRM + w_task * R_task (or similar).
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar
import torch

# ---------------------------------------------------------------------------
# Type placeholders for agent and buffer
# ---------------------------------------------------------------------------
Agent = TypeVar("Agent")
Buffer = TypeVar("Buffer")


class PPODualReward(ABC):
    """
    Abstract PPO-style algorithm that consumes dual rewards (process + task)
    and updates Thought + Action policies.
    """

    def __init__(
        self,
        agent: Any,
        buffer: Any,
        weight_process: float = 1.0,
        weight_task: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.agent = agent
        self.buffer = buffer
        self.weight_process = weight_process
        self.weight_task = weight_task

    # ---------------------------------------------------------------------------
    # Reward combination
    # ---------------------------------------------------------------------------

    @abstractmethod
    def combine_rewards(
        self,
        reward_process: torch.Tensor,
        reward_task: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Combine r_PRM and R_task into scalar reward (e.g. weighted sum or shaped).
        """
        ...

    # ---------------------------------------------------------------------------
    # Advantage / GAE (with dual rewards)
    # ---------------------------------------------------------------------------

    @abstractmethod
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns (e.g. GAE). rewards already combined if desired.
        Returns (advantages, returns).
        """
        ...

    # ---------------------------------------------------------------------------
    # Update step
    # ---------------------------------------------------------------------------

    @abstractmethod
    def update(self, **kwargs: Any) -> dict[str, float]:
        """
        One PPO update: sample from buffer, compute loss (policy + value),
        optionally separate losses for thought vs action. Return metrics dict.
        """
        ...
