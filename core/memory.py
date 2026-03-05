"""
TrajectoryBuffer: Storage for Thought-Action traces and dual rewards.

Stores (obs, thought, action, r_PRM, R_task, belief, ...) for training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar
import torch

# ---------------------------------------------------------------------------
# Type placeholders
# ---------------------------------------------------------------------------
Observation = TypeVar("Observation")
Thought = TypeVar("Thought")
Action = TypeVar("Action")
BeliefState = TypeVar("BeliefState")


class TrajectoryBuffer(ABC):
    """
    Abstract buffer for trajectories with Thought-Action structure and dual rewards.
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    # ---------------------------------------------------------------------------
    # Append / add
    # ---------------------------------------------------------------------------

    @abstractmethod
    def add(
        self,
        observation: Observation,
        thought: Any,
        action: Any,
        reward_process: float | torch.Tensor | None = None,
        reward_task: float | torch.Tensor | None = None,
        belief: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Add one step: (obs, thought, action, r_PRM, R_task, belief, ...).
        """
        ...

    @abstractmethod
    def finish_trajectory(
        self,
        last_observation: Any | None = None,
        last_value: torch.Tensor | None = None,
        last_done: bool = False,
        **kwargs: Any,
    ) -> None:
        """Mark end of trajectory (e.g. for GAE or episode boundary)."""
        ...

    # ---------------------------------------------------------------------------
    # Sampling for training
    # ---------------------------------------------------------------------------

    @abstractmethod
    def sample(
        self,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Sample minibatches for policy update. Returns dict with keys such as:
        observations, thoughts, actions, rewards_process, rewards_task, beliefs, ...
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear buffer after update (or for new epoch)."""
        ...

    # ---------------------------------------------------------------------------
    # Optional: size / stats
    # ---------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of steps (or trajectories) stored. Override for clarity."""
        return 0
