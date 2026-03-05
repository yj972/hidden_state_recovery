"""
GRPO: Group Relative Policy Optimization (optional).

Uses group-relative advantages for multi-turn / multi-agent or preference-based training.
Interface aligned with dual-reward setup where applicable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import torch


class GRPO(ABC):
    """
    Abstract GRPO algorithm. Concrete implementation to compute group-relative
    advantages and update policy (thought + action) accordingly.
    """

    def __init__(self, agent: Any, **kwargs: Any) -> None:
        self.agent = agent

    @abstractmethod
    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages (e.g. normalize within group).
        """
        ...

    @abstractmethod
    def update(self, **kwargs: Any) -> dict[str, float]:
        """
        One GRPO update. Return metrics dict.
        """
        ...
