"""
ValueHead: Value function V(s) or V(s, b) for advantage estimation.

Consumes backbone hidden state (and optionally belief) and outputs scalar value.
"""

from abc import ABC, abstractmethod
from typing import Any
import torch


class ValueHead(ABC):
    """
    Abstract value function head. Input: hidden state from backbone (and optionally belief).
    Output: scalar value per batch item.
    """

    def __init__(self, input_dim: int, **kwargs: Any) -> None:
        self.input_dim = input_dim

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        belief: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute value V. hidden_states: (batch, hidden_size). Return (batch,) or (batch, 1).
        """
        ...
