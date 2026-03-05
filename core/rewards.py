"""
IntrinsicRewardModule: Process reward based on Information Gain / Entropy Reduction.

Dual-reward system:
- r_PRM (Process Reward): Intrinsic, ΔH(b_t) = H(b_{t-1}) - H(b_t)
- R_task (Outcome Reward): Sparse, for correctly identifying y*
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar
import torch

# ---------------------------------------------------------------------------
# Type placeholders for belief and optional context
# ---------------------------------------------------------------------------
BeliefState = TypeVar("BeliefState")
HiddenState = TypeVar("HiddenState")


class IntrinsicRewardModule(ABC):
    """
    Abstract module for computing process reward (e.g. entropy reduction / information gain).
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    # ---------------------------------------------------------------------------
    # Entropy over belief
    # ---------------------------------------------------------------------------

    @abstractmethod
    def entropy(self, belief: BeliefState) -> torch.Tensor:
        """
        Compute H(b) for belief distribution b.
        belief: batched if needed; return shape (batch,) or scalar.
        """
        ...

    @abstractmethod
    def entropy_reduction(
        self,
        belief_before: BeliefState,
        belief_after: BeliefState,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        ΔH = H(before) - H(after). Use as r_PRM (process reward).
        Return shape (batch,) or scalar.
        """
        ...

    # ---------------------------------------------------------------------------
    # Optional: information gain relative to hidden state (for analysis)
    # ---------------------------------------------------------------------------

    def information_gain(
        self,
        belief_before: BeliefState,
        belief_after: BeliefState,
        hidden: HiddenState | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Optional: information gain (e.g. KL or mutual information). Default: use entropy_reduction.
        """
        return self.entropy_reduction(belief_before, belief_after, **kwargs)
