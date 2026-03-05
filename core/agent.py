"""
System2Agent: Thought + Action policies for the Cognitive Stackelberg Game.

The agent performs internal "Thought" (Chain-of-Thought) to reduce entropy about
hidden intent y* before committing to an external "Action".

Meta-actions: Ask (Information Seeking), Hypothesize (Reasoning), Answer (Terminal).
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar
import torch

# ---------------------------------------------------------------------------
# Type placeholders
# ---------------------------------------------------------------------------
Observation = TypeVar("Observation")
Thought = TypeVar("Thought")   # Internal CoT representation
Action = TypeVar("Action")     # Meta-action: Ask | Hypothesize | Answer
HiddenState = TypeVar("HiddenState")
BeliefState = TypeVar("BeliefState")


class System2Agent(ABC):
    """
    Abstract agent with separable Thought policy and Action policy.

    - Thought policy: given obs (and optionally belief), produce internal thought (CoT).
    - Action policy: given obs + thought (and optionally belief), produce meta-action.
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    # ---------------------------------------------------------------------------
    # Thought policy (entropy reduction / information seeking)
    # ---------------------------------------------------------------------------

    @abstractmethod
    def think(
        self,
        observation: Observation,
        belief: BeliefState | None = None,
        **kwargs: Any,
    ) -> Thought:
        """
        Produce internal thought (Chain-of-Thought) from observation (and optional belief).
        Used to reduce entropy about y* before acting.
        """
        ...

    @abstractmethod
    def get_thought_logits_or_sample(
        self,
        observation: Observation,
        belief: BeliefState | None = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Thought, dict[str, torch.Tensor] | None]:
        """
        For RL: return thought and optional dict (logits, log_probs, entropy) for loss.
        """
        ...

    # ---------------------------------------------------------------------------
    # Action policy (Ask / Hypothesize / Answer)
    # ---------------------------------------------------------------------------

    @abstractmethod
    def act(
        self,
        observation: Observation,
        thought: Thought,
        belief: BeliefState | None = None,
        **kwargs: Any,
    ) -> Action:
        """
        Given observation and thought, produce meta-action (Ask | Hypothesize | Answer).
        """
        ...

    @abstractmethod
    def get_action_logits_or_sample(
        self,
        observation: Observation,
        thought: Thought,
        belief: BeliefState | None = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Action, dict[str, torch.Tensor] | None]:
        """
        For RL: return action and optional dict (logits, log_probs, entropy) for loss.
        """
        ...

    # ---------------------------------------------------------------------------
    # Optional: unified step for rollout
    # ---------------------------------------------------------------------------

    def step(
        self,
        observation: Observation,
        belief: BeliefState | None = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Thought, Action, dict[str, torch.Tensor] | None]:
        """One agent step: think then act. Returns (thought, action, aux_dict)."""
        thought, thought_aux = self.get_thought_logits_or_sample(
            observation, belief, deterministic=deterministic, **kwargs
        )
        action, action_aux = self.get_action_logits_or_sample(
            observation, thought, belief, deterministic=deterministic, **kwargs
        )
        aux = {}
        if thought_aux:
            aux["thought"] = thought_aux
        if action_aux:
            aux["action"] = action_aux
        return thought, action, aux if aux else None
