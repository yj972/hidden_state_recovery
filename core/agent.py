"""System2Agent: Thought then Action (text-only); reward from external RewardModel."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

# ---------------------------------------------------------------------------
# Types: Thought and Action are text (str) or token sequences; no fixed label space
# ---------------------------------------------------------------------------
Observation = TypeVar("Observation")
Thought = TypeVar("Thought")   # Typically str or sequence of tokens
Action = TypeVar("Action")      # Typically str (question/response text)


class System2Agent(ABC):
    """
    Abstract agent with separable Thought and Action policies.

    Purely generative: outputs text (Thought + Action). No classification head
    over a finite label space; no belief state b_t or entropy computed here.
    Reward is assigned externally by a RewardModel (e.g. LLM-as-Judge).
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    # ---------------------------------------------------------------------------
    # Thought policy: produce internal CoT (text)
    # ---------------------------------------------------------------------------

    @abstractmethod
    def think(
        self,
        observation: Observation,
        **kwargs: Any,
    ) -> Thought:
        """
        Produce internal thought (Chain-of-Thought) from observation.
        Returns text or token sequence; no belief vector.
        """
        ...

    @abstractmethod
    def get_thought_logits_or_sample(
        self,
        observation: Observation,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Thought, dict[str, Any] | None]:
        """
        For RL: return thought and optional aux (e.g. log_probs, entropy from LM).
        Thought is text/tokens; no classification logits.
        """
        ...

    # ---------------------------------------------------------------------------
    # Action policy: produce external action (text, e.g. question or answer)
    # ---------------------------------------------------------------------------

    @abstractmethod
    def act(
        self,
        observation: Observation,
        thought: Thought,
        **kwargs: Any,
    ) -> Action:
        """
        Given observation and thought, produce action (e.g. question or answer text).
        """
        ...

    @abstractmethod
    def get_action_logits_or_sample(
        self,
        observation: Observation,
        thought: Thought,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Action, dict[str, Any] | None]:
        """
        For RL: return action and optional aux (e.g. log_probs from LM).
        Action is text; no classification logits.
        """
        ...

    # ---------------------------------------------------------------------------
    # Unified step for rollout
    # ---------------------------------------------------------------------------

    def step(
        self,
        observation: Observation,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Thought, Action, dict[str, Any] | None]:
        """One agent step: think then act. Returns (thought, action, aux_dict)."""
        thought, thought_aux = self.get_thought_logits_or_sample(
            observation, deterministic=deterministic, **kwargs
        )
        action, action_aux = self.get_action_logits_or_sample(
            observation, thought, deterministic=deterministic, **kwargs
        )
        aux = {}
        if thought_aux:
            aux["thought"] = thought_aux
        if action_aux:
            aux["action"] = action_aux
        return thought, action, aux if aux else None
