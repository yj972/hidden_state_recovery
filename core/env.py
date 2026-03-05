"""StackelbergEnv: POMDP with hidden y*, belief b_t; Gymnasium reset/step."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
import gymnasium as gym

# ---------------------------------------------------------------------------
# Type placeholders for hidden state and belief (domain-specific)
# ---------------------------------------------------------------------------
HiddenState = TypeVar("HiddenState")   # y*
BeliefState = TypeVar("BeliefState")   # b_t
Observation = TypeVar("Observation")
Info = TypeVar("Info")


class StackelbergEnv(gym.Env, ABC, Generic[HiddenState, BeliefState, Observation, Info]):
    """
    Abstract POMDP environment for the Cognitive Stackelberg Game.

    The environment maintains:
    - A hidden state y* (ground truth) sampled at episode start.
    - A belief state b_t (distribution over y*) that the agent updates via observations.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # ---------------------------------------------------------------------------
    # Hidden state & belief (POMDP semantics)
    # ---------------------------------------------------------------------------

    @abstractmethod
    def get_hidden_state(self) -> HiddenState:
        """Return current hidden state y* (for training/simulation; not exposed to agent)."""
        ...

    @abstractmethod
    def get_belief_state(self) -> BeliefState:
        """Return current belief b_t (distribution over y*). May be derived from history."""
        ...

    @abstractmethod
    def sample_hidden_state(self) -> HiddenState:
        """Sample and set y* at episode start (e.g. from task distribution)."""
        ...

    # ---------------------------------------------------------------------------
    # Gymnasium interface
    # ---------------------------------------------------------------------------

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset env; sample new y*; return initial observation and info (e.g. initial belief)."""
        ...

    @abstractmethod
    def step(
        self,
        action: Any,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """
        Execute action. Returns obs, reward, terminated, truncated, info.
        Reward may be 0 here; dual rewards (r_PRM, R_task) can be computed in trainer.
        """
        ...

    # ---------------------------------------------------------------------------
    # Optional: belief update given observation (for belief-augmented obs)
    # ---------------------------------------------------------------------------

    def update_belief(self, observation: Observation, action: Any) -> BeliefState:
        """
        Update belief b_t from (obs, action). Override for explicit belief tracking.
        Default: return current belief unchanged.
        """
        return self.get_belief_state()
