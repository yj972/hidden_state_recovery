"""
Model-Based Reward Shaping: Black-Box Process Reward via LLM-as-Judge.

We no longer compute explicit Shannon entropy (−∑ p log p). In open-ended domains
(coding, law, etc.) the label space Y is infinite, so a fixed belief vector b_t is
not feasible. Instead, r_PRM is produced by a Reward Model (Teacher LLM or heuristic)
that evaluates:
  - Information Gain: Did this question narrow down the hypothesis space? (implicit entropy reduction)
  - Efficiency: Was this question necessary?
  - Safety: Did it miss critical risks?

The "entropy reduction" is a latent concept judged by the model, not a computed scalar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# RewardModel interface: history, action, observation, optional outcome → float
# ---------------------------------------------------------------------------


class RewardModel(ABC):
    """
    Abstract reward model for process reward (r_PRM) and optional outcome shaping.
    Black-box: no classification head; reward is computed from dialogue/trajectory.
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def compute_reward(
        self,
        history: Any,
        action: Any,
        observation: Any,
        outcome: Any = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute scalar reward at this step (process reward r_t).

        - history: dialogue/trajectory so far (e.g. list of (speaker, text) or list of dicts).
        - action: current agent action (e.g. the question or response text).
        - observation: current observation (e.g. last user utterance or context).
        - outcome: optional; terminal outcome (e.g. correct y* or not) for shaping.

        Returns a float (e.g. in [0, 1] for quality, or negative for penalties).
        """
        ...


# ---------------------------------------------------------------------------
# LLM-as-Judge: score quality of the question / step
# ---------------------------------------------------------------------------

DEFAULT_LLM_JUDGE_PROMPT = """You are a reward model for a reasoning agent that asks questions to infer a hidden user intent (e.g. diagnosis, goal, bug cause).

Given the dialogue history and the agent's latest question/action, rate the QUALITY of this step from 0.0 to 1.0. Consider:
1. Information Gain: Does this question significantly narrow down the hypothesis space? (Implicit entropy reduction.)
2. Efficiency: Is this question necessary, or redundant?
3. Safety: Does it miss any critical risks or important dimensions?

Reply with ONLY a single float in [0.0, 1.0], e.g. 0.85"""


class LLMRewardModel(RewardModel):
    """
    Process reward via an LLM (API or local) that scores the quality of the agent's question/action.
    No explicit entropy; the LLM implicitly evaluates information gain, efficiency, and safety.
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        prompt: str | None = None,
        api_client: Any = None,
        max_score: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.prompt_template = prompt or DEFAULT_LLM_JUDGE_PROMPT
        self.api_client = api_client  # Optional: e.g. OpenAI client or HF pipeline
        self.max_score = max_score

    def _format_prompt(self, history: Any, action: Any, observation: Any) -> str:
        """Build the prompt string for the judge."""
        hist_str = str(history)[:2000] if history else "(none)"
        act_str = str(action)[:500] if action else "(none)"
        obs_str = str(observation)[:500] if observation else "(none)"
        return (
            self.prompt_template
            + "\n\n---\nDialogue history (recent):\n"
            + hist_str
            + "\n\nAgent's question/action:\n"
            + act_str
            + "\n\nCurrent observation/context:\n"
            + obs_str
            + "\n\nScore (single float 0.0-1.0):"
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM (override in subclass for API or local model).
        Default: return placeholder; implement with transformers or API.
        """
        if self.api_client is not None:
            # Assume api_client has a .generate(prompt) or .chat(...) interface
            if hasattr(self.api_client, "generate"):
                return str(self.api_client.generate(prompt))
            if hasattr(self.api_client, "chat"):
                return str(self.api_client.chat(prompt))
        # Placeholder: no model loaded
        return "0.5"

    def _parse_score(self, response: str) -> float:
        """Extract a float from the LLM response."""
        import re
        match = re.search(r"[-+]?\d*\.?\d+", response.strip())
        if match:
            v = float(match.group())
            return max(0.0, min(self.max_score, v))
        return 0.0

    def compute_reward(
        self,
        history: Any,
        action: Any,
        observation: Any,
        outcome: Any = None,
        **kwargs: Any,
    ) -> float:
        prompt = self._format_prompt(history, action, observation)
        response = self._call_llm(prompt)
        return self._parse_score(response)


# ---------------------------------------------------------------------------
# Heuristic reward (for debugging / ablations)
# ---------------------------------------------------------------------------


class HeuristicRewardModel(RewardModel):
    """
    Simple rule-based reward for debugging. No LLM.
    E.g. penalize length (-0.1 per turn), small bonus per turn, or constant.
    """

    def __init__(
        self,
        per_turn_penalty: float = -0.1,
        per_turn_bonus: float = 0.0,
        constant_reward: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.per_turn_penalty = per_turn_penalty
        self.per_turn_bonus = per_turn_bonus
        self.constant_reward = constant_reward

    def compute_reward(
        self,
        history: Any,
        action: Any,
        observation: Any,
        outcome: Any = None,
        **kwargs: Any,
    ) -> float:
        # Count turns: history can be list of turns or list of (speaker, text)
        try:
            n = len(history) if history else 0
        except Exception:
            n = 0
        r = self.constant_reward + n * (self.per_turn_penalty + self.per_turn_bonus)
        return float(r)
