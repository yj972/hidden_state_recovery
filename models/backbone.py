"""
TransformerBackbone: HuggingFace Transformer wrapper (e.g. Llama, Qwen).

Provides unified interface for encoding observations (and optionally thought)
for use by Thought policy, Action policy, and Critic.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar
import torch

# ---------------------------------------------------------------------------
# Type placeholders for input/output
# ---------------------------------------------------------------------------
InputIds = TypeVar("InputIds")
HiddenStates = TypeVar("HiddenStates")


class TransformerBackbone(ABC):
    """
    Abstract wrapper around a HuggingFace Transformer (Llama, Qwen, etc.).
    Used to encode sequences and produce hidden states for policy heads and critic.
    """

    def __init__(
        self,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> None:
        self.model_name_or_path = model_name_or_path

    # ---------------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Encode input tokens. Return last hidden state (or pooled) of shape
        (batch, seq_len, hidden_size) or (batch, hidden_size).
        """
        ...

    @abstractmethod
    def get_last_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Return last token hidden state (e.g. for causal LM) for policy/value heads.
        Shape: (batch, hidden_size).
        """
        ...

    # ---------------------------------------------------------------------------
    # Optional: tokenizer / config
    # ---------------------------------------------------------------------------

    def get_hidden_size(self) -> int:
        """Return model hidden size (for head dimension). Override if needed."""
        raise NotImplementedError("get_hidden_size not implemented")
