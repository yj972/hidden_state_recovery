"""
BaseStackelbergDataset: Abstract dataset loader for Cognitive Stackelberg tasks.

Each sample provides (or can be used to construct):
- Hidden state y* (ground truth)
- Context/observation that the agent sees
- Optional belief prior, action space, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, TypeVar

# ---------------------------------------------------------------------------
# Type placeholders for hidden state and sample structure
# ---------------------------------------------------------------------------
HiddenState = TypeVar("HiddenState")
Sample = TypeVar("Sample")


class BaseStackelbergDataset(ABC):
    """
    Abstract dataset that yields tasks for the Stackelberg env.
    Each item can be used to instantiate or reset an episode (sample y*, provide context).
    """

    def __init__(self, data_path: str | None = None, **kwargs: Any) -> None:
        self.data_path = data_path

    # ---------------------------------------------------------------------------
    # Iteration / sampling
    # ---------------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples (or episodes) in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Return one sample as dict. Expected keys (at least):
        - hidden_state or y_star: ground truth
        - context or observation: initial context for the agent
        Optional: prior_belief, candidate_set, metadata, etc.
        """
        ...

    @abstractmethod
    def sample(self, batch_size: int = 1, **kwargs: Any) -> Iterator[dict[str, Any]]:
        """
        Sample batch_size items (with or without replacement). Yield or return list.
        """
        ...

    # ---------------------------------------------------------------------------
    # Optional: reset env from dataset sample
    # ---------------------------------------------------------------------------

    def get_hidden_state_from_sample(self, sample: dict[str, Any]) -> Any:
        """Extract y* from sample dict. Override if key name differs."""
        return sample.get("hidden_state") or sample.get("y_star")

    def get_initial_context_from_sample(self, sample: dict[str, Any]) -> Any:
        """Extract initial observation/context from sample. Override if needed."""
        return sample.get("context") or sample.get("observation")
