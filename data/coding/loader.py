"""
StackelbergCoderLoader: Data loader for ambiguous bug / root cause tasks — App 2.

Hidden state y* = true bug root cause; context = code, error message, repro steps.
"""

from ..base_dataset import BaseStackelbergDataset
from typing import Any, Iterator


class StackelbergCoderLoader(BaseStackelbergDataset):
    """
    Loader for Stackelberg Coder dataset. Concrete implementation to be added.
    - hidden_state: true root cause (e.g. bug id or location)
    - context: code snippet, error message, repro steps, etc.
    """

    def __init__(self, data_path: str | None = None, **kwargs: Any) -> None:
        super().__init__(data_path=data_path, **kwargs)

    def __len__(self) -> int:
        # TODO: implement (e.g. number of bug scenarios)
        return 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        # TODO: load one scenario; return {"hidden_state": ..., "context": ..., ...}
        raise NotImplementedError("StackelbergCoderLoader.__getitem__ not implemented")

    def sample(self, batch_size: int = 1, **kwargs: Any) -> Iterator[dict[str, Any]]:
        # TODO: sample batch_size scenarios
        raise NotImplementedError("StackelbergCoderLoader.sample not implemented")
