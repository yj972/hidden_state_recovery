"""
DDXPlusLoader: Data loader for DDXPlus (Differential Diagnosis) — App 1.

Hidden state y* = true disease; context = symptoms / history.
"""

from ..base_dataset import BaseStackelbergDataset
from typing import Any, Iterator


class DDXPlusLoader(BaseStackelbergDataset):
    """
    Loader for DDXPlus dataset. Concrete implementation to be added.
    - hidden_state: true disease (e.g. diagnosis code or label)
    - context: symptoms, patient history, etc.
    """

    def __init__(self, data_path: str | None = None, **kwargs: Any) -> None:
        super().__init__(data_path=data_path, **kwargs)

    def __len__(self) -> int:
        # TODO: implement (e.g. number of cases)
        return 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        # TODO: load one case; return {"hidden_state": ..., "context": ..., ...}
        raise NotImplementedError("DDXPlusLoader.__getitem__ not implemented")

    def sample(self, batch_size: int = 1, **kwargs: Any) -> Iterator[dict[str, Any]]:
        # TODO: sample batch_size cases
        raise NotImplementedError("DDXPlusLoader.sample not implemented")
