from abc import ABC
from typing import Optional

import torch


class Base(ABC):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """
        Moves any tensor attributes of this object to `device` and changes the
        dtype of any floating-point tensor attributes to `dtype`.
        """
        ...
