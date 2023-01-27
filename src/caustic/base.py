from abc import ABC

import torch


class Base(ABC):
    def __init__(self, device: torch.device):
        self.device = device

    def to(self, device: torch.device):
        """
        Moves any tensor attributes of this object to the specified device.
        """
        raise NotImplementedError()
