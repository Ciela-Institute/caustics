from abc import abstractmethod

import torch

from ..base import Base


class Source(Base):
    def __init__(self, device: torch.device):
        super().__init__(device)

    @abstractmethod
    def brightness(self, thx, thy, **kwargs):
        ...
