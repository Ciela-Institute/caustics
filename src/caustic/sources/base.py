from abc import abstractmethod

import torch

from ..base import Base
from ..cosmology import AbstractCosmology


class AbstractSource(Base):
    def __init__(self, cosmology: AbstractCosmology, device: torch.device):
        super().__init__(cosmology, device)

    @abstractmethod
    def brightness(self, thx, thy, z=None):
        ...
