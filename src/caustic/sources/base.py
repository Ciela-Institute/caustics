from abc import abstractmethod

import torch

from ..base import Base


class Source(Base):
    @abstractmethod
    def brightness(self, thx, thy, **kwargs):
        ...
