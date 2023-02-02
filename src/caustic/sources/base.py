from abc import abstractmethod

import torch

from ..base import Base


class AbstractSource(Base):
    @abstractmethod
    def brightness(self, thx, thy, **kwargs):
        ...
