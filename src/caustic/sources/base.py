from abc import abstractmethod

from ..base import Base

__all__ = ("Source",)

class Source(Base):
    @abstractmethod
    def brightness(self, thx, thy, **kwargs):
        ...
