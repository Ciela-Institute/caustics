from abc import abstractmethod
from typing import Any, Optional

from torch import Tensor

from ..parametrized import Parametrized

__all__ = ("Source",)


class Source(Parametrized):
    @abstractmethod
    def brightness(
        self, thx: Tensor, thy: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        ...
