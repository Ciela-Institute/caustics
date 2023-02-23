from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict

from torch import Tensor

from ..parametrized import Parametrized

__all__ = ("Source",)


class Source(Parametrized):
    @abstractmethod
    def brightness(
        self, thx: Tensor, thy: Tensor, x: Dict[str, Any] = defaultdict(list)
    ):
        ...
