from typing import Optional
from abc import abstractmethod

from torch import Tensor
from caustic import Parametrized


class PointSpreadFunction(Parametrized):
    def __init__(self, name: str = None):
        super().__init__(name)
    
    @abstractmethod
    def kernel(self, x: Tensor, y: Tensor, params: Optional["Packed"] = None) -> Tensor:
        ...
    
