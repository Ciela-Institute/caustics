from typing import Union, Optional

import torch
from torch.special import bessel_j1
from torch import Tensor
from .base import PointSpreadFunction
from ..utils import safe_divide, translate_rotate

__all__ = ("Airy",)


class Airy(PointSpreadFunction):
    # Note: we only work in the far field approximation, i.e. theta << 1
    def __init__(
            self, 
            scale: Optional[Union[Tensor, float]] = None,
            x0: Optional[Union[Tensor, float]] = 0.,
            y0: Optional[Union[Tensor, float]] = 0.,
            I: Optional[Union[Tensor, float]] = 1.,
            name: str = None
            ):
        super().__init__(name)
        self.add_param("scale", scale)
        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("I", I)
    
    def kernel(self, x: Tensor, y: Tensor, params: Optional["Packed"] = None) -> Tensor:
        scale, x0, y0, I = self.unpack(params)
        x, y = translate_rotate(x, y, x0, y0)
        r = scale * torch.maximum(torch.hypot(x, y), torch.ones_like(x) * 1e-8) # manually avoid division by 0
        # Since we cannot use safe_divide in an operation that can be vmapped. This is annoying, otherwise func is trivial to batch
        # _kernel = (2*safe_divide(bessel_j1(r), r))**2
        _kernel = (2*bessel_j1(r) / r)**2
        return I * _kernel.view(x.shape) / _kernel.sum()

