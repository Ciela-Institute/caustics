from typing import Union, Optional

from torch import Tensor
from .base import PointSpreadFunction
from ..utils import translate_rotate


__all__ = ("GaussianPSF",)


class GaussianPSF(PointSpreadFunction):
    def __init__(
            self, 
            width: Optional[Union[Tensor, float]] = None,
            q: Optional[Union[Tensor, float]] = 1., 
            phi: Optional[Union[Tensor, float]] = 0., 
            x0: Optional[Union[Tensor, float]] = 0.,
            y0: Optional[Union[Tensor, float]] = 0.,
            I: Optional[Union[Tensor, float]] = 1.,
            name: str = None):
        super().__init__(name)
        self.add_param("width", width)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("I", I)
        
    def kernel(self, x: Tensor, y: Tensor, params: Optional["Packed"] = None):
        width, q, phi, x0, y0, I = self.unpack(params) 

        x, y = translate_rotate(x, y, x0, y0, phi)
        r_sq = x**2 * q**2 + y**2
        _kernel = (-0.5 * r_sq / width**2).exp()
        return I * _kernel/ _kernel.sum()
            
