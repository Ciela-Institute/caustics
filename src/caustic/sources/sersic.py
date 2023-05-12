from typing import Optional

import torch
from torch import Tensor

from ..utils import to_elliptical, translate_rotate
from .base import Source

__all__ = ("Sersic",)


class Sersic(Source):
    """
    `Sersic` is a subclass of the abstract class `Source`. It represents a source in a strong 
    gravitational lensing system that follows a Sersic profile, a mathematical function that describes 
    how the intensity I of a galaxy varies with distance r from its center.
    
    The Sersic profile is often used to describe elliptical galaxies and spiral galaxies' bulges.

    Attributes:
        thx0 (Optional[Tensor]): The x-coordinate of the Sersic source's center. 
        thy0 (Optional[Tensor]): The y-coordinate of the Sersic source's center.
        q (Optional[Tensor]): The axis ratio of the Sersic source.
        phi (Optional[Tensor]): The orientation of the Sersic source.
        index (Optional[Tensor]): The Sersic index, which describes the degree of concentration of the source.
        th_e (Optional[Tensor]): The scale length of the Sersic source.
        I_e (Optional[Tensor]): The intensity at the effective radius.
        s (float): A small constant for numerical stability.
        lenstronomy_k_mode (bool): A flag indicating whether to use lenstronomy to compute the value of k.
    """
    def __init__(
        self,
        name: str,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        index: Optional[Tensor] = None,
        th_e: Optional[Tensor] = None,
        I_e: Optional[Tensor] = None,
        s: float = 0.0,
        use_lenstronomy_k=False,
    ):
        """
        Constructs the `Sersic` object with the given parameters. 

        Args:
            name (str): The name of the source.
            thx0 (Optional[Tensor]): The x-coordinate of the Sersic source's center.
            thy0 (Optional[Tensor]): The y-coordinate of the Sersic source's center.
            q (Optional[Tensor]): The axis ratio of the Sersic source.
            phi (Optional[Tensor]): The orientation of the Sersic source.
            index (Optional[Tensor]): The Sersic index, which describes the degree of concentration of the source.
            th_e (Optional[Tensor]): The scale length of the Sersic source.
            I_e (Optional[Tensor]): The intensity at the effective radius.
            s (float): A small constant for numerical stability.
            use_lenstronomy_k (bool): A flag indicating whether to use lenstronomy to compute the value of k.
        """
        super().__init__(name)
        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("index", index)
        self.add_param("th_e", th_e)
        self.add_param("I_e", I_e)
        self.s = s

        self.lenstronomy_k_mode = use_lenstronomy_k

    def brightness(self, thx, thy, x):
        """
        Implements the `brightness` method for `Sersic`. The brightness at a given point is 
        determined by the Sersic profile formula.

        Args:
            thx (Tensor): The x-coordinate(s) at which to calculate the source brightness. 
                This could be a single value or a tensor of values.
            thy (Tensor): The y-coordinate(s) at which to calculate the source brightness. 
                This could be a single value or a tensor of values.
            x (dict[str, Any]): A dictionary containing additional parameters.

        Returns:
            Tensor: The brightness of the source at the given point(s). The output tensor has the same shape as `thx` and `thy`.

        Notes:
            The Sersic profile is defined as: I(r) = I_e * exp(-k * ((r / r_e)^(1/n) - 1)), 
            where I_e is the intensity at the effective radius r_e, n is the Sersic index 
            that describes the concentration of the source, and k is a parameter that 
            depends on n. In this implementation, we use elliptical coordinates ex and ey, 
            and the transformation from Cartesian coordinates is handled by `to_elliptical`.
            The value of k can be calculated in two ways, controlled by `lenstronomy_k_mode`. 
            If `lenstronomy_k_mode` is True, we use the approximation from Lenstronomy, 
            otherwise, we use the approximation from Ciotti & Bertin (1999).
        """
        thx0, thy0, q, phi, index, th_e, I_e = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        ex, ey = to_elliptical(thx, thy, q)
        e = (ex**2 + ey**2).sqrt() + self.s

        if self.lenstronomy_k_mode:
            k = 1.9992 * index - 0.3271
        else:
            k = 2 * index - 1 / 3 + 4 / 405 / index + 46 / 25515 / index**2

        exponent = -k * ((e / th_e) ** (1 / index) - 1)
        return I_e * exponent.exp()
