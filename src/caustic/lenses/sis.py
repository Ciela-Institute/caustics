from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

__all__ = ("SIS",)


class SIS(ThinLens):
    """
    A class representing the Singular Isothermal Sphere (SIS) model. 
    This model inherits from the base `ThinLens` class.

    Attributes:
        name (str): The name of the SIS lens.
        cosmology (Cosmology): An instance of the Cosmology class.
        z_l (Optional[Tensor]): The lens redshift.
        thx0 (Optional[Tensor]): The x-coordinate of the lens center.
        thy0 (Optional[Tensor]): The y-coordinate of the lens center.
        th_ein (Optional[Tensor]): The Einstein radius of the lens.
        s (float): A smoothing factor, default is 0.0.
    """
    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        th_ein: Optional[Tensor] = None,
        s: float = 0.0,
    ):
        """
        Initialize the SIS lens model.
        """
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("th_ein", th_ein)
        self.s = s

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the deflection angle of the SIS lens.

        Args:
            thx (Tensor): The x-coordinate of the lens.
            thy (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tuple[Tensor, Tensor]: The deflection angle in the x and y directions.
        """
        z_l, thx0, thy0, th_ein = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        ax = th_ein * thx / th
        ay = th_ein * thy / th
        return ax, ay

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Compute the lensing potential of the SIS lens.

        Args:
            thx (Tensor): The x-coordinate of the lens.
            thy (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The lensing potential.
        """
        z_l, thx0, thy0, th_ein = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        return th_ein * th

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Calculate the projected mass density of the SIS lens.

        Args:
            thx (Tensor): The x-coordinate of the lens.
            thy (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The projected mass density.
        """
        z_l, thx0, thy0, th_ein = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        return 0.5 * th_ein / th
