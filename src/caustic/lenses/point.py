from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

__all__ = ("Point",)


class Point(ThinLens):
    """
    Class representing a point mass lens in strong gravitational lensing.

    Attributes:
        name (str): The name of the point lens.
        cosmology (Cosmology): The cosmology used for calculations.
        z_l (Optional[Tensor]): Redshift of the lens.
        thx0 (Optional[Tensor]): x-coordinate of the center of the lens.
        thy0 (Optional[Tensor]): y-coordinate of the center of the lens.
        th_ein (Optional[Tensor]): Einstein radius of the lens.
        s (float): Softening parameter to prevent numerical instabilities.
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
        Initialize the Point class.

        Args:
            name (str): The name of the point lens.
            cosmology (Cosmology): The cosmology used for calculations.
            z_l (Optional[Tensor]): Redshift of the lens.
            thx0 (Optional[Tensor]): x-coordinate of the center of the lens.
            thy0 (Optional[Tensor]): y-coordinate of the center of the lens.
            th_ein (Optional[Tensor]): Einstein radius of the lens.
            s (float): Softening parameter to prevent numerical instabilities.
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
        Compute the deflection angles.

        Args:
            thx (Tensor): x-coordinates in the lens plane.
            thy (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            tuple[Tensor, Tensor]: The deflection angles in the x and y directions.
        """
        z_l, thx0, thy0, th_ein = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        ax = thx / th**2 * th_ein**2
        ay = thy / th**2 * th_ein**2
        return ax, ay

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Compute the lensing potential.

        Args:
            thx (Tensor): x-coordinates in the lens plane.
            thy (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The lensing potential.
        """
        z_l, thx0, thy0, th_ein = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        return th_ein**2 * th.log()

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Compute the convergence (dimensionless surface mass density).

        Args:
            thx (Tensor): x-coordinates in the lens plane.
            thy (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The convergence (dimensionless surface mass density).
        """
        z_l, thx0, thy0, th_ein = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        return torch.where((thx == 0) & (thy == 0), torch.inf, 0.0)
