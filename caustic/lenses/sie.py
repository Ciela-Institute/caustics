from typing import Any, Optional

from torch import Tensor

from ..cosmology import Cosmology
from ..utils import derotate, translate_rotate
from .base import ThinLens

__all__ = ("SIE",)


class SIE(ThinLens):
    """
    A class representing a Singular Isothermal Ellipsoid (SIE) strong gravitational lens model. 
    This model is based on Keeton 2001, which can be found at https://arxiv.org/abs/astro-ph/0102341.
    
    Attributes:
        name (str): The name of the lens.
        cosmology (Cosmology): An instance of the Cosmology class.
        z_l (Tensor, optional): The redshift of the lens.
        x0 (Tensor, optional): The x-coordinate of the lens center.
        y0 (Tensor, optional): The y-coordinate of the lens center.
        q (Tensor, optional): The axis ratio of the lens.
        phi (Tensor, optional): The orientation angle of the lens (position angle).
        b (Tensor, optional): The Einstein radius of the lens.
        s (float): The core radius of the lens (defaults to 0.0).
    """

    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,# TODO change to true axis ratio
        phi: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        s: float = 0.0,
    ):
        """
        Initialize the SIE lens model.
        """
        super().__init__(name, cosmology, z_l)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("b", b)
        self.s = s

    def _get_potential(self, x, y, q):
        """
        Compute the radial coordinate in the lens plane.

        Args:
            x (Tensor): The x-coordinate in the lens plane.
            y (Tensor): The y-coordinate in the lens plane.
            q (Tensor): The axis ratio of the lens.

        Returns:
            Tensor: The radial coordinate in the lens plane.
        """
        return (q**2 * (x**2 + self.s**2) + y**2).sqrt()

    def reduced_deflection_angle(
        self,
            x: Tensor,
            y: Tensor,
            z_s: Tensor,
            params: Optional["Packed"] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the physical deflection angle.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tuple[Tensor, Tensor]: The deflection angle in the x and y directions.
        """
        z_l, x0, y0, q, phi, b = self.unpack(params)

        x, y = translate_rotate(x, y, x0, y0, phi)
        psi = self._get_potential(x, y, q)
        f = (1 - q**2).sqrt()
        ax = b * q.sqrt() / f * (f * x / (psi + self.s)).atan()
        ay = b * q.sqrt() / f * (f * y / (psi + q**2 * self.s)).atanh()

        return derotate(ax, ay, phi)

    def potential( 
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the lensing potential.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The lensing potential.
        """
        z_l, x0, y0, q, phi, b = self.unpack(params)

        ax, ay = self.reduced_deflection_angle(x, y, z_s, params)
        ax, ay = derotate(ax, ay, -phi)
        x, y = translate_rotate(x, y, x0, y0, phi)
        return x * ax + y * ay

    def convergence(
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Calculate the projected mass density.

        Args:
            x (Tensor): The x-coordinate of the lens.
            y (Tensor): The y-coordinate of the lens.
            z_s (Tensor): The source redshift.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The projected mass.
        """
        z_l, x0, y0, q, phi, b = self.unpack(params)

        x, y = translate_rotate(x, y, x0, y0, phi)
        psi = self._get_potential(x, y, q)
        return 0.5 * q.sqrt() * b / psi
