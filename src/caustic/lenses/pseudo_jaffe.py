from math import pi
from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

__all__ = ("PseudoJaffe",)


class PseudoJaffe(ThinLens):
    """
    Class representing a Pseudo Jaffe lens in strong gravitational lensing, 
    based on `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_ and 
    the `lenstronomy` source code.

    Attributes:
        name (str): The name of the Pseudo Jaffe lens.
        cosmology (Cosmology): The cosmology used for calculations.
        z_l (Optional[Tensor]): Redshift of the lens.
        x0 (Optional[Tensor]): x-coordinate of the center of the lens.
        y0 (Optional[Tensor]): y-coordinate of the center of the lens.
        convergence_0 (Optional[Tensor]): Central convergence of the lens.
        th_core (Optional[Tensor]): Core radius of the lens.
        th_s (Optional[Tensor]): Scaling radius of the lens.
        s (float): Softening parameter to prevent numerical instabilities.
    """

    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        convergence_0: Optional[Tensor] = None,
        th_core: Optional[Tensor] = None,
        th_s: Optional[Tensor] = None,
        s: float = 0.0,
    ):
        """
        Initialize the PseudoJaffe class.

        Args:
            name (str): The name of the Pseudo Jaffe lens.
            cosmology (Cosmology): The cosmology used for calculations.
            z_l (Optional[Tensor]): Redshift of the lens.
            x0 (Optional[Tensor]): x-coordinate of the center of the lens.
            y0 (Optional[Tensor]): y-coordinate of the center of the lens.
            convergence_0 (Optional[Tensor]): Central convergence of the lens.
            th_core (Optional[Tensor]): Core radius of the lens.
            th_s (Optional[Tensor]): Scaling radius of the lens.
            s (float): Softening parameter to prevent numerical instabilities.
        """
        super().__init__(name, cosmology, z_l)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("convergence_0", convergence_0)
        self.add_param("th_core", th_core)
        self.add_param("th_s", th_s)
        self.s = s

    def mass_enclosed_2d(self, th, z_s, P: "Packed" = None):
        """
        Calculate the mass enclosed within a two-dimensional radius.

        Args:
            th (Tensor): Radius at which to calculate enclosed mass.
            z_s (Tensor): Source redshift.
            P (Packed): Additional parameters.

        Returns:
            Tensor: The mass enclosed within the given radius.
        """
        z_l, x0, y0, convergence_0, th_core, th_s = self.unpack(P)

        th = th + self.s
        Sigma_0 = convergence_0 * self.cosmology.critical_surface_density(z_l, z_s, P)
        return (
            2
            * pi
            * Sigma_0
            * th_core
            * th_s
            / (th_s - th_core)
            * (
                (th_core**2 + th**2).sqrt()
                - th_core
                - (th_s**2 + th**2).sqrt()
                + th_s
            )
        )

    @staticmethod
    def convergence_0(
        z_l,
        z_s,
        rho_0,
        th_core,
        th_s,
        cosmology: Cosmology,
        P: "Packed" = None,
    ):
        """
        Compute the central convergence.

        Args:
            z_l (Tensor): Lens redshift.
            z_s (Tensor): Source redshift.
            rho_0 (Tensor): Central mass density.
            th_core (Tensor): Core radius of the lens.
            th_s (Tensor): Scaling radius of the lens.
            cosmology (Cosmology): The cosmology used for calculations.
            P (Packed): Additional parameters.

        Returns:
            Tensor: The central convergence.
        """
        return (
            pi
            * rho_0
            * th_core
            * th_s
            / (th_core + th_s)
            / cosmology.critical_surface_density(z_l, z_s, P)
        )

    def deflection_angle(
        self, x: Tensor, y: Tensor, z_s: Tensor, P: "Packed" = None
    ) -> tuple[Tensor, Tensor]:
        """ Calculate the deflection angle.

        Args:
            x (Tensor): x-coordinate of the lens.
            y (Tensor): y-coordinate of the lens.
            z_s (Tensor): Source redshift.
            P (Packed): Additional parameters.
    
        Returns:
            Tuple[Tensor, Tensor]: The deflection angle in the x and y directions.
        """
        z_l, x0, y0, convergence_0, th_core, th_s = self.unpack(P)

        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        f = th / th_core / (1 + (1 + (th / th_core) ** 2).sqrt()) - th / th_s / (
            1 + (1 + (th / th_s) ** 2).sqrt()
        )
        alpha = 2 * convergence_0 * th_core * th_s / (th_s - th_core) * f
        ax = alpha * x / th
        ay = alpha * y / th
        return ax, ay

    def potential(
        self, x: Tensor, y: Tensor, z_s: Tensor, P: "Packed" = None
    ) -> Tensor:
        """
        Compute the lensing potential. This calculation is based on equation A18.
    
        Args:
            x (Tensor): x-coordinate of the lens.
            y (Tensor): y-coordinate of the lens.
            z_s (Tensor): Source redshift.
            P (Packed): Additional parameters.
    
        Returns:
            Tensor: The lensing potential.
        """
        z_l, x0, y0, convergence_0, th_core, th_s = self.unpack(P)

        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        coeff = -2 * convergence_0 * th_core * th_s / (th_s - th_core)
        return coeff * (
            (th_s**2 + th**2).sqrt()
            - (th_core**2 + th**2).sqrt()
            + th_core * (th_core + (th_core**2 + th**2).sqrt()).log()
            - th_s * (th_s + (th_s**2 + th**2).sqrt()).log()
        )

    def convergence(
        self, x: Tensor, y: Tensor, z_s: Tensor, P: "Packed" = None
    ) -> Tensor:
        """
        Calculate the projected mass density, based on equation A6.
    
        Args:
            x (Tensor): x-coordinate of the lens.
            y (Tensor): y-coordinate of the lens.
            z_s (Tensor): Source redshift.
            P (Packed): Additional parameters.
    
        Returns:
            Tensor: The projected mass density.
        """
        z_l, x0, y0, convergence_0, th_core, th_s = self.unpack(P)

        x, y = translate_rotate(x, y, x0, y0)
        th = (x**2 + y**2).sqrt() + self.s
        coeff = convergence_0 * th_core * th_s / (th_s - th_core)
        return coeff * (
            1 / (th_core**2 + th**2).sqrt() - 1 / (th_s**2 + th**2).sqrt()
        )
