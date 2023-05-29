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
        thx0 (Optional[Tensor]): x-coordinate of the center of the lens.
        thy0 (Optional[Tensor]): y-coordinate of the center of the lens.
        kappa_0 (Optional[Tensor]): Central convergence of the lens.
        th_core (Optional[Tensor]): Core radius of the lens.
        th_s (Optional[Tensor]): Scaling radius of the lens.
        s (float): Softening parameter to prevent numerical instabilities.
    """

    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        kappa_0: Optional[Tensor] = None,
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
            thx0 (Optional[Tensor]): x-coordinate of the center of the lens.
            thy0 (Optional[Tensor]): y-coordinate of the center of the lens.
            kappa_0 (Optional[Tensor]): Central convergence of the lens.
            th_core (Optional[Tensor]): Core radius of the lens.
            th_s (Optional[Tensor]): Scaling radius of the lens.
            s (float): Softening parameter to prevent numerical instabilities.
        """
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("kappa_0", kappa_0)
        self.add_param("th_core", th_core)
        self.add_param("th_s", th_s)
        self.s = s

    def mass_enclosed_2d(self, th, z_s, x: Optional[dict[str, Any]] = None):
        """
        Calculate the mass enclosed within a two-dimensional radius.

        Args:
            th (Tensor): Radius at which to calculate enclosed mass.
            z_s (Tensor): Source redshift.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The mass enclosed within the given radius.
        """
        z_l, thx0, thy0, kappa_0, th_core, th_s = self.unpack(x)

        th = th + self.s
        Sigma_0 = kappa_0 * self.cosmology.Sigma_cr(z_l, z_s, x)
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
    def kappa_0(
        z_l,
        z_s,
        rho_0,
        th_core,
        th_s,
        cosmology: Cosmology,
        x: Optional[dict[str, Any]] = None,
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
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The central convergence.
        """
        return (
            pi
            * rho_0
            * th_core
            * th_s
            / (th_core + th_s)
            / cosmology.Sigma_cr(z_l, z_s, x)
        )

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """ Calculate the deflection angle.

        Args:
            thx (Tensor): x-coordinate of the lens.
            thy (Tensor): y-coordinate of the lens.
            z_s (Tensor): Source redshift.
            x (Optional[dict[str, Any]]): Additional parameters.
    
        Returns:
            Tuple[Tensor, Tensor]: The deflection angle in the x and y directions.
        """
        z_l, thx0, thy0, kappa_0, th_core, th_s = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        f = th / th_core / (1 + (1 + (th / th_core) ** 2).sqrt()) - th / th_s / (
            1 + (1 + (th / th_s) ** 2).sqrt()
        )
        alpha = 2 * kappa_0 * th_core * th_s / (th_s - th_core) * f
        ax = alpha * thx / th
        ay = alpha * thy / th
        return ax, ay

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Compute the lensing potential. This calculation is based on equation A18.
    
        Args:
            thx (Tensor): x-coordinate of the lens.
            thy (Tensor): y-coordinate of the lens.
            z_s (Tensor): Source redshift.
            x (Optional[dict[str, Any]]): Additional parameters.
    
        Returns:
            Tensor: The lensing potential.
        """
        z_l, thx0, thy0, kappa_0, th_core, th_s = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        coeff = -2 * kappa_0 * th_core * th_s / (th_s - th_core)
        return coeff * (
            (th_s**2 + th**2).sqrt()
            - (th_core**2 + th**2).sqrt()
            + th_core * (th_core + (th_core**2 + th**2).sqrt()).log()
            - th_s * (th_s + (th_s**2 + th**2).sqrt()).log()
        )

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Calculate the projected mass density, based on equation A6.
    
        Args:
            thx (Tensor): x-coordinate of the lens.
            thy (Tensor): y-coordinate of the lens.
            z_s (Tensor): Source redshift.
            x (Optional[dict[str, Any]]): Additional parameters.
    
        Returns:
            Tensor: The projected mass density.
        """
        z_l, thx0, thy0, kappa_0, th_core, th_s = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        th = (thx**2 + thy**2).sqrt() + self.s
        coeff = kappa_0 * th_core * th_s / (th_s - th_core)
        return coeff * (
            1 / (th_core**2 + th**2).sqrt() - 1 / (th_s**2 + th**2).sqrt()
        )
