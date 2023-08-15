from math import pi
from typing import Any, Optional, Union

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens
from ..parametrized import unpack

__all__ = ("PseudoJaffe",)


class PseudoJaffe(ThinLens):
    """
    Class representing a Pseudo Jaffe lens in strong gravitational lensing, 
    based on `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_ and 
    the `lenstronomy` source code.

    Attributes:
        name (str): The name of the Pseudo Jaffe lens.
        cosmology (Cosmology): The cosmology used for calculations.
        z_l (Optional[Union[Tensor, float]]): Redshift of the lens.
        x0 (Optional[Union[Tensor, float]]): x-coordinate of the center of the lens.
        y0 (Optional[Union[Tensor, float]]): y-coordinate of the center of the lens.
        convergence_0 (Optional[Union[Tensor, float]]): Central convergence of the lens.
        core_radius (Optional[Union[Tensor, float]]): Core radius of the lens.
        scale_radius (Optional[Union[Tensor, float]]): Scaling radius of the lens.
        s (float): Softening parameter to prevent numerical instabilities.
    """

    def __init__(
        self,
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        convergence_0: Optional[Union[Tensor, float]] = None,
        core_radius: Optional[Union[Tensor, float]] = None,
        scale_radius: Optional[Union[Tensor, float]] = None,
        s: float = 0.0,
        name: str = None,
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
            core_radius (Optional[Tensor]): Core radius of the lens.
            scale_radius (Optional[Tensor]): Scaling radius of the lens.
            s (float): Softening parameter to prevent numerical instabilities.
        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("convergence_0", convergence_0)
        self.add_param("core_radius", core_radius)
        self.add_param("scale_radius", scale_radius)
        self.s = s

    @unpack(2)
    def mass_enclosed_2d(self, theta, z_s, z_l, x0, y0, convergence_0, core_radius, scale_radius, *args, params: Optional["Packed"] = None, **kwargs):
        """
        Calculate the mass enclosed within a two-dimensional radius.

        Args:
            theta (Tensor): Radius at which to calculate enclosed mass.
            z_s (Tensor): Source redshift.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The mass enclosed within the given radius.
        """
        theta = theta + self.s
        surface_density_0 = convergence_0 * self.cosmology.critical_surface_density(z_l, z_s, params)
        return (
            2
            * pi
            * surface_density_0
            * core_radius
            * scale_radius
            / (scale_radius - core_radius)
            * (
                (core_radius**2 + theta**2).sqrt()
                - core_radius
                - (scale_radius**2 + theta**2).sqrt()
                + scale_radius
            )
        )

    @staticmethod
    def central_convergence(
        z_l,
        z_s,
        rho_0,
        core_radius,
        scale_radius,
        cosmology: Cosmology,
        *args,
        params: Optional["Packed"] = None, **kwargs,
    ):
        """
        Compute the central convergence.

        Args:
            z_l (Tensor): Lens redshift.
            z_s (Tensor): Source redshift.
            rho_0 (Tensor): Central mass density.
            core_radius (Tensor): Core radius of the lens.
            scale_radius (Tensor): Scaling radius of the lens.
            cosmology (Cosmology): The cosmology used for calculations.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The central convergence.
        """
        return (
            pi
            * rho_0
            * core_radius
            * scale_radius
            / (core_radius + scale_radius)
            / cosmology.critical_surface_density(z_l, z_s, params = params)
        )

    @unpack(3)
    def reduced_deflection_angle(
        self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, convergence_0, core_radius, scale_radius, *args, params: Optional["Packed"] = None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """ Calculate the deflection angle.

        Args:
            x (Tensor): x-coordinate of the lens.
            y (Tensor): y-coordinate of the lens.
            z_s (Tensor): Source redshift.
            params (Packed, optional): Dynamic parameter container.
    
        Returns:
            Tuple[Tensor, Tensor]: The deflection angle in the x and y directions.
        """
        x, y = translate_rotate(x, y, x0, y0)
        R = (x**2 + y**2).sqrt() + self.s
        f = R / core_radius / (1 + (1 + (R / core_radius) ** 2).sqrt()) - R / scale_radius / (
            1 + (1 + (R / scale_radius) ** 2).sqrt()
        )
        alpha = 2 * convergence_0 * core_radius * scale_radius / (scale_radius - core_radius) * f
        ax = alpha * x / R
        ay = alpha * y / R
        return ax, ay

    @unpack(3)
    def potential(
        self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, convergence_0, core_radius, scale_radius, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the lensing potential. This calculation is based on equation A18.
    
        Args:
            x (Tensor): x-coordinate of the lens.
            y (Tensor): y-coordinate of the lens.
            z_s (Tensor): Source redshift.
            params (Packed, optional): Dynamic parameter container.
    
        Returns:
            Tensor: The lensing potential.
        """
        x, y = translate_rotate(x, y, x0, y0)
        R_squared = x**2 + y**2 + self.s
        coeff = -2 * convergence_0 * core_radius * scale_radius / (scale_radius - core_radius)
        return coeff * (
            (scale_radius**2 + R_squared).sqrt()
            - (core_radius**2 + R_squared).sqrt()
            + core_radius * (core_radius + (core_radius**2 + R_squared).sqrt()).log()
            - scale_radius * (scale_radius + (scale_radius**2 + R_squared).sqrt()).log()
        )

    @unpack(3)
    def convergence(
        self, x: Tensor, y: Tensor, z_s: Tensor, z_l, x0, y0, convergence_0, core_radius, scale_radius, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Calculate the projected mass density, based on equation A6.
    
        Args:
            x (Tensor): x-coordinate of the lens.
            y (Tensor): y-coordinate of the lens.
            z_s (Tensor): Source redshift.
            params (Packed, optional): Dynamic parameter container.
    
        Returns:
            Tensor: The projected mass density.
        """
        x, y = translate_rotate(x, y, x0, y0)
        R_squared = x**2 + y**2 + self.s
        coeff = convergence_0 * core_radius * scale_radius / (scale_radius - core_radius)
        return coeff * (
            1 / (core_radius**2 + R_squared).sqrt() - 1 / (scale_radius**2 + R_squared).sqrt()
        )
