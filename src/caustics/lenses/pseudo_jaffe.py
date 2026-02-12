# mypy: disable-error-code="operator,dict-item"
from math import pi
from typing import Optional, Union, Annotated

from caskade import forward, Param

from ..backend_obj import backend, ArrayLike
from ..constants import arcsec_to_rad
from .base import ThinLens, CosmologyType, NameType, ZType
from . import func

__all__ = ("PseudoJaffe",)


class PseudoJaffe(ThinLens):
    """
    Class representing a Pseudo Jaffe lens in strong gravitational lensing,
    based on `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_ and
    the `lenstronomy` source code.

    Attributes
    ----------
    name: string
        The name of the Pseudo Jaffe lens.
    cosmology: Cosmology
        The cosmology used for calculations.
    z_l: Optional[Union[ArrayLike, float]]
        Redshift of the lens.

        *Unit: unitless*

    z_s: Optional[Union[ArrayLike, float]]
        Redshift of the source.

        *Unit: unitless*

    x0: Optional[Union[ArrayLike, float]]
        x-coordinate of the center of the lens (arcsec).

        *Unit: arcsec*

    y0: Optional[Union[ArrayLike, float]]
        y-coordinate of the center of the lens (arcsec).

        *Unit: arcsec*

    mass: Optional[Union[ArrayLike, float]]
        Total mass of the lens (Msun).

        *Unit: Msun*

    Rc: Optional[Union[ArrayLike, float]]
        Core radius of the lens (arcsec).

        *Unit: arcsec*

    Rs: Optional[Union[ArrayLike, float]]
        Scaling radius of the lens (arcsec).

        *Unit: arcsec*

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*

    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "mass": 1e12,
        "Rc": 0.1,
        "Rs": 1.0,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[ArrayLike, float]],
            "X coordinate of the center of the lens",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[ArrayLike, float]],
            "Y coordinate of the center of the lens",
            True,
        ] = None,
        mass: Annotated[
            Optional[Union[ArrayLike, float]], "Total mass of the lens", True, "Msol"
        ] = None,
        Rc: Annotated[
            Optional[Union[ArrayLike, float]], "Core radius of the lens", True, "arcsec"
        ] = None,
        Rs: Annotated[
            Optional[Union[ArrayLike, float]],
            "Scaling radius of the lens",
            True,
            "arcsec",
        ] = None,
        s: Annotated[
            float, "Softening parameter to prevent numerical instabilities"
        ] = 0.0,
        name: NameType = None,
    ):
        """
        Initialize the PseudoJaffe class.

        Parameters
        ----------
        name: string
            The name of the Pseudo Jaffe lens.

        cosmology: Cosmology
            The cosmology used for calculations.

        z_l: Optional[ArrayLike]
            Redshift of the lens.

            *Unit: unitless*

        x0: Optional[ArrayLike]
            x-coordinate of the center of the lens.

            *Unit: arcsec*

        y0: Optional[ArrayLike]
            y-coordinate of the center of the lens.

            *Unit: arcsec*

        mass: Optional[ArrayLike]
            Total mass of the lens (Msun).

            *Unit: Msun*

        Rc: Optional[ArrayLike]
            Core radius of the lens.

            *Unit: arcsec*

        Rs: Optional[ArrayLike]
            Scaling radius of the lens.

            *Unit: arcsec*

        s: float
            Softening parameter to prevent numerical instabilities.

            *Unit: arcsec*

        """
        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, shape=(), units="arcsec")
        self.y0 = Param("y0", y0, shape=(), units="arcsec")
        self.mass = Param("mass", mass, shape=(), units="Msun", valid=(0, None))
        self.Rc = Param("Rc", Rc, shape=(), units="arcsec", valid=(0, None))
        self.Rs = Param("Rs", Rs, shape=(), units="arcsec", valid=(0, None))
        self.s = s

    @forward
    def get_convergence_0(
        self,
        z_s: Annotated[ArrayLike, "Param"],
        z_l: Annotated[ArrayLike, "Param"],
        mass: Annotated[ArrayLike, "Param"],
        Rc: Annotated[ArrayLike, "Param"],
        Rs: Annotated[ArrayLike, "Param"],
    ):
        d_l = self.cosmology.angular_diameter_distance(z_l)
        sigma_crit = self.cosmology.critical_surface_density(z_l, z_s)
        return mass / (2 * backend.pi * sigma_crit * Rc * Rs * (d_l * arcsec_to_rad) ** 2)  # fmt: skip

    @forward
    def mass_enclosed_2d(
        self,
        theta,
        mass: Annotated[ArrayLike, "Param"],
        Rc: Annotated[ArrayLike, "Param"],
        Rs: Annotated[ArrayLike, "Param"],
    ):
        """
        Calculate the mass enclosed within a two-dimensional radius. Using equation A10 from `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_.

        Parameters
        ----------
        theta: ArrayLike
            Radius at which to calculate enclosed mass (arcsec).

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The mass enclosed within the given radius.

            *Unit: Msun*

        """
        return func.mass_enclosed_2d_pseudo_jaffe(theta, mass, Rc, Rs)

    @staticmethod
    def central_convergence(
        rho_0,
        Rc,
        Rs,
        critical_surface_density,
    ):
        """
        Compute the central convergence.

        Parameters
        -----------
        rho_0: ArrayLike
            Central mass density.

            *Unit: Msun/Mpc^3*

        Rc: ArrayLike
            Core radius of the lens (must be in Mpc).

            *Unit: Mpc*

        Rs: ArrayLike
            Scaling radius of the lens (must be in Mpc).

            *Unit: Mpc*

        cosmology: Cosmology
            The cosmology used for calculations.

        Returns
        --------
        ArrayLike
            The central convergence.

            *Unit: unitless*

        """
        return pi * rho_0 * Rc * Rs / ((Rc + Rs) * critical_surface_density)  # fmt: skip

    @forward
    def reduced_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z_s: Annotated[ArrayLike, "Param"],
        z_l: Annotated[ArrayLike, "Param"],
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        mass: Annotated[ArrayLike, "Param"],
        Rc: Annotated[ArrayLike, "Param"],
        Rs: Annotated[ArrayLike, "Param"],
    ) -> tuple[ArrayLike, ArrayLike]:
        """Calculate the deflection angle.

        Parameters
        ----------
        x: ArrayLike
            x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        --------
        x_component: ArrayLike
            x-component of the deflection angle.

            *Unit: arcsec*

        y_component: ArrayLike
            y-component of the deflection angle.

            *Unit: arcsec*

        """
        d_l = self.cosmology.angular_diameter_distance(z_l)
        critical_surface_density = self.cosmology.critical_surface_density(z_l, z_s)
        return func.reduced_deflection_angle_pseudo_jaffe(x0, y0, mass, Rc, Rs, x, y, d_l, critical_surface_density)  # fmt: skip

    @forward
    def potential(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z_s: Annotated[ArrayLike, "Param"],
        z_l: Annotated[ArrayLike, "Param"],
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        mass: Annotated[ArrayLike, "Param"],
        Rc: Annotated[ArrayLike, "Param"],
        Rs: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Compute the lensing potential. This calculation is based on equation A18 from `Eliasdottir et al 2007 <https://arxiv.org/abs/0710.5636>`_.

        Parameters
        --------
        x: ArrayLike
            x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        --------
        ArrayLike
            The lensing potential (arcsec^2).

            *Unit: arcsec^2*

        """

        d_l = self.cosmology.angular_diameter_distance(z_l)  # Mpc
        d_s = self.cosmology.angular_diameter_distance(z_s)  # Mpc
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)  # Mpc

        return func.potential_pseudo_jaffe(x0, y0, mass, Rc, Rs, x, y, d_l, d_s, d_ls)  # fmt: skip

    @forward
    def convergence(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z_s: Annotated[ArrayLike, "Param"],
        z_l: Annotated[ArrayLike, "Param"],
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        mass: Annotated[ArrayLike, "Param"],
        Rc: Annotated[ArrayLike, "Param"],
        Rs: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Calculate the projected mass density, based on equation A6.

        Parameters
        -----------
        x: ArrayLike
            x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The projected mass density.

            *Unit: unitless*

        """
        d_l = self.cosmology.angular_diameter_distance(z_l)
        critical_surface_density = self.cosmology.critical_surface_density(z_l, z_s)
        return func.convergence_pseudo_jaffe(
            x0, y0, mass, Rc, Rs, x, y, d_l, critical_surface_density
        )
