# mypy: disable-error-code="operator,union-attr,dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor
from caskade import forward, Param

from .base import ThinLens, CosmologyType, NameType, ZType
from . import func

DELTA = 200.0

__all__ = ("TNFW",)


class TNFW(ThinLens):
    """Truncated Navaro-Frenk-White profile

    TNFW lens class. This class models a lens using the truncated
    Navarro-Frenk-White (NFW) profile.  The NFW profile is a spatial
    density profile of dark matter halo that arises in cosmological
    simulations. It is truncated with an extra scaling term which
    smoothly reduces the density such that it does not diverge to
    infinity. This is based off the paper by Baltz et al. 2009:

    https://arxiv.org/abs/0705.0682

    https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract

    Notes
    ------
        The mass `m` in the TNFW profile corresponds to the total mass
        of the lens. This is different from the NFW profile where the
        mass `m` parameter corresponds to the mass within R200. If you
        prefer the "mass within R200" version you can set:
        `interpret_m_total_mass = False` on initialization of the
        object. However, the mass within R200 will be computed for an
        NFW profile, not a TNFW profile. This is in line with how
        lenstronomy interprets the mass parameter.

    Parameters
    -----
    name: string
        Name of the lens instance.

    cosmology: Cosmology
        An instance of the Cosmology class which contains
        information about the cosmological model and parameters.

    z_l: Optional[Tensor]
        Redshift of the lens.

        *Unit: unitless*

    z_s: Optional[Tensor]
        Redshift of the source.

        *Unit: unitless*

    x0: Optional[Tensor]
        Center of lens position on x-axis.

        *Unit: arcsec*

    y0: Optional[Tensor]
        Center of lens position on y-axis.

        *Unit: arcsec*

    mass: Optional[Tensor]
        Mass of the lens.

        *Unit: Msun*

    Rs: Optional[Tensor]
        Scale radius of the TNFW lens.

        *Unit: arcsec*

    tau: Optional[Tensor]
        Truncation scale. Ratio of truncation radius to scale radius.

        *Unit: unitless*

    s: float
        Softening parameter to avoid singularities at the center of the lens.
        Default is 0.0.

        *Unit: arcsec*

    interpret_m_total_mass: boolean
        Indicates how to interpret the mass variable "m". If true
        the mass is interpreted as the total mass of the halo (good because it makes sense). If
        false it is interpreted as what the mass would have been within R200 of a an NFW that
        isn't truncated (good because it is easily compared with an NFW).

    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "mass": 1e13,
        "Rs": 1.0,
        "tau": 3.0,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]],
            "Center of lens position on x-axis",
            True,
            "arcsec",
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]],
            "Center of lens position on y-axis",
            True,
            "arcsec",
        ] = None,
        mass: Annotated[
            Optional[Union[Tensor, float]], "Mass of the lens", True, "Msol"
        ] = None,
        Rs: Annotated[
            Optional[Union[Tensor, float]],
            "Scale radius of the TNFW lens",
            True,
            "arcsec",
        ] = None,
        tau: Annotated[
            Optional[Union[Tensor, float]],
            "Truncation scale. Ratio of truncation radius to scale radius",
            True,
            "rt/rs",
        ] = None,
        s: Annotated[
            float,
            "Softening parameter to avoid singularities at the center of the lens",
        ] = 0.0,
        interpret_m_total_mass: Annotated[
            bool, "Indicates how to interpret the mass variable 'm'"
        ] = True,
        name: NameType = None,
    ):
        """
        Initialize an instance of the TNFW lens class.

        """
        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.mass = Param("mass", mass, units="Msun", valid=(0, None))
        self.Rs = Param("Rs", Rs, units="arcsec", valid=(0, None))
        self.tau = Param("tau", tau, units="unitless", valid=(0, None))
        self.s = s
        self.interpret_m_total_mass = interpret_m_total_mass

    @forward
    def get_concentration(
        self,
        z_l: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Compute the concentration parameter "c" for a TNFW profile.

        Parameters
        ----------
        z_l: Tensor
            Redshift of the lens.

            *Unit: unitless*

        x0: Tensor
            Center of lens position on x-axis.

            *Unit: arcsec*

        y0: Tensor
            Center of lens position on y-axis.

            *Unit: arcsec*

        mass: Optional[Tensor]
            Mass of the lens.

            *Unit: Msun*

        Rs: Optional[Tensor]
            Scale radius of the TNFW lens.

            *Unit: arcsec*

        tau: Optional[Tensor]
            Truncation scale. Ratio of truncation radius to scale radius.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The concentration parameter "c" for a TNFW profile.

            *Unit: unitless*

        """
        critical_density = self.cosmology.critical_density(z_l)
        d_l = self.cosmology.angular_diameter_distance(z_l)
        return func.concentration_tnfw(mass, Rs, critical_density, d_l, DELTA)

    @forward
    def get_truncation_radius(
        self,
        Rs: Annotated[Tensor, "Param"],
        tau: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Calculate the truncation radius of the TNFW lens.

        Returns
        -------
        Tensor
            The truncation radius of the lens.

            *Unit: arcsec*

        """
        return tau * Rs

    @forward
    def M0(
        self,
        z_l: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
        tau: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Calculate the reference mass.
        This is an abstract reference mass used internally
        in the equations from Baltz et al. 2009.


        Returns
        -------
        Tensor
            The reference mass of the lens in Msun.

            *Unit: Msun*

        """
        if self.interpret_m_total_mass:
            return func.M0_totmass_tnfw(mass, tau)
        else:
            d_l = self.cosmology.angular_diameter_distance(z_l)
            critical_density = self.cosmology.critical_density(z_l)
            c = func.concentration_tnfw(mass, Rs, critical_density, d_l, DELTA)
            return func.M0_scalemass_tnfw(Rs, c, critical_density, d_l, DELTA)

    @forward
    def get_scale_density(
        self,
        z_l: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Calculate the scale density of the lens.

        Returns
        --------
        Tensor
            The scale density of the lens.

            *Unit: Msun/Mpc^3*

        """
        d_l = self.cosmology.angular_diameter_distance(z_l)
        critical_density = self.cosmology.critical_density(z_l)
        c = func.concentration_tnfw(mass, Rs, critical_density, d_l, DELTA)
        return func.scale_density_tnfw(c, critical_density, DELTA)

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Annotated[Tensor, "Param"],
        z_l: Annotated[Tensor, "Param"],
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
        tau: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        TNFW convergence as given in Baltz et al. 2009.
        This is unitless since it is Sigma(x) / Sigma_crit.

        Parameters
        ----------
        x: Tensor
            The x-coordinate on the lens plane.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate on the lens plane.

            *Unit: arcsec*

        Returns
        ---------
        Tensor
            Convergence at requested position.

            *Unit: unitless*

        """

        d_l = self.cosmology.angular_diameter_distance(z_l)
        critical_density = self.cosmology.critical_surface_density(z_l, z_s)
        M0 = self.M0(z_l=z_l, mass=mass, Rs=Rs, tau=tau)
        return func.convergence_tnfw(
            x0,
            y0,
            Rs,
            tau,
            x,
            y,
            critical_density,
            M0,
            d_l,
            self.s,
        )

    @forward
    def mass_enclosed_2d(
        self,
        r: Tensor,
        Rs: Annotated[Tensor, "Param"],
        tau: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Total projected mass (Msun) within a radius r (arcsec).

        Parameters
        -----------
        r: Tensor
            Radius within which to calculate the mass.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            Integrated mass projected in infinite cylinder within radius r.

            *Unit: Msun*

        """

        M0 = self.M0()
        return func.mass_enclosed_2d_tnfw(r, Rs, tau, M0)

    @forward
    def physical_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_l: Annotated[Tensor, "Param"],
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
        tau: Annotated[Tensor, "Param"],
    ) -> tuple[Tensor, Tensor]:
        """Compute the physical deflection angle (arcsec) for this lens at
        the requested position. Note that the NFW/TNFW profile is more
        naturally represented as a physical deflection angle, this is
        easily internally converted to a reduced deflection angle.

        Parameters
        ----------
        x: Tensor
            The x-coordinate on the lens plane.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate on the lens plane.

            *Unit: arcsec*

        Returns
        --------
        x_component: Tensor
            Deflection Angle in x-direction.

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle in y-direction.

            *Unit: arcsec*

        """
        d_l = self.cosmology.angular_diameter_distance(z_l)
        M0 = self.M0(z_l=z_l, mass=mass, Rs=Rs, tau=tau)
        return func.physical_deflection_angle_tnfw(
            x0, y0, Rs, tau, x, y, M0, d_l, self.s
        )

    @forward
    def reduced_deflection_angle(self, x, y, z_s, z_l):
        d_s = self.cosmology.angular_diameter_distance(z_s)
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)
        deflection_angle_x, deflection_angle_y = self.physical_deflection_angle(x, y)
        return func.reduced_from_physical_deflection_angle(
            deflection_angle_x, deflection_angle_y, d_s, d_ls
        )

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Annotated[Tensor, "Param"],
        z_l: Annotated[Tensor, "Param"],
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        mass: Annotated[Tensor, "Param"],
        Rs: Annotated[Tensor, "Param"],
        tau: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Compute the lensing potential.
        Note that this is not a unitless potential!
        This is the potential as given in Baltz et al. 2009.

        TODO: convert to dimensionless potential.

        Parameters
        -----------
        x: Tensor
            x-coordinate in the lens plane.

            *Unit: arcsec*

        y: Tensor
            y-coordinate in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """

        d_l = self.cosmology.angular_diameter_distance(z_l)
        d_s = self.cosmology.angular_diameter_distance(z_s)
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)

        M0 = self.M0(z_l=z_l, mass=mass, Rs=Rs, tau=tau)
        return func.potential_tnfw(x0, y0, Rs, tau, x, y, M0, d_l, d_s, d_ls, self.s)
