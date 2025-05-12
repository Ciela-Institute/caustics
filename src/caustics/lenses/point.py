# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated, Literal
from warnings import warn

from torch import Tensor
from caskade import forward, Param

from .base import ThinLens, CosmologyType, NameType, ZType
from . import func

__all__ = ("Point",)


class Point(ThinLens):
    """
    Class representing a point mass lens in strong gravitational lensing.

    Attributes
    ----------
    name: str
        The name of the point lens.

    cosmology: Cosmology
        The cosmology used for calculations.

    z_l: Optional[Union[Tensor, float]]
        Redshift of the lens.

        *Unit: unitless*

    x0: Optional[Union[Tensor, float]]
        x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: Optional[Union[Tensor, float]]
        y-coordinate of the center of the lens.

        *Unit: arcsec*

    Rein: Optional[Union[Tensor, float]]
        Einstein radius of the lens.

        *Unit: arcsec*

    s: float
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*

    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "Rein": 1.0,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]],
            "X coordinate of the center of the lens",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]],
            "Y coordinate of the center of the lens",
            True,
        ] = None,
        Rein: Annotated[
            Optional[Union[Tensor, float]], "Einstein radius of the lens", True
        ] = None,
        parametrization: Literal["Rein", "mass"] = "Rein",
        s: Annotated[
            float, "Softening parameter to prevent numerical instabilities"
        ] = 0.0,
        name: NameType = None,
        **kwargs,
    ):
        """
        Initialize the Point class.

        Parameters
        ----------
        name: string
            The name of the point lens.

        cosmology: Cosmology
            The cosmology used for calculations.

        z_l: Optional[Tensor]
            Redshift of the lens.

            *Unit: unitless*

        x0: Optional[Tensor]
            x-coordinate of the center of the lens.

            *Unit: arcsec*

        y0: Optional[Tensor]
            y-coordinate of the center of the lens.

            *Unit: arcsec*

        Rein: Optional[Tensor]
            Einstein radius of the lens.

            *Unit: arcsec*

        s: float
            Softening parameter to prevent numerical instabilities.

            *Unit: arcsec*

        """
        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.Rein = Param("Rein", Rein, units="arcsec", valid=(0, None))
        self._parametrization = "Rein"
        self.parametrization = parametrization
        if self.parametrization == "mass":
            self.mass = kwargs.get("mass", None)
        self.s = s

    @property
    def parametrization(self):
        return self._parametrization

    @parametrization.setter
    def parametrization(self, value: str):
        if value not in ["Rein", "mass"]:
            raise ValueError(
                f"Invalid parametrization {value}. Choose from ['Rein', 'mass']"
            )
        if value == "mass" and self.parametrization != "mass":
            self.mass = Param(
                "mass", shape=self.Rein.shape if self.Rein.static else (), units="Msol"
            )

            def mass_to_rein(p):
                Dls = p["cosmology"].angular_diameter_distance_z1z2(
                    p["z_l"].value, p["z_s"].value
                )
                Dl = p["cosmology"].angular_diameter_distance(p["z_l"].value)
                Ds = p["cosmology"].angular_diameter_distance(p["z_s"].value)
                return func.mass_to_rein_point(p["mass"].value, Dls, Dl, Ds)

            if self.Rein.static:
                warn(
                    f"Parameter {self.Rein.name} was static, value now overridden by new {value} parametrization. To remove this warning, have {self.Rein.name} be dynamic when changing parametrizations.",
                )
            self.Rein.value = mass_to_rein
            self.Rein.link(self.mass)
            self.Rein.link(self.z_s)
            self.Rein.link(self.z_l)
            self.Rein.link("cosmology", self.cosmology)
        if value == "Rein" and self.parametrization != "Rein":
            try:
                self.Rein = None
                if self.mass.static:
                    warn(
                        f"Parameter {self.mass.name} was static, value now overridden by new {value} parametrization. To remove this warning, have {self.mass.name} be dynamic when changing parametrizations.",
                    )
                del self.mass
            except AttributeError:
                pass

        self._parametrization = value

    @forward
    def mass_to_rein(
        self,
        mass: Tensor,
        z_s: Annotated[Tensor, "Param"],
        z_l: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Convert mass to the Einstein radius.

        Parameters
        ----------
        mass: Tensor
            The mass of the lens

            *Unit: solar mass*

        Returns
        -------
        Tensor
            The Einstein radius.

            *Unit: arcsec*

        """

        Dls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)
        Dl = self.cosmology.angular_diameter_distance(z_l)
        Ds = self.cosmology.angular_diameter_distance(z_s)
        return func.mass_to_rein_point(mass, Dls, Dl, Ds)

    @forward
    def rein_to_mass(
        self,
        r: Tensor,
        z_s: Annotated[Tensor, "Param"],
        z_l: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Convert Einstein radius to mass.

        Parameters
        ----------
        r: Tensor
            The Einstein radius.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The mass of the lens

            *Unit: solar mass*

        """

        Dls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)
        Dl = self.cosmology.angular_diameter_distance(z_l)
        Ds = self.cosmology.angular_diameter_distance(z_s)
        return func.rein_to_mass_point(r, Dls, Dl, Ds)

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        Rein: Annotated[Tensor, "Param"],
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the deflection angles.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.

            *Unit: arcsec*

        y: Tensor
            y-coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        x_component: Tensor
            Deflection Angle in the x-direction.

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle in the y-direction.

            *Unit: arcsec*

        """
        return func.reduced_deflection_angle_point(x0, y0, Rein, x, y, self.s)

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        Rein: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Compute the lensing potential.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.

            *Unit: arcsec*

        y: Tensor
            y-coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        return func.potential_point(x0, y0, Rein, x, y, self.s)

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Compute the convergence (dimensionless surface mass density).

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.

            *Unit: arcsec*

        y: Tensor
            y-coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        --------
        Tensor
            The convergence (dimensionless surface mass density).

            *Unit: unitless*

        """
        return func.convergence_point(x0, y0, x, y)

    @forward
    def magnification(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "param"],
        y0: Annotated[Tensor, "param"],
        Rein: Annotated[Tensor, "param"],
    ) -> Tensor:
        """
        Compute the magnification for a point mass lens.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.

            *Unit: arcsec*
        y: Tensor
            y-coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The magnification.

            *Unit: unitless*

        """
        return func.magnification_point(x0, y0, Rein, x, y, self.s)
