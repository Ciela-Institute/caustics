# mypy: disable-error-code="operator,union-attr,dict-item"
from typing import Optional, Union, Annotated, Literal
from warnings import warn

from caskade import forward, Param

from . import func
from .base import ThinLens, CosmologyType, NameType, ZType
from ..angle_mixin import Angle_Mixin
from ..backend_obj import ArrayLike, backend

__all__ = ("SIE",)


class SIE(Angle_Mixin, ThinLens):
    """
    A class representing a Singular Isothermal Ellipsoid (SIE) strong gravitational lens model.
    This model is based on Keeton 2001, which can be found at https://arxiv.org/abs/astro-ph/0102341.

    Attributes
    ----------
    name: str
        The name of the lens.

    cosmology: Cosmology
        An instance of the Cosmology class.

    z_l: Optional[Union[ArrayLike, float]]
        The redshift of the lens.

        *Unit: unitless*

    z_s: Optional[Union[ArrayLike, float]]
        The redshift of the source.

        *Unit: unitless*

    x0: Optional[Union[ArrayLike, float]]
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Optional[Union[ArrayLike, float]]
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: Optional[Union[ArrayLike, float]]
        The axis ratio of the lens.

        *Unit: unitless*

    phi: Optional[Union[ArrayLike, float]]
        The orientation angle of the lens (position angle).

        *Unit: radians*

    Rein: Optional[Union[ArrayLike, float]]
        The Einstein radius of the lens.

        *Unit: arcsec*

    s: float
        The core radius of the lens (defaults to 0.0).

        *Unit: arcsec*

    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "q": 0.5,
        "phi": 0.0,
        "Rein": 1.0,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[ArrayLike, float]],
            "The x-coordinate of the lens center",
            True,
        ] = None,
        y0: Annotated[
            Optional[Union[ArrayLike, float]],
            "The y-coordinate of the lens center",
            True,
        ] = None,
        q: Annotated[
            Optional[Union[ArrayLike, float]],
            "The axis ratio of the lens convergence",
            True,
        ] = None,
        phi: Annotated[
            Optional[Union[ArrayLike, float]],
            "The orientation angle of the lens (position angle)",
            True,
        ] = None,
        Rein: Annotated[
            Optional[Union[ArrayLike, float]], "The Einstein radius of the lens", True
        ] = None,
        parametrization: Literal["Rein", "velocity_dispersion"] = "Rein",
        sigma_v: Optional[Union[ArrayLike, float]] = None,
        angle_system: str = "q_phi",
        e1: Optional[Union[ArrayLike, float]] = None,
        e2: Optional[Union[ArrayLike, float]] = None,
        c1: Optional[Union[ArrayLike, float]] = None,
        c2: Optional[Union[ArrayLike, float]] = None,
        s: Annotated[float, "The core radius of the lens"] = 0.0,
        name: NameType = None,
        **kwargs,
    ):
        """
        Initialize the SIE lens model.
        """
        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, shape=(), units="arcsec")
        self.y0 = Param("y0", y0, shape=(), units="arcsec")
        self.q = Param("q", q, shape=(), units="unitless", valid=(0, 1))
        self.phi = Param(
            "phi", phi, shape=(), units="radians", valid=(0, backend.pi), cyclic=True
        )
        self.Rein = Param("Rein", Rein, shape=(), units="arcsec", valid=(0, None))
        self._parametrization = "Rein"
        self.parametrization = parametrization
        if self.parametrization == "velocity_dispersion":
            self.sigma_v = sigma_v
        self.angle_system = angle_system
        if self.angle_system == "e1_e2":
            self.e1 = e1
            self.e2 = e2
        elif self.angle_system == "c1_c2":
            self.c1 = c1
            self.c2 = c2
        self.s = s

    @property
    def parametrization(self) -> str:
        return self._parametrization

    @parametrization.setter
    def parametrization(self, value: str):
        if value not in ["Rein", "velocity_dispersion"]:
            raise ValueError(
                f"Invalid parametrization: {value}. Must be 'Rein' or 'velocity_dispersion'."
            )
        if (
            value == "velocity_dispersion"
            and self._parametrization != "velocity_dispersion"
        ):
            self.sigma_v = Param(
                "sigma_v",
                shape=self.Rein.shape if self.Rein.static else (),
                units="km/s",
                valid=(0, None),
            )
            if self.Rein.static:
                warn(
                    f"Parameter {self.Rein.name} is static, value now overridden by new {value} parametrization. To remove this warning, have {self.Rein.name} be dynamic when changing parametrizations.",
                )

            def sigma_v_to_rein(p):
                Dls = p["cosmology"].angular_diameter_distance_z1z2(
                    p["z_l"].value, p["z_s"].value
                )
                Ds = p["cosmology"].angular_diameter_distance(p["z_s"].value)
                return func.sigma_v_to_rein_sie(p["sigma_v"].value, Dls, Ds)

            self.Rein.value = lambda p: sigma_v_to_rein(p)
            self.Rein.link(self.sigma_v)
            self.Rein.link(self.z_s)
            self.Rein.link(self.z_l)
            self.Rein.link("cosmology", self.cosmology)
        if value == "Rein" and self.parametrization != "Rein":
            try:
                self.Rein = None
                if self.sigma_v.static:
                    warn(
                        f"Parameter {self.sigma_v.name} was static, value now overridden by new {value} parametrization. To remove this warning, have {self.sigma_v.name} be dynamic when changing parametrizations.",
                    )
                del self.sigma_v
            except AttributeError:
                pass

        self._parametrization = value

    @forward
    def reduced_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        q: Annotated[ArrayLike, "Param"],
        phi: Annotated[ArrayLike, "Param"],
        Rein: Annotated[ArrayLike, "Param"],
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculate the physical deflection angle.

        Parameters
        ----------
        x: ArrayLike
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        --------
        x_component: ArrayLike
            The x-component of the deflection angle.

            *Unit: arcsec*

        y_component: ArrayLike
            The y-component of the deflection angle.

            *Unit: arcsec*

        """
        return func.reduced_deflection_angle_sie(x0, y0, q, phi, Rein, x, y, self.s)

    @forward
    def potential(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        q: Annotated[ArrayLike, "Param"],
        phi: Annotated[ArrayLike, "Param"],
        Rein: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Compute the lensing potential.

        Parameters
        ----------
        x: ArrayLike
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The lensing potential.

            *Unit: arcsec^2*

        """
        return func.potential_sie(x0, y0, q, phi, Rein, x, y, self.s)

    @forward
    def convergence(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
        q: Annotated[ArrayLike, "Param"],
        phi: Annotated[ArrayLike, "Param"],
        Rein: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Calculate the projected mass density.

        Parameters
        ----------
        x: ArrayLike
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The projected mass density.

            *Unit: unitless*

        """
        return func.convergence_sie(x0, y0, q, phi, Rein, x, y, self.s)
