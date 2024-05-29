# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor

from .base import ThinLens, CosmologyType, NameType, ZLType
from ..parametrized import unpack
from ..packed import Packed
from . import func

__all__ = ("Multipole",)


class Multipole(ThinLens):
    """
    Represents a multipole effect in a gravitational lensing system.

    Attributes
    ----------
    name: str
        Identifier for the lens instance.
    cosmology: Cosmology
        The cosmological model used for lensing calculations.
    z_l: Optional[Union[Tensor, float]]
        The redshift of the lens.
    x0, y0: Optional[Union[Tensor, float]]
        Coordinates of the shear center in the lens plane.
    a_m: Optional[Union[Tensor, float]]
        Strength of multipole.
    phi_m: Optional[Union[Tensor, float]]
        Orientation of multiple.
    m: Optional[Union[Tensor, int]]
        Order of multipole.
        
    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "a_m": 0.1,
        "phi_m": 0.,
        "m": 2
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        a_m: Optional[Union[Tensor, float]] = None,
        phi_m: Optional[Union[Tensor, float]] = None,
        m: Optional[Union[Tensor, int]] = None,
        s: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("a_m", a_m)
        self.add_param("phi_m", phi_m)
        if m < 3:
            raise ValueError("Multipole order must be greater than 2")
        self.add_param("m", m)
        self.s = s

    @unpack
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        a_m: Optional[Tensor] = None,
        phi_m: Optional[Tensor] = None,
        m: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the deflection angle of the multipole.

        Parameters
        ----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        x_component: Tensor
            Deflection Angle

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle

            *Unit: arcsec*

        Equation (B11) and (B12) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

        """
        return func.reduced_deflection_angle_multipole(x0, y0, m, a_m, phi_m, x, y)

    @unpack
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        a_m: Optional[Tensor] = None,
        phi_m: Optional[Tensor] = None,
        m: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the lensing potential of the multiplane.

        Parameters
        ----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        Equation (B11) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

        """
        return func.potential_multipole(x0, y0, m, a_m, phi_m, x, y)

    @unpack
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        a_m: Optional[Tensor] = None,
        phi_m: Optional[Tensor] = None,
        m: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the projected mass density of the multipole.

        Parameters
        ----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The projected mass density.

            *Unit: unitless*
        
        Equation (B10) and (B3) https://arxiv.org/pdf/1307.4220, Xu et al. 2014

        """
        return func.convergence_multipole(x0, y0, m, a_m, phi_m, x, y)
