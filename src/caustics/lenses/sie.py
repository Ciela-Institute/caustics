# mypy: disable-error-code="operator,union-attr"
from typing import Optional, Union

from torch import Tensor

from ..cosmology import Cosmology
from ..utils import derotate, translate_rotate
from .base import ThinLens
from ..parametrized import unpack
from ..packed import Packed

__all__ = ("SIE",)


class SIE(ThinLens):
    """
    A class representing a Singular Isothermal Ellipsoid (SIE) strong gravitational lens model.
    This model is based on Keeton 2001, which can be found at https://arxiv.org/abs/astro-ph/0102341.

    Attributes
    ----------
    name: str
        The name of the lens.

    cosmology: Cosmology
        An instance of the Cosmology class.

    z_l: Optional[Union[Tensor, float]]
        The redshift of the lens.

        *Unit: unitless*

    x0: Optional[Union[Tensor, float]]
        The x-coordinate of the lens center.

        *Unit: arcsec*

    y0: Optional[Union[Tensor, float]]
        The y-coordinate of the lens center.

        *Unit: arcsec*

    q: Optional[Union[Tensor, float]]
        The axis ratio of the lens.

        *Unit: unitless*

    phi: Optional[Union[Tensor, float]]
        The orientation angle of the lens (position angle).

        *Unit: radians*

    b: Optional[Union[Tensor, float]]
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
        "b": 1.0,
    }

    def __init__(
        self,
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        q: Optional[Union[Tensor, float]] = None,  # TODO change to true axis ratio
        phi: Optional[Union[Tensor, float]] = None,
        b: Optional[Union[Tensor, float]] = None,
        s: float = 0.0,
        name: Optional[str] = None,
    ):
        """
        Initialize the SIE lens model.
        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("b", b)
        self.s = s

    def _get_potential(self, x, y, q):
        """
        Compute the radial coordinate in the lens plane.

        Parameters
        ----------
        x: Tensor
            The x-coordinate in the lens plane.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate in the lens plane.

            *Unit: arcsec*

        q: Tensor
            The axis ratio of the lens.

            *Unit: unitless*

        Returns
        --------
        Tensor
            The radial coordinate in the lens plane.

            *Unit: arcsec*

        """
        return (q**2 * (x**2 + self.s**2) + y**2).sqrt()  # fmt: skip

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
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the physical deflection angle.

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

        params: Packed, optional
            Dynamic parameter container.

        Returns
        --------
        x_component: Tensor
            The x-component of the deflection angle.

            *Unit: arcsec*

        y_component: Tensor
            The y-component of the deflection angle.

            *Unit: arcsec*

        """
        x, y = translate_rotate(x, y, x0, y0, phi)
        psi = self._get_potential(x, y, q)
        f = (1 - q**2).sqrt()
        ax = b * q.sqrt() / f * (f * x / (psi + self.s)).atan()  # fmt: skip
        ay = b * q.sqrt() / f * (f * y / (psi + q**2 * self.s)).atanh()  # fmt: skip

        return derotate(ax, ay, phi)

    @unpack
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        x0: Optional[Tensor] = None,
        z_l: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the lensing potential.

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

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        ax, ay = self.reduced_deflection_angle(x, y, z_s, params)
        ax, ay = derotate(ax, ay, -phi)
        x, y = translate_rotate(x, y, x0, y0, phi)
        return x * ax + y * ay

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
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the projected mass density.

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

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The projected mass density.

            *Unit: unitless*

        """
        x, y = translate_rotate(x, y, x0, y0, phi)
        psi = self._get_potential(x, y, q)
        return 0.5 * q.sqrt() * b / psi
