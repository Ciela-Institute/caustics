# mypy: disable-error-code="operator,union-attr,dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor

from ..utils import derotate, translate_rotate
from .base import ThinLens, CosmologyType, NameType, ZLType
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
    x0: Optional[Union[Tensor, float]]
        The x-coordinate of the lens center.
    y0: Optional[Union[Tensor, float]]
        The y-coordinate of the lens center.
    q: Optional[Union[Tensor, float]]
        The axis ratio of the lens.
    phi: Optional[Union[Tensor, float]]
        The orientation angle of the lens (position angle).
    b: Optional[Union[Tensor, float]]
        The Einstein radius of the lens.
    s: float
        The core radius of the lens (defaults to 0.0).
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
        cosmology: CosmologyType,
        z_l: ZLType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]], "The x-coordinate of the lens center", True
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]], "The y-coordinate of the lens center", True
        ] = None,
        q: Annotated[
            Optional[Union[Tensor, float]], "The axis ratio of the lens", True
        ] = None,  # TODO change to true axis ratio
        phi: Annotated[
            Optional[Union[Tensor, float]],
            "The orientation angle of the lens (position angle)",
            True,
        ] = None,
        b: Annotated[
            Optional[Union[Tensor, float]], "The Einstein radius of the lens", True
        ] = None,
        s: Annotated[float, "The core radius of the lens"] = 0.0,
        name: NameType = None,
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
        y: Tensor
            The y-coordinate in the lens plane.
        q: Tensor
            The axis ratio of the lens.

        Returns
        --------
        Tensor
            The radial coordinate in the lens plane.
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
        y: Tensor
            The y-coordinate of the lens.
        z_s: Tensor
            The source redshift.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        --------
        Tuple[Tensor, Tensor]
            The deflection angle in the x and y directions.
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
        y: Tensor
            The y-coordinate of the lens.
        z_s: Tensor
            The source redshift.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The lensing potential.
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
        y: Tensor
            The y-coordinate of the lens.
        z_s: Tensor
            The source redshift.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The projected mass.
        """
        x, y = translate_rotate(x, y, x0, y0, phi)
        psi = self._get_potential(x, y, q)
        return 0.5 * q.sqrt() * b / psi
