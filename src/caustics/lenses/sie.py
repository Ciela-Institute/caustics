from typing import Optional, Union

from torch import Tensor

from ..cosmology import Cosmology
from .base import ThinLens
from ..parametrized import unpack
from ..packed import Packed
from . import func

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
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        q: Optional[Union[Tensor, float]] = None,  # TODO change to true axis ratio
        phi: Optional[Union[Tensor, float]] = None,
        b: Optional[Union[Tensor, float]] = None,
        s: float = 0.0,
        name: str = None,
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

    @unpack
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Tensor = None,
        x0: Tensor = None,
        y0: Tensor = None,
        q: Tensor = None,
        phi: Tensor = None,
        b: Tensor = None,
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
        return func.reduced_deflection_angle_sie(x0, y0, q, phi, b, x, y, self.s)

    @unpack
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        x0: Tensor = None,
        z_l: Tensor = None,
        y0: Tensor = None,
        q: Tensor = None,
        phi: Tensor = None,
        b: Tensor = None,
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
        return func.potential_sie(x0, y0, q, phi, b, x, y, self.s)

    @unpack
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Tensor = None,
        x0: Tensor = None,
        y0: Tensor = None,
        q: Tensor = None,
        phi: Tensor = None,
        b: Tensor = None,
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
        return func.convergence_sie(x0, y0, q, phi, b, x, y, self.s)
