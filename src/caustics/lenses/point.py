from typing import Optional, Union

from torch import Tensor

from ..cosmology import Cosmology
from .base import ThinLens
from ..parametrized import unpack
from ..packed import Packed
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
    x0: Optional[Union[Tensor, float]]
        x-coordinate of the center of the lens.
    y0: Optional[Union[Tensor, float]]
        y-coordinate of the center of the lens.
    th_ein: Optional[Union[Tensor, float]]
        Einstein radius of the lens.
    s: float
        Softening parameter to prevent numerical instabilities.
    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "th_ein": 1.0,
    }

    def __init__(
        self,
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        th_ein: Optional[Union[Tensor, float]] = None,
        s: float = 0.0,
        name: str = None,
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
        x0: Optional[Tensor]
            x-coordinate of the center of the lens.
        y0: Optional[Tensor]
            y-coordinate of the center of the lens.
        th_ein: Optional[Tensor]
            Einstein radius of the lens.
        s: float
            Softening parameter to prevent numerical instabilities.
        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("th_ein", th_ein)
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
        th_ein: Tensor = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the deflection angles.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.
        y: Tensor
            y-coordinates in the lens plane.
        z_s: Tensor
            Redshifts of the sources.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        tuple[Tensor, Tensor]
            The deflection angles in the x and y directions.
        """
        return func.reduced_deflection_angle_point(x0, y0, th_ein, x, y, self.s)

    @unpack
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Tensor = None,
        x0: Tensor = None,
        y0: Tensor = None,
        th_ein: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the lensing potential.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.
        y: Tensor
            y-coordinates in the lens plane.
        z_s: Tensor
            Redshifts of the sources.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The lensing potential.
        """
        return func.potential_point(x0, y0, th_ein, x, y, self.s)

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
        th_ein: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the convergence (dimensionless surface mass density).

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.
        y: Tensor
            y-coordinates in the lens plane.
        z_s: Tensor
            Redshifts of the sources.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        --------
        Tensor
            The convergence (dimensionless surface mass density).
        """
        return func.convergence_point(x0, y0, x, y)
