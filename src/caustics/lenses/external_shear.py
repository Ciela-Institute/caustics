from typing import Optional, Union

from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens
from ..parametrized import unpack

__all__ = ("ExternalShear",)


class ExternalShear(ThinLens):
    """
    Represents an external shear effect in a gravitational lensing system.

    Attributes:
        name (str): Identifier for the lens instance.
        cosmology (Cosmology): The cosmological model used for lensing calculations.
        z_l (Optional[Union[Tensor, float]]): The redshift of the lens.
        x0, y0 (Optional[Union[Tensor, float]]): Coordinates of the shear center in the lens plane.
        gamma_1, gamma_2 (Optional[Union[Tensor, float]]): Shear components.

    Note: The shear components gamma_1 and gamma_2 represent an external shear, a gravitational
    distortion that can be caused by nearby structures outside of the main lens galaxy.
    """

    def __init__(
        self,
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        gamma_1: Optional[Union[Tensor, float]] = None,
        gamma_2: Optional[Union[Tensor, float]] = None,
        s: float = 0.0,
        name: str = None,
    ):
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("gamma_1", gamma_1)
        self.add_param("gamma_2", gamma_2)
        self.s = s

    @unpack(3)
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        z_l,
        x0,
        y0,
        gamma_1,
        gamma_2,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculates the reduced deflection angle.

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            tuple[Tensor, Tensor]: The reduced deflection angles in the x and y directions.
        """
        x, y = translate_rotate(x, y, x0, y0)
        # Meneghetti eq 3.83
        # TODO, why is it not:
        # th = (x**2 + y**2).sqrt() + self.s
        # a1 = x/th + x * gamma_1 + y * gamma_2
        # a2 = y/th + x * gamma_2 - y * gamma_1
        a1 = x * gamma_1 + y * gamma_2
        a2 = x * gamma_2 - y * gamma_1

        return a1, a2  # I'm not sure but I think no derotation necessary

    @unpack(3)
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        z_l,
        x0,
        y0,
        gamma_1,
        gamma_2,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculates the lensing potential.

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: The lensing potential.
        """
        ax, ay = self.reduced_deflection_angle(x, y, z_s, params)
        x, y = translate_rotate(x, y, x0, y0)
        return 0.5 * (x * ax + y * ay)

    @unpack(3)
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        z_l,
        x0,
        y0,
        gamma_1,
        gamma_2,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        The convergence is undefined for an external shear.

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Raises:
            NotImplementedError: This method is not implemented as the convergence is not defined
            for an external shear.
        """
        raise NotImplementedError("convergence undefined for external shear")
