from typing import Any, Optional

from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

__all__ = ("ExternalShear",)


class ExternalShear(ThinLens):
    """
    Represents an external shear effect in a gravitational lensing system.

    Attributes:
        name (str): Identifier for the lens instance.
        cosmology (Cosmology): The cosmological model used for lensing calculations.
        z_l (Optional[Tensor]): The redshift of the lens.
        thx0, thy0 (Optional[Tensor]): Coordinates of the shear center in the lens plane.
        gamma_1, gamma_2 (Optional[Tensor]): Shear components.

    Note: The shear components gamma_1 and gamma_2 represent an external shear, a gravitational 
    distortion that can be caused by nearby structures outside of the main lens galaxy. 
    """
    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        gamma_1: Optional[Tensor] = None,
        gamma_2: Optional[Tensor] = None,
    ):
        
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("gamma_1", gamma_1)
        self.add_param("gamma_2", gamma_2)

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Calculates the reduced deflection angle.

        Args:
            thx (Tensor): x-coordinates in the lens plane.
            thy (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            tuple[Tensor, Tensor]: The reduced deflection angles in the x and y directions.
        """
        z_l, thx0, thy0, gamma_1, gamma_2 = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        # Meneghetti eq 3.83
        a1 = thx * gamma_1 + thy * gamma_2
        a2 = thx * gamma_2 - thy * gamma_1
        return a1, a2  # I'm not sure but I think no derotation necessary

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Calculates the lensing potential.

        Args:
            thx (Tensor): x-coordinates in the lens plane.
            thy (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            x (Optional[dict[str, Any]]): Additional parameters.

        Returns:
            Tensor: The lensing potential.
        """
        z_l, thx0, thy0, gamma_1, gamma_2 = self.unpack(x)

        ax, ay = self.alpha(thx, thy, z_s, x)
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        return 0.5 * (thx * ax + thy * ay)

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        The convergence is undefined for an external shear.

        Args:
            thx (Tensor): x-coordinates in the lens plane.
            thy (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            x (Optional[dict[str, Any]]): Additional parameters.

        Raises:
            NotImplementedError: This method is not implemented as the convergence is not defined 
            for an external shear.
        """
        raise NotImplementedError("convergence undefined for external shear")
