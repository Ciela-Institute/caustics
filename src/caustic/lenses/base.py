from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

from torch import Tensor

from ..constants import arcsec_to_rad, c_Mpc_s
from ..cosmology import Cosmology
from ..parametrized import Parametrized

__all__ = ("ThinLens", "ThickLens")


class ThickLens(Parametrized):
    """
    Base class for lenses that can't be treated in the thin lens approximation.
    """

    def __init__(self, name: str, cosmology: Cosmology):
        super().__init__(name)
        self.cosmology = cosmology

    @abstractmethod
    def alpha(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tuple[Tensor, Tensor]:
        """
        Reduced deflection angle [arcsec]
        """
        ...

    def raytrace(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tuple[Tensor, Tensor]:
        ax, ay = self.alpha(thx, thy, z_s, x)
        return thx - ax, thy - ay

    @abstractmethod
    def Sigma(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tensor:
        """
        Projected mass density.

        Returns:
            [solMass / Mpc^2]
        """
        ...

    @abstractmethod
    def time_delay(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tensor:
        ...

    def magnification(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tensor:
        raise NotImplementedError()
        # return get_magnification(
        #     self.raytrace, thx, thy, z_l, z_s, cosmology, *args, **kwargs
        # )


class ThinLens(Parametrized):
    """
    Base class for lenses that can be treated in the thin lens approximation.
    """

    def __init__(self, name: str, cosmology: Cosmology, z_l: Optional[Tensor] = None):
        super().__init__(name)
        self.cosmology = cosmology
        self.add_param("z_l", z_l)

    @abstractmethod
    def alpha(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tuple[Tensor, Tensor]:
        """
        Reduced deflection angle [arcsec]
        """
        ...

    def alpha_hat(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tuple[Tensor, Tensor]:
        """
        Physical deflection angle immediately after passing through this lens'
        plane [arcsec].
        """
        z_l = self.unpack(x)[0]

        d_s = self.cosmology.angular_diameter_dist(z_s, x)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s, x)
        alpha_x, alpha_y = self.alpha(thx, thy, z_s, x)
        return (d_s / d_ls) * alpha_x, (d_s / d_ls) * alpha_y

    @abstractmethod
    def kappa(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tensor:
        """
        Convergence [1]
        """
        ...

    @abstractmethod
    def Psi(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tensor:
        """
        Potential [arcsec^2]
        """
        ...

    def Sigma(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tensor:
        """
        Surface mass density.

        Returns:
            [solMass / Mpc^2]
        """
        # Superclass params come before subclass ones
        z_l = self.unpack(x)[0]

        Sigma_cr = self.cosmology.Sigma_cr(z_l, z_s, x)
        return self.kappa(thx, thy, z_s, x) * Sigma_cr

    def raytrace(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tuple[Tensor, Tensor]:
        ax, ay = self.alpha(thx, thy, z_s, x)
        return thx - ax, thy - ay

    def time_delay(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ):
        z_l = self.unpack(x)[0]

        d_l = self.cosmology.angular_diameter_dist(z_l, x)
        d_s = self.cosmology.angular_diameter_dist(z_s, x)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s, x)
        ax, ay = self.alpha(thx, thy, z_s, x)
        Psi = self.Psi(thx, thy, z_s, x)
        factor = (1 + z_l) / c_Mpc_s * d_s * d_l / d_ls
        fp = 0.5 * d_ls**2 / d_s**2 * (ax**2 + ay**2) - Psi
        return factor * fp * arcsec_to_rad**2

    def magnification(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ):
        raise NotImplementedError()
        # return get_magnification(
        #     self.raytrace, thx, thy, z_l, z_s, cosmology, *args, **kwargs
        # )
