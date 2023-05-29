from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import derotate, translate_rotate
from .base import ThinLens

__all__ = ("EPL",)


class EPL(ThinLens):
    """
    Elliptical power law (EPL, aka singular power-law ellipsoid) profile.

    This class represents a thin gravitational lens model with an elliptical power law profile. The lensing equations are solved 
    iteratively using an approach based on Tessore et al. 2015.

    Attributes:
        n_iter (int): Number of iterations for the iterative solver.
        s (float): Softening length for the elliptical power-law profile.
    
    Parameters:
        z_l (Tensor, optional): This is the redshift of the lens. In the context of gravitational lensing, the lens is the galaxy or other mass distribution that is bending the light from a more distant source.
        thx0 and thy0 (Tensors, optional): These are the coordinates of the lens center in the lens plane. The lens plane is the plane perpendicular to the line of sight in which the deflection of light by the lens is considered.
        q (Tensor, optional): This is the axis ratio of the lens, i.e., the ratio of the minor axis to the major axis of the elliptical lens.
        phi (Tensor, optional): This is the orientation of the lens on the sky, typically given as an angle measured counter-clockwise from some reference direction.
        b (Tensor, optional): This is the scale length of the lens, which sets the overall scale of the lensing effect. In some contexts, this is referred to as the Einstein radius.
        t (Tensor, optional): This is the power-law slope parameter of the lens model. In the context of the EPL model, t is equivalent to the gamma parameter minus one, where gamma is the power-law index of the radial mass distribution of the lens.

    """

    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        s: float = 0.0,
        n_iter: int = 18,
    ):
        """
        Initialize an EPL lens model.

        Args:
            name (str): Name of the lens model.
            cosmology (Cosmology): Cosmology object that provides cosmological distance calculations.
            z_l (Optional[Tensor]): Redshift of the lens. If not provided, it is considered as a free parameter.
            thx0 (Optional[Tensor]): X coordinate of the lens center. If not provided, it is considered as a free parameter.
            thy0 (Optional[Tensor]): Y coordinate of the lens center. If not provided, it is considered as a free parameter.
            q (Optional[Tensor]): Axis ratio of the lens. If not provided, it is considered as a free parameter.
            phi (Optional[Tensor]): Position angle of the lens. If not provided, it is considered as a free parameter.
            b (Optional[Tensor]): Scale length of the lens. If not provided, it is considered as a free parameter.
            t (Optional[Tensor]): Power law slope (`gamma-1`) of the lens. If not provided, it is considered as a free parameter.
            s (float): Softening length for the elliptical power-law profile.
            n_iter (int): Number of iterations for the iterative solver.
        """
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("b", b)
        self.add_param("t", t)
        self.s = s

        self.n_iter = n_iter

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the reduced deflection angles of the lens.

        Args:
            thx (Tensor): X coordinates in the lens plane.
            thy (Tensor): Y coordinates in the lens plane.
            z_s (Tensor): Source redshifts.
            x (Optional[dict[str, Any]]): Additional parameters for the lens model.

        Returns:
            tuple[Tensor, Tensor]: Reduced deflection angles in the x and y directions.
        """
        z_l, thx0, thy0, q, phi, b, t = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)

        # follow Tessore et al 2015 (eq. 5)
        z = q * thx + thy * 1j
        r = torch.abs(z)

        # Tessore et al 2015 (eq. 23)
        r_omega = self._r_omega(z, t, q)
        alpha_c = 2.0 / (1.0 + q) * (b / r) ** t * r_omega

        alpha_real = torch.nan_to_num(alpha_c.real, posinf=10**10, neginf=-(10**10))
        alpha_imag = torch.nan_to_num(alpha_c.imag, posinf=10**10, neginf=-(10**10))
        return derotate(alpha_real, alpha_imag, phi)

    def _r_omega(self, z, t, q):
        """
        Iteratively computes `R * omega(phi)` (eq. 23 in Tessore et al 2015).

        Args:
            z (Tensor): `R * e^(i * phi)`, position vector in the lens plane.
            t (Tensor): Power law slow (`gamma-1`).
            q (Tensor): Axis ratio.

        Returns:
            Tensor: The value of `R * omega(phi)`.
        """
        # constants
        f = (1.0 - q) / (1.0 + q)
        phi = z / torch.conj(z)

        # first term in series
        omega_i = z
        part_sum = omega_i

        for i in range(1, self.n_iter):
            factor = (2.0 * i - (2.0 - t)) / (2.0 * i + (2.0 - t))
            omega_i = -f * factor * phi * omega_i
            part_sum = part_sum + omega_i

        return part_sum

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ):
        """
        Compute the lensing potential of the lens.

        Args:
            thx (Tensor): X coordinates in the lens plane.
            thy (Tensor): Y coordinates in the lens plane.
            z_s (Tensor): Source redshifts.
            x (Optional[dict[str, Any]]): Additional parameters for the lens model.

        Returns:
            Tensor: The lensing potential.
        """
        z_l, thx0, thy0, q, phi, b, t = self.unpack(x)

        ax, ay = self.alpha(thx, thy, z_s, x)
        ax, ay = derotate(ax, ay, -phi)
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        return (thx * ax + thy * ay) / (2 - t)

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ):
        """
        Compute the convergence of the lens, which describes the local density of the lens.

        Args:
            thx (Tensor): X coordinates in the lens plane.
            thy (Tensor): Y coordinates in the lens plane.
            z_s (Tensor): Source redshifts.
            x (Optional[dict[str, Any]]): Additional parameters for the lens model.

        Returns:
            Tensor: The convergence of the lens.
        """
        z_l, thx0, thy0, q, phi, b, t = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = (q**2 * (thx**2 + self.s**2) + thy**2).sqrt()
        return (2 - t) / 2 * (b / psi) ** t
