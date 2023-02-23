from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import derotate, translate_rotate
from .base import ThinLens

__all__ = ("EPL",)


class EPL(ThinLens):
    """
    Elliptical power law (aka singular power-law ellipsoid) profile.
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
        s: Optional[Tensor] = torch.tensor(0.0),
        n_iter: int = 18,
    ):
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("b", b)
        self.add_param("t", t)
        self.add_param("s", s)

        self.n_iter = n_iter

    def alpha(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            b: scale length.
            t: power law slope (`gamma-1`).
            s: core radius.
        """
        z_l, thx0, thy0, q, phi, b, t, s = self.unpack(x)

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
        Iteratively computes `R * omega(phi)` (eq. 23 in the reference).

        Args:
            z: `R * e^(i * phi)`, position vector in the lens plane.
            t: power law slow (`gamma-1`).
            q: axis ratio.
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
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ):
        z_l, thx0, thy0, q, phi, b, t, s = self.unpack(x)

        ax, ay = self.alpha(thx, thy, z_s, x)
        ax, ay = derotate(ax, ay, -phi)
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        return (thx * ax + thy * ay) / (2 - t)

    def kappa(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ):
        z_l, thx0, thy0, q, phi, b, t, s = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = (q**2 * (thx**2 + s**2) + thy**2).sqrt()
        return (2 - t) / 2 * (b / psi) ** t
