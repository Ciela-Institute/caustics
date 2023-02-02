import torch
from scipy.special import hyp2f1

from ..utils import derotate, translate_rotate
from .base import AbstractThinLens


class EPL(AbstractThinLens):
    """
    Elliptical power law (aka singular power-law ellipsoid) profile.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        n_iter=18,
    ):
        """
        Args:
            n_iter: number of iterations for approximation of hypergeometric function.
        """
        super().__init__(device, dtype)
        self.n_iter = n_iter

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, t, s=None):
        """
        Args:
            b: scale length.
            t: power law slope (`gamma-1`).
            s: core radius.
        """
        s = torch.tensor(0.0, device=self.device, dtype=thx.dtype) if s is None else s
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

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, t, s=None):
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        # Don't translate or rotate in call to alpha
        thx0_origin = torch.tensor(0.0, dtype=thx0.dtype, device=thx0.device)
        thy0_origin = torch.tensor(0.0, dtype=thy0.dtype, device=thy0.device)
        phi_0 = torch.tensor(0.0, dtype=phi.dtype, device=phi.device)
        ax, ay = self.alpha(
            thx, thy, z_l, z_s, cosmology, thx0_origin, thy0_origin, q, phi_0, b, t, s
        )
        return (thx * ax + thy * ay) / (2 - t)

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, t, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = (q**2 * (thx**2 + s**2) + thy**2).sqrt()
        return (2 - t) / 2 * (b / psi) ** t
