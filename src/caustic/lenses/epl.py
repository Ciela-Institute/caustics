import torch

from ..utils import derotate, translate_rotate
from .base import AbstractThinLens


class EPL(AbstractThinLens):
    """
    Elliptical power law (aka singular power-law ellipsoid) profile.
    """

    def __init__(self, device=torch.device("cpu")):
        super().__init__(device)

    def _get_psi(self, x, y, q, s):
        return (q**2 * (x**2 + s**2) + y**2).sqrt()

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, t, s=None):
        """
        Args:
            b: scale length.
            t: power law slope.  (I believe for us t = gamma-1?)
            s: core radius.
        """
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = self._get_psi(thx, thy, q, s)
        
        # follow Tessore et al 2015 (eq. 5)
        z = q * thx + thy * 1j
        r = torch.abs(z)
        
        # Tessore et al 2015 (eq. 23)
        alpha_c = 2. / (1. + q) * (b / r)**t * self._r_omega(z, t, q, n=25) #not sure if we want the n higher/lower or costumizable
        
        alpha_real = torch.nan_to_num(alpha_c.real, posinf=(1e12), neginf=(-1e12))
        alpha_imag = torch.nan_to_num(alpha_c.imag, posinf=(1e12), neginf=(-1e12))
        return derotate(alpha_real, alpha_imag, psi)

    def _r_omega(self, z, t, q, n):
        '''
        z = R * e^(i * phi):    position vector in the lens plane (complex torch tensor)
        t = gamma - 1:          mass slope of the profile
        q:                      axis ratio
        n:                      number of iterations
        This iterative implementation of the hypergeometric function `hyp2f1` in `scipy.special`
        Note: the value returned includes an extra factor R multiplying eq. (23) for omega(phi).
        '''
        # constants
        f = (1. - q)/(1. + q)
        phi = z/torch.conj(z)
        
        # first term in series
        omega_i = z 
        part_sum = omega_i
        
        for i in range(1, n):
            factor = (2. * i - (2. - t)) / (2. * i + (2. - t))
            omega_i = -f * factor * phi * omega_i
            part_sum += omega_i
        return part_sum

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, t, s=None):
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)

        # Only transform coordinates once: pass thx0=0, thy=0, phi=None to alpha
        ax, ay = self.alpha(thx, thy, z_l, z_s, cosmology, 0.0, 0.0, q, None, b, s)
        return (thx * ax + thy * ay) / (2 - t)

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, q, phi, b, t, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = self._get_psi(thx, thy, q, s)
        return (2 - t) / 2 * (b / psi) ** t
