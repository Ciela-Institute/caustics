# from math import pi

import torch

# from ..constants import arcsec_to_rad, c_km_s, rad_to_arcsec
from ..utils import to_elliptical  # translate_rotate,
from ..utils import flip_axis_ratio, transform_scalar_fn, transform_vector_fn
from .base import AbstractLens

# from torch import Tensor


class SIE(AbstractLens):
    def __init__(self, thx0, thy0, th_ein, q, phi, z_l, cosmology=None, device=None):
        super().__init__(z_l, cosmology, device)
        self.thx0 = thx0
        self.thy0 = thy0
        self.th_ein = th_ein
        self.q, self.phi = flip_axis_ratio(q, phi)

    @transform_vector_fn
    def alpha(self, thx, thy, z_s):
        # Transform to elliptical angular coordinates aligned with semimajor/minor
        # axes of ellipse
        ex, ey = to_elliptical(thx, thy, self.q)
        phi_e = torch.atan2(ey, ex)
        a_complex = (
            2
            * self.th_ein
            * self.q.sqrt()
            * ((1j * phi_e).exp() * ((1 - self.q) / (1 + self.q)).sqrt()).atan()
            / (1 - self.q**2).sqrt()
        )
        return a_complex.real, a_complex.imag

    @transform_scalar_fn
    def Psi(self, thx, thy, z_s):
        ax, ay = self.alpha(thx, thy, z_s)
        return thx * ax + thy * ay

    @transform_scalar_fn
    def kappa(self, thx, thy, z_s):
        ex, ey = to_elliptical(thx, thy, self.q)
        r_e = (ex**2 + ey**2).sqrt()
        return self.q.sqrt() * self.th_ein / r_e / 2


# class SIEOld(AbstractLens):
#     def __init__(
#         self,
#         thx0,
#         thy0,
#         th_ein,
#         q,
#         phi,
#         z_l_ref=None,
#         z_s_ref=None,
#         cosmology=None,
#         device=None,
#     ):
#         """
#         Args:
#             z_l_ref, z_s_ref: reference redshifts at which this lens' Einstein radius
#                 parameter is equal to the provided value.
#         """
#         super().__init__(cosmology, device)
#         self.thx0 = torch.as_tensor(thx0, dtype=torch.float32, device=device)
#         self.thy0 = torch.as_tensor(thy0, dtype=torch.float32, device=device)
#         self.th_ein = torch.as_tensor(th_ein, dtype=torch.float32, device=device)
#         self.q = torch.as_tensor(q, dtype=torch.float32, device=device)
#         self.phi = torch.as_tensor(phi, dtype=torch.float32, device=device)
#
#         self.z_l_ref = z_l_ref
#         self.z_s_ref = z_s_ref
#         self.d_s_ref = self.cosmology.angular_diameter_dist(z_s_ref)
#         self.d_l_ref = self.cosmology.angular_diameter_dist(z_l_ref)
#         self.d_ls_ref = self.cosmology.angular_diameter_dist_z1z2(z_l_ref, z_s_ref)
#
#     def xi_0(self, z_l, z_s):
#         return self.d_l_ref * self.th_ein * arcsec_to_rad  # Mpc
#
#     def kappa(self, thx, thy, z_l, z_s):
#         d_l = self.cosmology.angular_diameter_dist(z_l)
#         thx = thx - self.thx0
#         thy = thy - self.thy0
#         r = d_l * (thx**2 + thy**2).sqrt()
#         return 1 / 2 * self.xi_0(z_l, z_s) / r
#
#     def Sigma(self, thx, thy, z_l, z_s):
#         Sigma_cr = self.Sigma_cr(z_l, z_s)
#         return Sigma_cr * self.kappa(thx, thy, z_l, z_s)
#
#     def Psi(self, thx, thy, z_l, z_s):
#         ax, ay = self.alpha(thx, thy, z_l, z_s)
#         Psi = thx * ax + thy * ay
#         return Psi  # arcsec
#
#     def Psi_hat(self, thx, thy, z_l, z_s):
#         d_l = self.cosmology.angular_diameter_dist(z_l)
#         factor = (self.xi_0(z_l, z_s) / d_l) ** 2
#         return factor * self.Psi(thx, thy, z_l, z_s)  # arcsec
#
#     def alpha(self, thx, thy, z_l, z_s):
#         # Transform to elliptical angular coordinates aligned with semimajor/minor
#         # axes of ellipse
#         q, phi = flip_axis_ratio(self.q, self.phi)
#         thx, thy = translate_rotate(thx, thy, -phi, self.thx0, self.thy0)
#         ex, ey = to_elliptical(thx, thy, q)
#         phi_e = torch.atan2(ey, ex)
#
#         q_ratio = ((1 + q) / (1 - q)).sqrt()
#         a_complex = (
#             2
#             * self.th_ein
#             * q.sqrt()
#             / (1 + q)
#             * q_ratio
#             * ((1j * phi_e).exp() / q_ratio).atan()
#         )
#         ax = a_complex.real
#         ay = a_complex.imag
#
#         # Rotate back to coordinate axes
#         return translate_rotate(ax, ay, phi)  # arcsec
#
#     def alpha_hat(self, thx, thy, z_l, z_s):
#         d_l = self.cosmology.angular_diameter_dist(z_l)
#         d_s = self.cosmology.angular_diameter_dist(z_s)
#         d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s)
#         ax, ay = self.alpha(thx, thy, z_l, z_s)
#         factor = self.xi_0(z_l, z_s) * d_s / d_l / d_ls
#         return factor * ax, factor * ay  # arcsec
