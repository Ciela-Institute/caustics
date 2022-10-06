import warnings
from math import pi

import torch
from torch import Tensor

from ..constants import arcsec_to_rad, c_km_s, rad_to_arcsec
from ..utils import flip_axis_ratio, to_elliptical, translate_rotate
from .base import AbstractLens


class SIE(AbstractLens):
    def __init__(
        self,
        thx0,
        thy0,
        th_ein,
        q,
        phi,
        z_l_ref=None,
        z_s_ref=None,
        cosmology=None,
        device=None,
    ):
        """
        Args:
            z_l_ref, z_s_ref: reference redshifts at which this lens' Einstein radius
                parameter is equal to the provided value.
        """
        super().__init__(cosmology, device)
        self.thx0 = torch.as_tensor(thx0, dtype=torch.float32, device=device)
        self.thy0 = torch.as_tensor(thy0, dtype=torch.float32, device=device)
        self.th_ein = torch.as_tensor(th_ein, dtype=torch.float32, device=device)
        self.q = torch.as_tensor(q, dtype=torch.float32, device=device)
        self.phi = torch.as_tensor(phi, dtype=torch.float32, device=device)

        self.z_l_ref = z_l_ref
        self.z_s_ref = z_s_ref
        self.d_s_ref = self.cosmology.angular_diameter_dist(z_s_ref)
        self.d_ls_ref = self.cosmology.angular_diameter_dist_z1z2(z_l_ref, z_s_ref)

    def Sigma(self, xix, xiy, z_l):
        ...

    def Psi_hat(self, thx, thy, d_l, d_s, d_ls):
        ...

    def alpha_hat(self, thx, thy, z_l):
        # Transform to elliptical angular coordinates aligned with semimajor/minor
        # axes of ellipse
        q, phi = flip_axis_ratio(self.q, self.phi)
        thx, thy = translate_rotate(thx, thy, -phi, self.thx0, self.thy0)
        ex, ey = to_elliptical(thx, thy, q)
        phi_e = torch.atan2(ey, ex)

        # Ensures th_ein scales properly as the z_l is changed
        # TODO: double-check
        th_ein_scaled = self.th_ein * self.d_s_ref / self.d_ls_ref

        q_ratio = ((1 + q) / (1 - q)).sqrt()
        a_complex = (
            2
            * th_ein_scaled
            * q.sqrt()
            / (1 + q)
            * q_ratio
            * ((1j * phi_e).exp() / q_ratio).atan()
        )
        ax = a_complex.real
        ay = a_complex.imag

        # Rotate back to coordinate axes
        return translate_rotate(ax, ay, phi)
