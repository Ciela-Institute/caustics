from typing import List

import torch

from ..base import Base
from ..constants import arcsec_to_rad, rad_to_arcsec
from .base import AbstractLens

# class PlaneLens(AbstractLens):
#     def __init__(self, lenses, z, cosmology, device):
#         super().__init__(cosmology, device)
#         self.lenses = lenses
#         self.z = z
#
#     def alpha(self, x, y, z=None, w=None, t=None):
#         return sum([lens.alpha(x, y, z, w, t) for lens in self.lenses])
#
#     def kappa(self, x, y, z=None, w=None, t=None):
#         return sum([lens.kappa(x, y, z, w, t) for lens in self.lenses])
#
#     def Psi(self, x, y, z=None, w=None, t=None):
#         return sum([lens.Psi(x, y, z, w, t) for lens in self.lenses])


class MultiplaneLens(Base):
    # TODO: is this an AbstractLens? It's different since it's not thin.
    def __init__(self, lenses: List[AbstractLens], z_ls, cosmology, device):
        """
        Args:
            z_red_ref: the reference redshift used to define the reduced deflection
                angles for the lenses.
        """
        super().__init__(cosmology, device)
        self.lenses = lenses
        self.z_ls = z_ls

    def _raytrace_to(self, thx_0, thy_0, ahx_sum, ahy_sum, ahx_chi_sum, ahy_chi_sum, z):
        chi = self.cosmology.comoving_dist(z)
        d_l = chi / (1 + z)  # angular diameter distance
        # Apply recursion relation
        xix = (chi * (thx_0 - ahx_sum) + ahx_chi_sum) / (1 + z) * arcsec_to_rad
        xiy = (chi * (thy_0 - ahy_sum) + ahy_chi_sum) / (1 + z) * arcsec_to_rad
        # Convert to angular coordinates to compute deflection field from lens i
        thx = xix / d_l * rad_to_arcsec
        thy = xiy / d_l * rad_to_arcsec
        return thx, thy

    def raytrace(self, thx, thy, z_ss):
        assert thx.shape == thy.shape
        thxy_final_shape = (*z_ss.shape, *thx.shape)
        z_ss = torch.atleast_1d(z_ss)

        thxs = torch.zeros_like(thx).repeat((len(z_ss), *(1 for _ in range(thx.ndim))))
        thys = torch.zeros_like(thy).repeat((len(z_ss), *(1 for _ in range(thy.ndim))))
        thx_0 = thx
        thy_0 = thy

        # Initial conditions
        ahx_sum = torch.zeros_like(thx)
        ahy_sum = torch.zeros_like(thy)
        ahx_chi_sum = torch.zeros_like(thx)
        ahy_chi_sum = torch.zeros_like(thy)
        idx_cur_source = 0
        z_s = z_ss[idx_cur_source]

        for lens, z_l in zip(self.lenses, self.z_ls):
            # Multiple sources can lie between lens planes
            while z_s <= z_l:
                # Ray-trace to current sources' redshift
                thx, thy = self._raytrace_to(thx_0, thy_0, ahx_sum, ahy_sum, ahx_chi_sum, ahy_chi_sum, z_s)
                thxs[idx_cur_source] = thx
                thys[idx_cur_source] = thy

                idx_cur_source += 1
                z_s = z_ss[idx_cur_source]

                if idx_cur_source == len(z_ss) - 1:
                    return thxs, thys

            # Ray-trace to current lens plane
            thx, thy = self._raytrace_to(thx_0, thy_0, ahx_sum, ahy_sum, ahx_chi_sum, ahy_chi_sum, z_l)
            ahx, ahy = lens.alpha_hat(thx, thy, z_l)

            # Update sums
            ahx_sum += ahx
            ahy_sum += ahy
            chi = self.cosmology.comoving_dist(z_l)
            ahx_chi_sum += chi * ahx
            ahy_chi_sum += chi * ahy

        # Multiple sources can lie beyond the last lens plane
        for idx in range(idx_cur_source, len(z_ss)):
            z_s = z_ss[idx]
            thx, thy = self._raytrace_to(thx_0, thy_0, ahx_sum, ahy_sum, ahx_chi_sum, ahy_chi_sum, z_s)
            thxs[idx_cur_source] = thx
            thys[idx_cur_source] = thy

        return thxs.reshape(thxy_final_shape), thys.reshape(thxy_final_shape)

    # def alpha(self, x, y, z=None, w=None, t=None):
    #     x_src = torch.zeros_like(x)
    #     y_src = torch.zeros_like(y)
    #     for lens, z in zip(self.lenses, self.z_ls):
    #         ...

    #     return sum([lens.alpha(x, y, z, w, t) for lens in self.lenses])

    def kappa(self, x, y, z=None, w=None, t=None):
        raise ValueError("undefined")

    def Psi(self, x, y, z=None, w=None, t=None):
        raise ValueError("undefined")
