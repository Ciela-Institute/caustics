from operator import itemgetter
from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from .base import ThickLens, ThinLens
from .pixelated_convergence import PixelatedConvergence

from ..constants import *

__all__ = ("Multiplane_ls",)


class Multiplane_ls(ThickLens):
    """
    Class for handling gravitational lensing with multiple lens planes.

    Attributes:
        lenses (list[ThinLens]): List of thin lenses.

    Args:
        name (str): Name of the lens.
        cosmology (Cosmology): Cosmological parameters used for calculations.
        lenses (list[ThinLens]): List of thin lenses.
    """
    def __init__(self, name: str, cosmology: Cosmology, lenses: list[ThinLens]):
        super().__init__(name, cosmology)
        self.lenses = lenses
        for lens in lenses:
            self.add_parametrized(lens)

    def get_z_ls(self, params: Optional["Packed"]) -> list[Tensor]:
        """
        Get the redshifts of each lens in the multiplane.

        Args:
            params (Packed, optional): Dynamic parameter container.

        Returns:
            List[Tensor]: Redshifts of the lenses.
        """
        # Relies on z_l being the first element to be unpacked, which should always
        # be the case for a ThinLens
        return torch.Tensor([lens.unpack(params)[0] for lens in self.lenses])

    def raytrace(self, th_x: Tensor, th_y: Tensor, z_s: Tensor, params: Optional["Packed"] = None) -> tuple[Tensor, Tensor]:
        x = torch.zeros_like(th_x, dtype=float)
        y = torch.zeros_like(th_y, dtype=float)
        alpha_x = torch.Tensor(th_x)
        alpha_y = torch.Tensor(th_y)
        x, y, _, _ = self.raytrace_z1z2(x, y, alpha_x, alpha_y, z_start=0,
                                        z_stop=z_s, z_s=z_s, params=params)
        return x, y

    def raytrace_z1z2(self, x: Tensor, y: Tensor, alpha_x: Tensor, alpha_y: Tensor, z_start: Tensor, z_stop: Tensor, z_s: Tensor, include_z_start = False, params: Optional["Packed"] = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Calculate the angular source positions corresponding to the
        observer positions x,y. See Margarita et al. 2013 for the
        formalism from the GLAMER -II code:
        https://ui.adsabs.harvard.edu/abs/2014MNRAS.445.1954P/abstract

        Args:
            x (Tensor): angular x-coordinates from the observer perspective [arcsec]
            y (Tensor): angular y-coordinates from the observer perspective [arcsec]
            alpha_x (Tensor): ray angle at z_start [arcsec]
            alpha_y (Tensor): ray angle at z_start [arcsec]
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            tuple[Tensor, Tensor]: The reduced deflection angle.

        """
        zero = torch.tensor(0.0, dtype=z_s.dtype, device=z_s.device)
        # argsort redshifts
        z_ls = self.get_z_ls(params)
        sorted_redshift_idx = [i for i, _ in sorted(enumerate(z_ls), key=itemgetter(1))]

        z_before = 0
        T_ij_list = []
        T_z_list = []
        z_s_arr = torch.ones_like(z_ls) * z_s
        reduced2physical_factor = self.cosmology.comoving_distance(z_s) / self.cosmology.comoving_distance_z1z2(z_ls, z_s_arr)

        for idx in sorted_redshift_idx:
            z_l = z_ls[idx]
            if z_before == z_l:
                delta_T = 0
            else:
                T_z = self.cosmology.transverse_comoving_distance(z_l)
                delta_T = self.cosmology.transverse_comoving_distance_z1z2(z_before, z_l)
            T_z_list.append(T_z)
            T_ij_list.append(delta_T)
            z_before = z_l

        # Units
        T_z = self.cosmology.transverse_comoving_distance(z_start)
        x, y = self._arcsec2Mpc(x, y, T_z)

        z_l_last = z_start
        first_deflector = True
        for i, idx in enumerate(sorted_redshift_idx):
            z_l = z_ls[idx]
            z_lp1 = z_s if i == len(z_ls) - 1 else z_ls[sorted_redshift_idx[i+1]]

            if self._start_condition(include_z_start, z_l, z_start) and z_l <= z_stop:
                if first_deflector:
                    if z_start == 0:
                        delta_T = T_ij_list[0]
                    else:
                        delta_T = self.cosmology.transverse_comoving_distance_z1z2(z_start, z_l)
                    first_deflector = False
                else:
                    delta_T = T_ij_list[i]
                x = x + alpha_x * arcsec_to_rad * delta_T
                y = y + alpha_y * arcsec_to_rad * delta_T

                x, y = self._Mpc2arcsec(x, y, T_z_list[i])
                # alpha_x_red, alpha_y_red = self.lenses[idx].reduced_deflection_angle(x, y, z_lp1, params)     # TODO: check if output in as or rad & if should be z_s or z_lp1
                alpha_x_red, alpha_y_red = self.lenses[idx].reduced_deflection_angle(x, y, z_s, params)     # TODO: check if output in as or rad & if should be z_s or z_lp1
                alpha_x_phys = reduced2physical_factor[idx] * alpha_x_red
                alpha_y_phys = reduced2physical_factor[idx] * alpha_y_red

                alpha_x = alpha_x - alpha_x_phys
                alpha_y = alpha_y - alpha_y_phys
                z_l_last = z_l
                x, y = self._arcsec2Mpc(x, y, T_z_list[i])

        if z_l_last == z_stop:
            delta_T = 0
        else:
            delta_T = self.cosmology.transverse_comoving_distance_z1z2(z_l_last, z_stop)
        x = x + alpha_x * arcsec_to_rad * delta_T
        y = y + alpha_y * arcsec_to_rad * delta_T

        T_stop = self.cosmology.transverse_comoving_distance(z_stop)
        x, y = self._Mpc2arcsec(x, y, T_stop)
        return x, y, alpha_x, alpha_y


    def effective_reduced_deflection_angle(
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> tuple[Tensor, Tensor]:
        bx, by = self.raytrace(x, y, z_s, params)
        return x - bx, y - by

    def surface_density(
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Calculate the projected mass density.

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: Projected mass density [solMass / Mpc^2].

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        # TODO: rescale mass densities of each lens and sum
        raise NotImplementedError()

    def time_delay(
        self, x: Tensor, y: Tensor, z_s: Tensor, params: Optional["Packed"] = None
    ) -> Tensor:
        """
        Compute the time delay of light caused by the lensing.

        Args:
            x (Tensor): x-coordinates in the lens plane.
            y (Tensor): y-coordinates in the lens plane.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            Tensor: Time delay caused by the lensing.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        # TODO: figure out how to compute this
        raise NotImplementedError()


    @staticmethod
    def _arcsec2Mpc(x, y, T_z):
        return x * arcsec_to_rad * T_z, y * arcsec_to_rad * T_z

    @staticmethod
    def _Mpc2arcsec(x, y, T_z):
        return x / T_z * rad_to_arcsec, y / T_z * rad_to_arcsec

    @staticmethod
    def _start_condition(inclusive, z_l, z_start):
        if inclusive:
            return z_l >= z_start
        else:
            return z_l > z_start
