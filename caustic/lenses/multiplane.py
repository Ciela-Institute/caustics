from operator import itemgetter
from typing import Any, Optional

import torch
from torch import Tensor

from ..constants import arcsec_to_rad, rad_to_arcsec
from ..cosmology import Cosmology
from .base import ThickLens, ThinLens
from ..parametrized import unpack

__all__ = ("Multiplane",)


class Multiplane(ThickLens):
    """
    Class for handling gravitational lensing with multiple lens planes.

    Attributes:
        lenses (list[ThinLens]): List of thin lenses.

    Args:
        name (str): Name of the lens.
        cosmology (Cosmology): Cosmological parameters used for calculations.
        lenses (list[ThinLens]): List of thin lenses.
    """
    def __init__(self, cosmology: Cosmology, lenses: list[ThinLens], name: str = None):
        super().__init__(cosmology, name=name)
        self.lenses = lenses
        for lens in lenses:
            self.add_parametrized(lens)

    @unpack(0)
    def get_z_ls(self, *args, params: Optional["Packed"] = None, **kwargs) -> list[Tensor]:
        """
        Get the redshifts of each lens in the multiplane.

        Args:
            params (Packed, optional): Dynamic parameter container.

        Returns:
            List[Tensor]: Redshifts of the lenses.
        """
        # Relies on z_l being the first element to be unpacked, which should always
        # be the case for a ThinLens
        return [lens.unpack(params)[0] for lens in self.lenses]

    @unpack(3)
    def raytrace(
            self, x: Tensor, y: Tensor, z_s: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """Calculate the angular source positions corresponding to the
        observer positions x,y. See Margarita et al. 2013 for the
        formalism from the GLAMER -II code:
        https://ui.adsabs.harvard.edu/abs/2014MNRAS.445.1954P/abstract

        Args:
            x (Tensor): angular x-coordinates from the observer perspective.
            y (Tensor): angular y-coordinates from the observer perspective.
            z_s (Tensor): Redshifts of the sources.
            params (Packed, optional): Dynamic parameter container.

        Returns:
            tuple[Tensor, Tensor]: The reduced deflection angle.

        """
        return self.raytrace_z1z2(x, y, torch.zeros_like(z_s), z_s, params)

    @unpack(4)
    def raytrace_z1z2(
            self, x: Tensor, y: Tensor, z_start: Tensor, z_end: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> tuple[Tensor, Tensor]:

        # Collect lens redshifts and ensure proper order
        z_ls = self.get_z_ls(params)
        lens_planes = [i for i, _ in sorted(enumerate(z_ls), key=itemgetter(1))]

        # Compute physical position on first lens plane
        D = self.cosmology.transverse_comoving_distance_z1z2(z_start, z_ls[lens_planes[0]], params)
        X, Y = x * arcsec_to_rad * D, y * arcsec_to_rad * D

        # Initial angles are observation angles (negative needed because of negative in propogation term)
        theta_x, theta_y = -x, -y
        
        for i in lens_planes:
            # Compute deflection angle at current ray positions
            D_l = self.cosmology.transverse_comoving_distance_z1z2(z_start, z_ls[i], params)
            alpha_x, alpha_y = self.lenses[i].physical_deflection_angle(
                X * rad_to_arcsec / D_l,
                Y * rad_to_arcsec / D_l,
                z_end,
                params,
            )

            # Update angle of rays after passing through lens
            theta_x = theta_x + alpha_x
            theta_y = theta_y + alpha_y

            # Propogate rays to next plane
            z_next = z_ls[i+1] if i != lens_planes[-1] else z_end
            D = self.cosmology.transverse_comoving_distance_z1z2(z_ls[i], z_next, params)
            X = X - theta_x * arcsec_to_rad * D
            Y = Y - theta_y * arcsec_to_rad * D

        # Convert from physical position to angular position on the source plane
        D_end = self.cosmology.transverse_comoving_distance_z1z2(z_start, z_end, params)
        return X * rad_to_arcsec / D_end, Y * rad_to_arcsec / D_end

    @unpack(3)
    def effective_reduced_deflection_angle(
            self, x: Tensor, y: Tensor, z_s: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        bx, by = self.raytrace(x, y, z_s, params)
        return x - bx, y - by

    @unpack(3)
    def surface_density(
            self, x: Tensor, y: Tensor, z_s: Tensor, *args, params: Optional["Packed"] = None, **kwargs
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

    @unpack(3)
    def time_delay(
            self, x: Tensor, y: Tensor, z_s: Tensor, *args, params: Optional["Packed"] = None, **kwargs
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
