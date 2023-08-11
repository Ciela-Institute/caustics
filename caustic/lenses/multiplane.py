from operator import itemgetter
from typing import Any, Optional

import torch
from torch import Tensor

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
        zero = torch.tensor(0.0, dtype=z_s.dtype, device=z_s.device)
        # argsort redshifts
        z_ls = self.get_z_ls(params)
        idxs = [i for i, _ in sorted(enumerate(z_ls), key=itemgetter(1))]
        D_0_s = self.cosmology.comoving_distance(z_s, params)
        X_im1 = 0.0
        Y_im1 = 0.0
        X_i = self.cosmology.comoving_distance(z_ls[0], params) * x 
        Y_i = self.cosmology.comoving_distance(z_ls[0], params) * y 
        X_ip1 = None
        Y_ip1 = None

        for i in idxs:
            z_im1 = zero if i == 0 else z_ls[i - 1]
            z_i = z_ls[i]
            z_ip1 = z_s if i == len(z_ls) - 1 else z_ls[i + 1]

            D_im1_i = self.cosmology.comoving_distance_z1z2(z_im1, z_i, params)
            D_i_ip1 = self.cosmology.comoving_distance_z1z2(z_i, z_ip1, params)
            D_0_i = self.cosmology.comoving_distance(z_i, params)
            D_i_s = self.cosmology.comoving_distance_z1z2(z_i, z_s, params)
            D_ratio = D_0_s / D_i_s

            # Get alphas at next plane
            ax, ay = self.lenses[i].reduced_deflection_angle(X_i / D_0_i, Y_i / D_0_i, z_ip1, params)
            X_ip1 = (
                (D_i_ip1 / D_im1_i + 1) * X_i
                - (D_i_ip1 / D_im1_i) * X_im1
                - D_i_ip1 * D_ratio * ax
            )
            Y_ip1 = (
                (D_i_ip1 / D_im1_i + 1) * Y_i
                - (D_i_ip1 / D_im1_i) * Y_im1
                - D_i_ip1 * D_ratio * ay
            )

            # Advanced indices
            X_im1 = X_i
            Y_im1 = Y_i
            X_i = X_ip1
            Y_i = Y_ip1

        # Handle edge case of lenses = []
        if X_ip1 is None or Y_ip1 is None:
            return x, y
        else:
            return X_ip1 / D_0_s, Y_ip1 / D_0_s

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
