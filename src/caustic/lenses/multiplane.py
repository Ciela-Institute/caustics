from operator import itemgetter
from typing import Any, Optional

import torch
from torch import Tensor

from ..cosmology import Cosmology
from .base import ThickLens, ThinLens

__all__ = ("MultiplaneLens",)


class MultiplaneLens(ThickLens):
    def __init__(self, name: str, cosmology: Cosmology, lenses: list[ThinLens]):
        super().__init__(name, cosmology)
        self.lenses = lenses
        for lens in lenses:
            self.add_parametrized(lens)

    def get_z_ls(self, x: Optional[dict[str, Any]]) -> list[Tensor]:
        # Relies on z_l being the first element to be unpacked, which should always
        # be the case for a ThinLens
        return [lens.unpack(x)[0] for lens in self.lenses]

    def raytrace(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Reduced deflection angle [arcsec].

        Args:
            lenses: list of instances of thin lenses.
            z_ls: lens redshifts
            lens_arg_list: list of args to pass to each lens.
        """
        zero = torch.tensor(0.0, dtype=z_s.dtype, device=z_s.device)

        # argsort redshifts
        z_ls = self.get_z_ls(x)
        idxs = [i for i, _ in sorted(enumerate(z_ls), key=itemgetter(1))]
        D_0_s = self.cosmology.comoving_dist(z_s)
        X_im1 = 0.0
        Y_im1 = 0.0
        X_i = None
        Y_i = None
        X_ip1 = None
        Y_ip1 = None

        for i in idxs:
            z_im1 = zero if i == 0 else z_ls[i - 1]
            z_i = z_ls[i]
            z_ip1 = z_s if i == len(z_ls) - 1 else z_ls[i + 1]

            D_im1_i = self.cosmology.comoving_dist_z1z2(z_im1, z_i)
            D_i_ip1 = self.cosmology.comoving_dist_z1z2(z_i, z_ip1)
            D_0_i = self.cosmology.comoving_dist(z_i)
            D_i_s = self.cosmology.comoving_dist_z1z2(z_i, z_s)
            D_ratio = D_0_s / D_i_s

            # Collect current alphas
            X_i = D_0_i * thx if X_i is None else X_i
            Y_i = D_0_i * thy if Y_i is None else Y_i

            # Get alphas at next plane
            ax, ay = self.lenses[i].alpha(X_i / D_0_i, Y_i / D_0_i, z_ip1, x)
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
            return thx, thy
        else:
            return X_ip1 / D_0_s, Y_ip1 / D_0_s

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Reduced deflection angle [arcsec].
        """
        bx, by = self.raytrace(thx, thy, z_s, x)
        return thx - bx, thy - by

    def Sigma(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Projected mass density.

        Returns:
            [solMass / Mpc^2]
        """
        # TODO: rescale mass densities of each lens and sum
        raise NotImplementedError()

    def time_delay(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        # TODO: figure out how to compute this
        raise NotImplementedError()
