from typing import Any, List, Tuple

import torch
from torch import Tensor

from .base import ThickLens, ThinLens

__all__ = ("MultiplaneLens",)


class MultiplaneLens(ThickLens):
    def raytrace(
        self,
        thx,
        thy,
        z_s,
        cosmology,
        lenses: List[ThinLens],
        z_ls: Tensor,
        lens_args: List[Tuple[Any, ...]],
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Reduced deflection angle [arcsec].

        Args:
            lenses: list of instances of thin lenses.
            z_ls: lens redshifts
            lens_arg_list: list of args to pass to each lens.
        """
        # Determine which plane is being processed
        i = kwargs.get("_depth", 0)
        # sort the lenses in redshift order
        z_ls_sorted, idxs = torch.sort(z_ls)
        # compute relevant cosmological distances
        D_im1_i = cosmology.comoving_dist_z1z2(
            0 if i == 0 else z_ls_sorted[i - 1], z_ls_sorted[i]
        )
        D_i_ip1 = cosmology.comoving_dist_z1z2(
            z_ls_sorted[i], z_ls_sorted[i + 1] if i + 1 < len(z_ls) else z_s
        )
        D_0_i = cosmology.comoving_dist_z1z2(0, z_ls_sorted[i])
        D_ratio = cosmology.comoving_dist_z1z2(0, z_s) / cosmology.comoving_dist_z1z2(
            z_ls_sorted[i], z_s
        )

        # Collect the current alphas
        X_im1 = kwargs.get("_X_im1", (0.0, 0.0))
        X_i = kwargs.get("_X_i", (D_0_i * thx, D_0_i * thy))

        # Compute the alphas at the next plane
        alphas = lenses[idxs[i]].alpha(
            X_i[0] / D_0_i,
            X_i[1] / D_0_i,
            z_ls_sorted[i],
            z_ls_sorted[i + 1] if i + 1 < len(z_ls) else z_s,
            cosmology,
            *lens_args[idxs[i]],
        )
        X_ip1 = (
            (D_i_ip1 / D_im1_i + 1) * X_i[0]
            - (D_i_ip1 / D_im1_i) * X_im1[0]
            - D_i_ip1 * D_ratio * alphas[0],
            (D_i_ip1 / D_im1_i + 1) * X_i[1]
            - (D_i_ip1 / D_im1_i) * X_im1[1]
            - D_i_ip1 * D_ratio * alphas[1],
        )

        # end recursion when last plane reached
        if (i + 1) == len(z_ls):
            D_0_s = cosmology.comoving_dist_z1z2(0, z_s)
            return X_ip1[0] / D_0_s, X_ip1[1] / D_0_s
        # continue recursion while more lensing planes are available
        return self.raytrace(
            thx,
            thy,
            z_s,
            cosmology,
            lenses,
            z_ls,
            lens_args,
            _depth=i + 1,
            _X_im1=X_i,
            _X_i=X_ip1,
        )

    def alpha(
        self,
        thx,
        thy,
        z_s,
        cosmology,
        lenses: List[ThinLens],
        z_ls: Tensor,
        lens_args: List[Tuple[Any, ...]],
    ) -> Tuple[Tensor, Tensor]:
        """
        Reduced deflection angle [arcsec].
        """
        bx, by = self.raytrace(thx, thy, z_s, cosmology, lenses, z_ls, lens_args)
        return thx - bx, thy - by

    def Sigma(
        self,
        thx,
        thy,
        z_s,
        cosmology,
        lenses: List[ThinLens],
        z_ls: Tensor,
        lens_args: List[Tuple[Any, ...]],
    ) -> Tensor:
        """
        Projected mass density.

        Returns:
            [solMass / Mpc^2]
        """
        # TODO: rescale mass densities of each lens and sum
        raise NotImplementedError()

    def time_delay(
        self,
        thx,
        thy,
        z_s,
        cosmology,
        lenses: List[ThinLens],
        z_ls: Tensor,
        lens_args: List[Tuple[Any, ...]],
    ):
        # TODO: figure out how to compute this
        raise NotImplementedError()
