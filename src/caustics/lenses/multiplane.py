from operator import itemgetter
from typing import Optional

import torch
from torch import Tensor

from ..constants import arcsec_to_rad, rad_to_arcsec
from ..cosmology import Cosmology
from .base import ThickLens, ThinLens
from ..parametrized import unpack
from ..packed import Packed

__all__ = ("Multiplane",)


class Multiplane(ThickLens):
    """
    Class for handling gravitational lensing with multiple lens planes.

    Attributes
    ----------
    lenses (list[ThinLens])
        List of thin lenses.

    Parameters
    ----------
    name: string
        Name of the lens.
    cosmology: Cosmology
        Cosmological parameters used for calculations.
    lenses: list[ThinLens]
        List of thin lenses.
    """

    def __init__(self, cosmology: Cosmology, lenses: list[ThinLens], name: str = None):
        super().__init__(cosmology, name=name)
        self.lenses = lenses
        for lens in lenses:
            self.add_parametrized(lens)

    @unpack(0)
    def get_z_ls(
        self, *args, params: Optional["Packed"] = None, **kwargs
    ) -> list[Tensor]:
        """
        Get the redshifts of each lens in the multiplane.

        Parameters
        ----------
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        --------
        List[Tensor]
            Redshifts of the lenses.
        """
        # Relies on z_l being the first element to be unpacked, which should always
        # be the case for a ThinLens
        return [lens.unpack(params)[0] for lens in self.lenses]

    @unpack(3)
    def raytrace(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Calculate the angular source positions corresponding to the
        observer positions x,y. See Margarita et al. 2013 for the
        formalism from the GLAMER -II code:
        https://ui.adsabs.harvard.edu/abs/2014MNRAS.445.1954P/abstract

        The primary equation used here is equation 18. With a slight correction it reads:

        .. math::

          \vec{x}^{i+1} = \vec{x}^i + D_{i+1,i}\left[\vec{\theta} - \sum_{j=1}^{i}\bf{\alpha}^j(\vec{x}^j)\right]

        As an initialization we set the physical positions at the first lensing plane to be :math:`\vec{\theta}D_{1,0}` which is just propagation through regular space to the first plane.
        Note that :math:`\vec{\alpha}` is a physical deflection angle. The equation above converts straightforwardly into a recursion formula:

        .. math::

          \vec{x}^{i+1} = \vec{x}^i + D_{i+1,i}\vec{\theta}^{i}
          \vec{\theta}^{i+1} = \vec{\theta}^{i} -  \alpha^i(\vec{x}^{i+1})

        Here we set as initialization :math:`\vec{\theta}^0 = theta` the observation angular coordinates and :math:`\vec{x}^0 = 0` the initial physical coordinates (i.e. the observation rays come from a point at the observer).
        The indexing of :math:`\vec{x}^i` and :math:`\vec{\theta}^i` indicates the properties at the plane :math:`i`,
        and 0 means the observer, 1 is the first lensing plane (infinitesimally after the plane since the deflection has been applied),
        and so on. Note that in the actual implementation we start at :math:`\vec{x}^1` and :math:`\vec{\theta}^0`
        and begin at the second step in the recursion formula.

        Parameters
        ----------
        x: Tensor
            angular x-coordinates from the observer perspective.
        y: Tensor
            angular y-coordinates from the observer perspective.
        z_s: Tensor
            Redshifts of the sources.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        (Tensor, Tensor)
            The reduced deflection angle.

        """  # noqa: E501
        return self.raytrace_z1z2(x, y, torch.zeros_like(z_s), z_s, params)

    @unpack(4)
    def raytrace_z1z2(
        self,
        x: Tensor,
        y: Tensor,
        z_start: Tensor,
        z_end: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Method to do multiplane ray tracing from arbitrary start/end redshift.
        """

        # Collect lens redshifts and ensure proper order
        z_ls = self.get_z_ls(params)
        lens_planes = [i for i, _ in sorted(enumerate(z_ls), key=itemgetter(1))]

        # Compute physical position on first lens plane
        D = self.cosmology.transverse_comoving_distance_z1z2(
            z_start, z_ls[lens_planes[0]], params
        )
        X, Y = x * arcsec_to_rad * D, y * arcsec_to_rad * D

        # Initial angles are observation angles
        # (negative needed because of negative in propagation term)
        theta_x, theta_y = x, y

        for i in lens_planes:
            # Compute deflection angle at current ray positions
            D_l = self.cosmology.transverse_comoving_distance_z1z2(
                z_start, z_ls[i], params
            )
            alpha_x, alpha_y = self.lenses[i].physical_deflection_angle(
                X * rad_to_arcsec / D_l,
                Y * rad_to_arcsec / D_l,
                z_end,
                params,
            )

            # Update angle of rays after passing through lens (sum in eq 18)
            theta_x = theta_x - alpha_x
            theta_y = theta_y - alpha_y

            # Propagate rays to next plane (basically eq 18)
            z_next = z_ls[i + 1] if i != lens_planes[-1] else z_end
            D = self.cosmology.transverse_comoving_distance_z1z2(
                z_ls[i], z_next, params
            )
            X = X + D * theta_x * arcsec_to_rad
            Y = Y + D * theta_y * arcsec_to_rad

        # Convert from physical position to angular position on the source plane
        D_end = self.cosmology.transverse_comoving_distance_z1z2(z_start, z_end, params)
        return X * rad_to_arcsec / D_end, Y * rad_to_arcsec / D_end

    @unpack(3)
    def effective_reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        bx, by = self.raytrace(x, y, z_s, params)
        return x - bx, y - by

    @unpack(3)
    def surface_density(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the projected mass density.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.
        y: Tensor
            y-coordinates in the lens plane.
        z_s: Tensor
            Redshifts of the sources.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            Projected mass density [solMass / Mpc^2].

        Raises
        -------
        NotImplementedError
            This method is not yet implemented.
        """
        # TODO: rescale mass densities of each lens and sum
        raise NotImplementedError()

    @unpack(3)
    def time_delay(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the time delay of light caused by the lensing.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the lens plane.
        y: Tensor
            y-coordinates in the lens plane.
        z_s: Tensor
            Redshifts of the sources.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        Tensor
            Time delay caused by the lensing.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        # TODO: figure out how to compute this
        raise NotImplementedError()
