from operator import itemgetter
from typing import Optional

import torch
from torch import Tensor

from ..constants import arcsec_to_rad, rad_to_arcsec, c_Mpc_s
from .base import ThickLens, NameType, CosmologyType, LensesType
from ..parametrized import unpack
from ..packed import Packed

__all__ = ("Multiplane",)


class Multiplane(ThickLens):
    """
    Class for handling gravitational lensing with multiple lens planes.

    Attributes
    ----------
    lenses list of ThinLens
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

    def __init__(
        self, cosmology: CosmologyType, lenses: LensesType, name: NameType = None
    ):
        super().__init__(cosmology, name=name)
        self.lenses = lenses
        for lens in lenses:
            self.add_parametrized(lens)

    @unpack
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

    @unpack
    def _raytrace_helper(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        shapiro_time_delay: bool = True,
        geometric_time_delay: bool = True,
        ray_coords: bool = True,
        **kwargs,
    ):
        # Collect lens redshifts and ensure proper order
        z_ls = self.get_z_ls(params)
        lens_planes = [i for i, _ in sorted(enumerate(z_ls), key=itemgetter(1))]
        D_s = self.cosmology.transverse_comoving_distance(z_s, params)

        # Compute physical position on first lens plane
        D = self.cosmology.transverse_comoving_distance(z_ls[lens_planes[0]], params)
        X, Y = x * arcsec_to_rad * D, y * arcsec_to_rad * D  # fmt: skip

        # Initial angles are observation angles
        theta_x, theta_y = x, y

        # Store the time delays
        TD = torch.zeros_like(x)

        for i in lens_planes:
            z_next = z_ls[i + 1] if i != lens_planes[-1] else z_s
            # Compute deflection angle at current ray positions
            D_l = self.cosmology.transverse_comoving_distance(z_ls[i], params)
            D = self.cosmology.transverse_comoving_distance_z1z2(
                z_ls[i], z_next, params
            )
            D_is = self.cosmology.transverse_comoving_distance_z1z2(
                z_ls[i], z_s, params
            )
            D_next = self.cosmology.transverse_comoving_distance(z_next, params)
            alpha_x, alpha_y = self.lenses[i].physical_deflection_angle(
                X * rad_to_arcsec / D_l,
                Y * rad_to_arcsec / D_l,
                z_s,
                params,
            )

            # Update angle of rays after passing through lens (sum in eq 18)
            theta_x = theta_x - alpha_x
            theta_y = theta_y - alpha_y

            # Compute time delay
            tau_ij = (1 + z_ls[i]) * D_l * D_next / (D * c_Mpc_s)
            if shapiro_time_delay:
                beta_ij = D * D_s / (D_next * D_is)
                potential = self.lenses[i].potential(
                    X * rad_to_arcsec / D_l, Y * rad_to_arcsec / D_l, z_s, params
                )
                TD += (-tau_ij * beta_ij * arcsec_to_rad**2) * potential
            if geometric_time_delay:
                TD += (tau_ij * arcsec_to_rad**2 * 0.5) * (alpha_x**2 + alpha_y**2)

            # Propagate rays to next plane (basically eq 18)
            X = X + D * theta_x * arcsec_to_rad
            Y = Y + D * theta_y * arcsec_to_rad

        # Convert from physical position to angular position on the source plane
        D_end = self.cosmology.transverse_comoving_distance(z_s, params)
        if ray_coords and not (shapiro_time_delay or geometric_time_delay):
            return X * rad_to_arcsec / D_end, Y * rad_to_arcsec / D_end
        elif ray_coords and (shapiro_time_delay or geometric_time_delay):
            return X * rad_to_arcsec / D_end, Y * rad_to_arcsec / D_end, TD
        elif shapiro_time_delay or geometric_time_delay:
            return TD
        raise ValueError(
            "No return value specified. Must choose one or more of: ray_coords, shapiro_time_delay, or geometric_time_delay to be True."
        )

    @unpack
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
            angular x-coordinates in the image plane.
        y: Tensor
            angular y-coordinates in the image plane.
        z_s: Tensor
            Redshifts of the sources.
        params: (Packed, optional)
            Dynamic parameter container.

        Returns
        -------
        (Tensor, Tensor)
            The reduced deflection angle.

        References
        ----------
        1. Margarita Petkova, R. Benton Metcalf, and Carlo Giocoli. 2014. GLAMER II: multiple-plane lensing. MNRAS 445, 1954-1966. DOI:https://doi.org/10.1093/mnras/stu1860

        """  # noqa: E501
        return self._raytrace_helper(
            x,
            y,
            z_s,
            params,
            shapiro_time_delay=False,
            geometric_time_delay=False,
            ray_coords=True,
            **kwargs,
        )

    @unpack
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

    @unpack
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

    @unpack
    def time_delay(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        shapiro_time_delay: bool = True,
        geometric_time_delay: bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Compute the time delay of light caused by the lensing.
        This is based on equation 6.22 in Petters et al. 2001.
        For the time delay of a light path from the observer to the source, the following equation is used::

            \Delta t = \sum_{i=1}^{N-1} \tau_{i,i+1} \left[ \frac{1}{2} \left( \vec{\alpha}^i \right)^2 - \beta_{i,i+1} \psi^i \right] \\
            \tau_{i,j} = (1 + z_i) \frac{D_i D_{j}}{D_{i,j} c} \\
            \beta_{i,j} = \frac{D_{i,j} D_s}{D_{j} D_{i,s}} \\

        where :math:`\vec{\alpha}^i` is the deflection angle at the i-th lens plane,
        :math:`\psi^i` is the lensing potential at the i-th lens plane,
        :math:`D_i` is the comoving distance to the i-th lens plane,
        :math:`D_{i,j}` is the comoving distance between the i-th and j-th lens plane,
        :math:`D_s` is the comoving distance to the source,
        and :math:`D_{i,s}` is the comoving distance between the i-th lens plane and the source.

        This performs the same ray tracing as the :func:`raytrace` method, but computes the time delay along the way.

        Parameters
        ----------
        x: Tensor
            x-coordinates in the image plane.
        y: Tensor
            y-coordinates in the image plane.
        z_s: Tensor
            Redshifts of the source.
        params: (Packed, optional)
            Dynamic parameter container.
        shapiro_time_delay: bool
            Whether to include the Shapiro time delay component.
        geometric_time_delay: bool
            Whether to include the geometric time delay component.

        Returns
        -------
        Tensor
            Time delay caused by the lensing.

        References
        ----------
        1. Petters A. O., Levine H., Wambsganss J., 2001, Singularity Theory and Gravitational Lensing. Birkhauser, Boston
        2. McCully et al. 2014, A new hybrid framework to efficiently model lines of sight to gravitational lenses

        """
        return self._raytrace_helper(
            x,
            y,
            z_s,
            params,
            shapiro_time_delay=shapiro_time_delay,
            geometric_time_delay=geometric_time_delay,
            ray_coords=False,
            **kwargs,
        )
