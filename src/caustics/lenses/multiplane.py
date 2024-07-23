from typing import Optional, Union

import torch
from torch import Tensor

from ..constants import arcsec_to_rad, c_Mpc_s, seconds_to_days
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

            *Unit: unitless*
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
        return_shapiro_time_delay: bool = False,
        return_geometric_time_delay: bool = False,
        return_time_delay: bool = False,
        return_coordinates: bool = True,
        **kwargs,
    ) -> Union[tuple[Tensor], Tensor]:
        # Order resdhift
        z_ls = self.get_z_ls(params)
        planes = torch.argsort(torch.stack(z_ls))

        theta_x = x
        theta_y = y

        shapiro_tau = torch.zeros_like(x)
        geometric_tau = torch.zeros_like(x)

        for plane in planes:
            z_l = z_ls[plane]
            z_next = z_ls[plane + 1] if plane != len(planes) - 1 else z_s
            alpha_x, alpha_y = self.lenses[plane].reduced_deflection_angle(
                theta_x, theta_y, z_next, params
            )

            # Time delay
            if any(
                [
                    return_shapiro_time_delay,
                    return_geometric_time_delay,
                    return_time_delay,
                ]
            ):
                D_l = self.cosmology.angular_diameter_distance(z_l)
                D_next = self.cosmology.angular_diameter_distance(z_next)
                D_l_next = self.cosmology.angular_diameter_distance_z1z2(z_l, z_next)
                time_delay_distance = (1 + z_l) * D_l * D_next / D_l_next  # Mpc
                dt = time_delay_distance / c_Mpc_s * seconds_to_days  # days
                if return_shapiro_time_delay or return_time_delay:
                    phi = self.lenses[plane].potential(  # arcsec^2
                        theta_x, theta_y, z_next, params
                    )
                    shapiro_tau += -dt * phi * arcsec_to_rad**2
                if return_geometric_time_delay or return_time_delay:
                    geometric_tau += (
                        dt * 0.5 * (alpha_x**2 + alpha_y**2) * arcsec_to_rad**2
                    )

            # Recurrent lens equation
            theta_x = theta_x - alpha_x
            theta_y = theta_y - alpha_y

        _out = []
        if return_coordinates:
            _out.append(theta_x)
            _out.append(theta_y)
        if return_time_delay:
            _out.append(shapiro_tau + geometric_tau)
        elif return_shapiro_time_delay:
            _out.append(shapiro_tau)
        elif return_geometric_time_delay:
            _out.append(geometric_tau)
        out = tuple(_out)
        if len(out) == 1:
            return out[0]
        return out

    @unpack
    def raytrace(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        return_time_delay: bool = False,
        **kwargs,
    ) -> tuple[Tensor]:
        """Calculate the angular source positions corresponding to the
         observer positions x,y. See Margarita et al. 2013 for the
         formalism from the GLAMER -II code:
         https://ui.adsabs.harvard.edu/abs/2014MNRAS.445.1954P/abstract

         The primary equation used for multiplane is the following recursive formula

         .. math::
             \boldsymbol{\theta}^{\ell + 1} = \boldsymbol{\theta}^{\ell} - \boldsymbol{\alpha}^{(\ell)}(\boldsymbol{\theta}^{(\ell)})\, .

        for :math:`\ell \in \{0, L-1\}`, where :math:`L` is the number of planes.
        :math:`\boldsymbol{\theta}^{(\ell)}` are angular coordinates of the rays angular coordinate at each plane. The equation is initialized at
        :math:`\boldsymbol{\theta}^{(0)}`, which are the initial coordinates (x, y).
        :math:`\boldsymbol{\alpha}^{(\ell)}` are the reduced deflection angles of the lens at plane :math:`(\ell)`.

        This method returns the angular coordinates at the source plane :math:`\boldsymbol{\beta} = \boldsymbol{\theta}^{(L)}`.
        This implementation is equivalent to equation (2.14) and (2.15) of Fleury et al. (2022).

         Parameters
         ----------
         x: Tensor
             angular x-coordinates in the image plane.

             *Unit: arcsec*

         y: Tensor
             angular y-coordinates in the image plane.

             *Unit: arcsec*

         z_s: Tensor
             Redshifts of the sources.

             *Unit: unitless*

         params: Packed, optional
             Dynamic parameter container.

         Returns
         -------
         x_component: Tensor
             Reduced deflection angle in the x-direction.

             *Unit: arcsec*

         y_component: Tensor
             Reduced deflection angle in the y-direction.

             *Unit: arcsec*

         References
         ----------
         1. Pierre Fleury, Julien Larena and Jean-Philippe Uzan. 2021. Line-of-sight effects in strong gravitational lensing. JCAP 08, 2021. DOI:https://doi.org/10.1088/1475-7516/2021/08/024
         2. Margarita Petkova, R. Benton Metcalf, and Carlo Giocoli. 2014. GLAMER II: multiple-plane lensing. MNRAS 445, 1954-1966. DOI:https://doi.org/10.1093/mnras/stu1860

        """  # noqa: E501
        return self._raytrace_helper(
            x,
            y,
            z_s,
            params,
            shapiro_time_delay=False,
            geometric_time_delay=False,
            return_time_delay=return_time_delay,
            return_coordinates=True,
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

            *Unit: arcsec*

        y: Tensor
            y-coordinates in the lens plane.

            *Unit: arcsec*

        z_s: Tensor
            Redshifts of the sources.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        Tensor
            Projected mass density.

            *Unit: Msun/Mpc^2*

        Raises
        -------
        NotImplementedError
            This method is not yet implemented.
        """
        # TODO: rescale mass densities of each lens and sum
        raise NotImplementedError()

    @unpack
    def shapiro_time_delay(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        return_coordinates: bool = False,
        **kwargs,
    ) -> Tensor:
        return self._raytrace_helper(
            x,
            y,
            z_s,
            params,
            return_shapiro_time_delay=True,
            return_coordinates=return_coordinates,
            **kwargs,
        )

    @unpack
    def geometric_time_delay(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        return_coordinates: bool = False,
        **kwargs,
    ) -> Tensor:
        return self._raytrace_helper(
            x,
            y,
            z_s,
            params,
            return_geometric_time_delay=True,
            return_coordinates=return_coordinates,
            **kwargs,
        )

    @unpack
    def time_delay(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        return_coordinates: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Our implementation is based on equation 6.22 in Petters et al. 2001.

            \tau(\boldsymbol{\theta}^{(0)}) = \sum_{\ell=0}^{L-1} \frac{D_{\Delta t}^{(\ell)}}{c}
                \left[
                    \frac{1}{2} \lVert\boldsymbol{\alpha}^{(\ell)}(\boldsymbol{\theta}^{(\ell)}) \rVert^2
                    - \phi^{(\ell)}(\boldsymbol{\theta}^{(\ell)})
                \right] \\
            D_{\Delta t}^{(\ell)} = (1 + z_\ell) \frac{D_\ell D_{\ell + 1}}{D_{\ell, \ell + 1}} \\

        where :math:`\boldsymbol{\alpha}^{(\ell)}` is the deflection angle at the :math:`\ell`-th lens plane,
        :math:`\phi^{(\ell)}` is the lensing potential at the :math:`\ell`-th lens plane,
        :math:`D_\ell` is the angular diameter distance to the :math:`\ell`-th lens plane,
        :math:`D_{\ell, \ell+1}` is the angular diameter distance between the :math:`\ell`-th and the next lens plane,

        Parameters
        ----------
        x: Tensor
            x-coordinates in the image plane.

            *Unit: arcsec*

        y: Tensor
            y-coordinates in the image plane.

            *Unit: arcsec*

        z_s: Tensor
            Redshifts of the source.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        shapiro_time_delay: bool
            Whether to include the Shapiro time delay component.

        geometric_time_delay: bool
            Whether to include the geometric time delay component.

        Returns
        -------
        Tensor
            Time delay caused by the lensing.

            *Unit: days*

        References
        ----------
        1. Pierre Fleury, Julien Larena and Jean-Philippe Uzan. 2021. Line-of-sight effects in strong gravitational lensing. JCAP 08, 2021. DOI:https://doi.org/10.1088/1475-7516/2021/08/024
        2. Petters A. O., Levine H., Wambsganss J., 2001, Singularity Theory and Gravitational Lensing. Birkhauser, Boston
        3. McCully et al. 2014, A new hybrid framework to efficiently model lines of sight to gravitational lenses

        """
        return self._raytrace_helper(
            x,
            y,
            z_s,
            params,
            return_coordinates=return_coordinates,
            return_time_delay=True,
            **kwargs,
        )
