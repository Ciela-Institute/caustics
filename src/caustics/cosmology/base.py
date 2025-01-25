# mypy: disable-error-code="operator"
from abc import abstractmethod
from math import pi
from typing import Optional, Annotated

from torch import Tensor
from caskade import Module, forward

from ..constants import G_over_c2

NameType = Annotated[Optional[str], "Name of the cosmology"]


class Cosmology(Module):
    """
    Abstract base class for cosmological models.

    This class provides an interface for cosmological computations used in lensing
    such as comoving distance and critical surface density.

    Distance

        *Unit: Mpc*

    Mass

        *Unit: Msun*

    Attributes
    ----------
    name: str
        Name of the cosmological model.
    """

    def __init__(self, name: NameType = None):
        """
        Initialize the Cosmology.

        Parameters
        ----------
        name: str
            Name of the cosmological model.
        """
        super().__init__(name)

    @abstractmethod
    @forward
    def critical_density(self, z: Tensor) -> Tensor:
        """
        Compute the critical density at redshift z.

        Parameters
        ----------
        z: Tensor
            The redshift.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The critical density at each redshift.

            *Unit: Msun/Mpc^3*

        """
        ...

    @abstractmethod
    @forward
    def comoving_distance(self, z: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute the comoving distance to redshift z.

        Parameters
        ----------
        z: Tensor
            The redshift.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The comoving distance to each redshift.

            *Unit: Mpc*

        """
        ...

    @abstractmethod
    @forward
    def transverse_comoving_distance(self, z: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute the transverse comoving distance to redshift z (Mpc).

        Parameters
        ----------
        z: Tensor
            The redshift.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The transverse comoving distance to each redshift in Mpc.

            *Unit: Mpc*

        """
        ...

    @forward
    def comoving_distance_z1z2(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Compute the comoving distance between two redshifts.

        Parameters
        ----------
        z1: Tensor
            The starting redshift.

            *Unit: unitless*

        z2: Tensor
            The ending redshift.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The comoving distance between each pair of redshifts.

            *Unit: Mpc*

        """
        return self.comoving_distance(z2) - self.comoving_distance(z1)

    @forward
    def transverse_comoving_distance_z1z2(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Compute the transverse comoving distance between two redshifts (Mpc).

        Parameters
        ----------
        z1: Tensor
            The starting redshift.

            *Unit: unitless*

        z2: Tensor
            The ending redshift.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The transverse comoving distance between each pair of redshifts in Mpc.

            *Unit: Mpc*

        """
        return self.transverse_comoving_distance(
            z2
        ) - self.transverse_comoving_distance(z1)

    @forward
    def angular_diameter_distance(self, z: Tensor) -> Tensor:
        """
        Compute the angular diameter distance to redshift z.

        Parameters
        -----------
        z: Tensor
            The redshift.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The angular diameter distance to each redshift.

            *Unit: Mpc*

        """
        return self.comoving_distance(z) / (1 + z)

    @forward
    def angular_diameter_distance_z1z2(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Compute the angular diameter distance between two redshifts.

        Parameters
        ----------
        z1: Tensor
            The starting redshift.

            *Unit: unitless*

        z2: Tensor
            The ending redshift.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The angular diameter distance between each pair of redshifts.

            *Unit: Mpc*

        """
        return self.comoving_distance_z1z2(z1, z2) / (1 + z2)

    @forward
    def time_delay_distance(
        self,
        z_l: Tensor,
        z_s: Tensor,
    ) -> Tensor:
        """
        Compute the time delay distance between lens and source planes.

        Parameters
        ----------
        z_l: Tensor
            The lens redshift.

            *Unit: unitless*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The time delay distance for each pair of lens and source redshifts.

            *Unit: Mpc*

        """
        d_l = self.angular_diameter_distance(z_l)
        d_s = self.angular_diameter_distance(z_s)
        d_ls = self.angular_diameter_distance_z1z2(z_l, z_s)
        return (1 + z_l) * d_l * d_s / d_ls

    @forward
    def critical_surface_density(
        self,
        z_l: Tensor,
        z_s: Tensor,
    ) -> Tensor:
        """
        Compute the critical surface density between lens and source planes.

        Parameters
        ----------
        z_l: Tensor
            The lens redshift.

            *Unit: unitless*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The critical surface density for each pair of lens and source redshifts.

            *Unit: Msun/Mpc^2*

        """
        d_l = self.angular_diameter_distance(z_l)
        d_s = self.angular_diameter_distance(z_s)
        d_ls = self.angular_diameter_distance_z1z2(z_l, z_s)
        return d_s / (4 * pi * G_over_c2 * d_l * d_ls)  # fmt: skip
