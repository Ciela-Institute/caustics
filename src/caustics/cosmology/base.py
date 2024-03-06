# mypy: disable-error-code="operator"
from abc import abstractmethod
from math import pi
from typing import Optional

from torch import Tensor

from ..constants import G_over_c2
from ..parametrized import Parametrized, unpack
from ..packed import Packed


class Cosmology(Parametrized):
    """
    Abstract base class for cosmological models.

    This class provides an interface for cosmological computations used in lensing
    such as comoving distance and critical surface density.

    Units
    -----
    Distance
        Mpc
    Mass
        solar mass

    Attributes
    ----------
    name: str
        Name of the cosmological model.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the Cosmology.

        Parameters
        ----------
        name: str
            Name of the cosmological model.
        """
        super().__init__(name)

    @abstractmethod
    def critical_density(self, z: Tensor, params: Optional["Packed"] = None) -> Tensor:
        """
        Compute the critical density at redshift z.

        Parameters
        ----------
        z: Tensor
            The redshifts.
        params: Packed, optional
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The critical density at each redshift.
        """
        ...

    @abstractmethod
    @unpack
    def comoving_distance(
        self, z: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the comoving distance to redshift z.

        Parameters
        ----------
        z: Tensor
            The redshifts.
        params: (Packed, optional0
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The comoving distance to each redshift.
        """
        ...

    @abstractmethod
    @unpack
    def transverse_comoving_distance(
        self, z: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the transverse comoving distance to redshift z (Mpc).

        Parameters
        ----------
        z: Tensor
            The redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The transverse comoving distance to each redshift in Mpc.
        """
        ...

    @unpack
    def comoving_distance_z1z2(
        self, z1: Tensor, z2: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the comoving distance between two redshifts.

        Parameters
        ----------
        z1: Tensor
            The starting redshifts.
        z2: Tensor
            The ending redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The comoving distance between each pair of redshifts.
        """
        return self.comoving_distance(z2, params) - self.comoving_distance(z1, params)

    @unpack
    def transverse_comoving_distance_z1z2(
        self, z1: Tensor, z2: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the transverse comoving distance between two redshifts (Mpc).

        Parameters
        ----------
        z1: Tensor
            The starting redshifts.
        z2: Tensor
            The ending redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The transverse comoving distance between each pair of redshifts in Mpc.
        """
        return self.transverse_comoving_distance(
            z2, params
        ) - self.transverse_comoving_distance(z1, params)

    @unpack
    def angular_diameter_distance(
        self, z: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the angular diameter distance to redshift z.

        Parameters
        -----------
        z: Tensor
            The redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The angular diameter distance to each redshift.
        """
        return self.comoving_distance(z, params, **kwargs) / (1 + z)

    @unpack
    def angular_diameter_distance_z1z2(
        self, z1: Tensor, z2: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the angular diameter distance between two redshifts.

        Parameters
        ----------
        z1: Tensor
            The starting redshifts.
        z2: Tensor
            The ending redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The angular diameter distance between each pair of redshifts.
        """
        return self.comoving_distance_z1z2(z1, z2, params, **kwargs) / (1 + z2)

    @unpack
    def time_delay_distance(
        self,
        z_l: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the time delay distance between lens and source planes.

        Parameters
        ----------
        z_l: Tensor
            The lens redshifts.
        z_s: Tensor
            The source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The time delay distance for each pair of lens and source redshifts.
        """
        d_l = self.angular_diameter_distance(z_l, params)
        d_s = self.angular_diameter_distance(z_s, params)
        d_ls = self.angular_diameter_distance_z1z2(z_l, z_s, params)
        return (1 + z_l) * d_l * d_s / d_ls

    @unpack
    def critical_surface_density(
        self,
        z_l: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the critical surface density between lens and source planes.

        Parameters
        ----------
        z_l: Tensor
            The lens redshifts.
        z_s: Tensor
            The source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The critical surface density for each pair of lens and source redshifts.
        """
        d_l = self.angular_diameter_distance(z_l, params)
        d_s = self.angular_diameter_distance(z_s, params)
        d_ls = self.angular_diameter_distance_z1z2(z_l, z_s, params)
        return d_s / (4 * pi * G_over_c2 * d_l * d_ls)  # fmt: skip
