import torch
from torch import Tensor
from caskade import forward

from .base import ThinLens, CosmologyType, NameType, ZLType

__all__ = ("BatchedPlane",)


class BatchedPlane(ThinLens):
    """
    A class for combining multiple thin lenses into a single lensing plane. It
    is assumed that the lens parameters will have a batch dimension, internally
    this class will vmap over the batch dimension and return the combined
    lensing quantity. This class can only handle a single lens type, if you want
    to combine different lens types, use the `SinglePlane` class.

    Attributes
    ----------
    name: str
        The name of the single plane lens.

    cosmology: Cosmology
        An instance of the Cosmology class.

    lens: ThinLens
        A ThinLens object that will be vmapped over into a single lensing plane.

    """

    def __init__(
        self,
        cosmology: CosmologyType,
        lens: ThinLens,
        name: NameType = None,
        z_l: ZLType = None,
    ):
        """
        Initialize the SinglePlane lens model.
        """
        super().__init__(cosmology, z_l=z_l, name=name)
        self.lens = lens

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the total deflection angle by summing
        the deflection angles of all individual lenses.

        Parameters
        ----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        x_component: Tensor
            The x-component of the deflection angle.

            *Unit: arcsec*

        y_component: Tensor
            The y-component of the deflection angle.

            *Unit: arcsec*

        """

        # Collect the dynamic parameters to vmap over
        params = dict((p.name, p.value) for p in self.lens.local_dynamic_params)
        batchdims = dict(
            (p.name, -(len(p.shape) + 1)) for p in self.lens.local_dynamic_params
        )
        ax, ay = torch.vmap(
            lambda p: self.lens.reduced_deflection_angle(x, y, z_s, **p),
            in_dims=(batchdims,),
        )(params)
        return ax.sum(dim=0), ay.sum(dim=0)

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
    ) -> Tensor:
        """
        Calculate the total projected mass density by
        summing the mass densities of all individual lenses.

        Parameters
        ----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The total projected mass density.

            *Unit: unitless*

        """
        # Collect the dynamic parameters to vmap over
        params = dict((p.name, p.value) for p in self.lens.local_dynamic_params)
        batchdims = dict(
            (p.name, -(len(p.shape) + 1)) for p in self.lens.local_dynamic_params
        )
        convergence = torch.vmap(
            lambda p: self.lens.convergence(x, y, z_s, **p),
            in_dims=(batchdims,),
        )(params)
        return convergence.sum(dim=0)

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
    ) -> Tensor:
        """
        Compute the total lensing potential by summing
        the lensing potentials of all individual lenses.

        Parameters
        -----------
        x: Tensor
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate of the lens.

            *Unit: arcsec*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: Packed, optional
            Dynamic parameter container.

        Returns
        -------
        Tensor
            The total lensing potential.

            *Unit: arcsec^2*

        """
        # Collect the dynamic parameters to vmap over
        params = dict((p.name, p.value) for p in self.lens.local_dynamic_params)
        batchdims = dict(
            (p.name, -(len(p.shape) + 1)) for p in self.lens.local_dynamic_params
        )
        potential = torch.vmap(
            lambda p: self.lens.potential(x, y, z_s, **p),
            in_dims=(batchdims,),
        )(params)
        return potential.sum(dim=0)
