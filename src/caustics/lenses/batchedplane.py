from typing import Optional
from warnings import warn

import torch
from torch import Tensor
from caskade import forward

from .base import ThinLens, CosmologyType, NameType, ZType
from ..utils import vmap_reduce

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
        z_l: ZType = None,
        z_s: ZType = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Initialize the SinglePlane lens model.
        """
        super().__init__(cosmology, z_l=z_l, name=name, z_s=z_s)
        self.lens = lens
        if lens.z_l.static:
            warn(
                f"Lens model {lens.name} has a static lens redshift. This is now overwritten by the BatchedPlane ({self.name}) lens redshift. To prevent this warning, set the lens redshift of the lens model to be dynamic before adding to the system."
            )
        self.lens.z_l = self.z_l
        if lens.z_s.static:
            warn(
                f"Lens model {lens.name} has a static source redshift. This is now overwritten by the BatchedPlane ({self.name}) source redshift. To prevent this warning, set the source redshift of the lens model to be dynamic before adding to the system."
            )
        self.lens.z_s = self.z_s
        self.chunk_size = chunk_size

    @property
    def s(self) -> float:
        """
        Softening parameter to prevent numerical instabilities.

        *Unit: arcsec*

        """
        return self.lens.s

    @s.setter
    def s(self, value: float) -> None:
        """
        Set the softening parameter to prevent numerical instabilities.

        Parameters
        ----------
        value: float
            The softening parameter.

            *Unit: arcsec*

        """
        self.lens.s = value

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
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
        params = dict(
            (p.name, p.value) for p in self.lens.local_dynamic_params.values()
        )
        batchdims = dict(
            (p.name, -(len(p.shape) + 1))
            for p in self.lens.local_dynamic_params.values()
        )
        batchdims["x"] = None
        batchdims["y"] = None
        vr_deflection_angle = vmap_reduce(
            lambda p: self.lens.reduced_deflection_angle(**p),
            reduce_func=lambda x: (x[0].sum(dim=0), x[1].sum(dim=0)),
            chunk_size=self.chunk_size,
            in_dims=batchdims,
            out_dims=(0, 0),
        )
        return vr_deflection_angle(x=x, y=y, **params)

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
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

        Returns
        -------
        Tensor
            The total projected mass density.

            *Unit: unitless*

        """
        # Collect the dynamic parameters to vmap over
        params = dict(
            (p.name, p.value) for p in self.lens.local_dynamic_params.values()
        )
        batchdims = dict(
            (p.name, -(len(p.shape) + 1))
            for p in self.lens.local_dynamic_params.values()
        )
        convergence = torch.vmap(
            lambda p: self.lens.convergence(x, y, **p),
            in_dims=(batchdims,),
            chunk_size=self.chunk_size,
        )(params)
        return convergence.sum(dim=0)

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
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

        Returns
        -------
        Tensor
            The total lensing potential.

            *Unit: arcsec^2*

        """
        # Collect the dynamic parameters to vmap over
        params = dict(
            (p.name, p.value) for p in self.lens.local_dynamic_params.values()
        )
        batchdims = dict(
            (p.name, -(len(p.shape) + 1))
            for p in self.lens.local_dynamic_params.values()
        )
        potential = torch.vmap(
            lambda p: self.lens.potential(x, y, **p),
            in_dims=(batchdims,),
            chunk_size=self.chunk_size,
        )(params)
        return potential.sum(dim=0)
