from typing import Optional
from warnings import warn

from caskade import forward

from ..backend_obj import backend, ArrayLike
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
        self.hierarchical_link("lens", lens)
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

    @forward
    def reduced_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        lens_params,
        lens_dims,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculate the total deflection angle by summing
        the deflection angles of all individual lenses.

        Parameters
        ----------
        x: ArrayLike
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        x_component: ArrayLike
            The x-component of the deflection angle.

            *Unit: arcsec*

        y_component: ArrayLike
            The y-component of the deflection angle.

            *Unit: arcsec*

        """
        print(lens_dims)
        vr_deflection_angle = vmap_reduce(
            lambda *args: self.lens.reduced_deflection_angle(
                args[0], args[1], args[2:]
            ),
            reduce_func=lambda x: (backend.sum(x[0], dim=0), backend.sum(x[1], dim=0)),
            chunk_size=self.chunk_size,
            in_dims=(None, None) + tuple(lens_dims),
            out_dims=(0, 0),
        )
        return vr_deflection_angle(x, y, *lens_params)

    @forward
    def convergence(
        self,
        x: ArrayLike,
        y: ArrayLike,
        lens_params,
        lens_dims,
    ) -> ArrayLike:
        """
        Calculate the total projected mass density by
        summing the mass densities of all individual lenses.

        Parameters
        ----------
        x: ArrayLike
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The total projected mass density.

            *Unit: unitless*

        """
        vr_convergence = vmap_reduce(
            lambda *args: self.lens.convergence(args[0], args[1], args[2:]),
            chunk_size=self.chunk_size,
            in_dims=(None, None) + tuple(lens_dims),
        )
        return vr_convergence(x, y, *lens_params)

    @forward
    def potential(
        self,
        x: ArrayLike,
        y: ArrayLike,
        lens_params,
        lens_dims,
    ) -> ArrayLike:
        """
        Compute the total lensing potential by summing
        the lensing potentials of all individual lenses.

        Parameters
        -----------
        x: ArrayLike
            The x-coordinate of the lens.

            *Unit: arcsec*

        y: ArrayLike
            The y-coordinate of the lens.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The total lensing potential.

            *Unit: arcsec^2*

        """
        vr_potential = vmap_reduce(
            lambda *args: self.lens.potential(args[0], args[1], args[2:]),
            chunk_size=self.chunk_size,
            in_dims=(None, None) + tuple(lens_dims),
        )
        return vr_potential(x, y, *lens_params)
