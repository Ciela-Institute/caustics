from typing import Optional, Annotated, Union, Literal
import torch
from torch import Tensor

from .simulator import Simulator, NameType
from ..lenses.base import Lens
from ..light.base import Source


__all__ = ("Microlens",)


class Microlens(Simulator):
    """Computes the total flux from a microlens system within an fov.

    Straightforward simulator to compute the total flux a lensed image of a
    source object within a given field of view. Constructs a sampling points
    internally based on the user settings.

    Example usage:: python

       import matplotlib.pyplot as plt
       import torch
       import caustics

       cosmo = caustics.FlatLambdaCDM()
       lens = caustics.lenses.SIS(cosmology = cosmo, x0 = 0., y0 = 0., th_ein = 1.)
       source = caustics.sources.Sersic(x0 = 0., y0 = 0., q = 0.5, phi = 0.4, n = 2., Re = 1., Ie = 1.)
       sim = caustics.sims.Microlens(lens, source, z_s = 1.)

       fov = torch.tensor([-1., 1., -1., 1.])
       print("Flux and uncertainty: ", sim(fov=fov))

    Attributes
    ----------
    lens: Lens
        caustics lens mass model object
    source: Source
        caustics light object which defines the background source
    z_s: optional
        redshift of the source
    name: string (default "sim")
        a name for this simulator in the parameter DAG.

    """  # noqa: E501

    def __init__(
        self,
        lens: Annotated[Lens, "caustics lens mass model object"],
        source: Annotated[
            Source, "caustics light object which defines the background source"
        ],
        z_s: Annotated[
            Optional[Union[Tensor, float]], "Redshift of the source", True
        ] = None,
        name: NameType = "sim",
    ):
        super().__init__(name)

        self.lens = lens
        self.source = source

        self.add_param("z_s", z_s)

    def forward(
        self,
        params,
        fov: Tensor,
        method: Literal["mcmc", "grid"] = "mcmc",
        N_mcmc: int = 10000,
        N_grid: int = 100,
    ):
        """Forward pass of the simulator.

        Parameters
        ----------
        params: dict
            Dictionary of parameters for the simulator
        fov: Tensor
            Field of view box of the simulation in arcseconds indexed as (x_min, x_max, y_min, y_max)
        method: str (default "mcmc")
            Method for sampling the image. Choose from "mcmc" or "grid"
        N_mcmc: int
            Number of sample points for the source sampling if method is "mcmc"
        N_grid: int
            Number of sample points for the sampling grid on each axis if method is "grid"

        Returns
        -------
        Tensor
            Total flux from the microlens system within the field of view

        Tensor
            Error estimate on the total flux

        """
        (z_s,) = self.unpack(params)

        if method == "mcmc":
            # Sample the source using MCMC
            sample_x = torch.rand(N_mcmc) * (fov[1] - fov[0]) + fov[0]
            sample_y = torch.rand(N_mcmc) * (fov[3] - fov[2]) + fov[2]
            bx, by = self.lens.raytrace(sample_x, sample_y, z_s, params)
            mu = self.source.brightness(bx, by, params)
            A = (fov[1] - fov[0]) * (fov[3] - fov[2])
            return mu.mean() * A, mu.std() * A / N_mcmc**0.5
        elif method == "grid":
            # Sample the source using a grid
            x = torch.linspace(fov[0], fov[1], N_grid)
            y = torch.linspace(fov[2], fov[3], N_grid)
            sample_x, sample_y = torch.meshgrid(x, y, indexing="ij")
            bx, by = self.lens.raytrace(sample_x, sample_y, z_s, params)
            mu = self.source.brightness(bx, by, params)
            A = (fov[1] - fov[0]) * (fov[3] - fov[2])
            return mu.mean() * A, mu.std() * A / N_grid
        else:
            raise ValueError(f"Invalid method: {method}, choose from 'mcmc' or 'grid'")
