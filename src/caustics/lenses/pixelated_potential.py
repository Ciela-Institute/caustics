# mypy: disable-error-code="index,dict-item"
from typing import Optional, Annotated, Union

import torch
from torch import Tensor
import numpy as np
from caskade import forward, Param

from ..utils import interp_bicubic
from .base import ThinLens, CosmologyType, NameType, ZType

__all__ = ("PixelatedPotential",)


class PixelatedPotential(ThinLens):
    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "potential_map": np.logspace(0, 1, 100, dtype=np.float32).reshape(10, 10),
    }

    def __init__(
        self,
        pixelscale: Annotated[float, "pixelscale"],
        cosmology: CosmologyType,
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]],
            "The x-coordinate of the center of the grid",
            True,
        ] = torch.tensor(0.0),
        y0: Annotated[
            Optional[Union[Tensor, float]],
            "The y-coordinate of the center of the grid",
            True,
        ] = torch.tensor(0.0),
        potential_map: Annotated[
            Optional[Tensor],
            "A 2D tensor representing the potential map",
            True,
        ] = None,
        scale: Annotated[
            Optional[Tensor], "A scale factor to multiply by the potential map", True
        ] = 1.0,
        shape: Annotated[
            Optional[tuple[int, ...]], "The shape of the potential map"
        ] = None,
        name: NameType = None,
    ):
        """Strong lensing with user provided kappa map

        PixelatedConvergence is a class for strong gravitational lensing with a
        user-provided kappa map. It inherits from the ThinLens class.
        This class enables the computation of deflection angles and
        lensing potential by applying the user-provided kappa map to a
        grid using either Fast Fourier Transform (FFT) or a 2D
        convolution.

        Attributes
        ----------
        name: string
            The name of the PixelatedConvergence object.

        fov: float
            The field of view in arcseconds.

            *Unit: arcsec*

        cosmology: Cosmology
            An instance of the cosmological parameters.

        z_l: Optional[Tensor]
            The redshift of the lens.

            *Unit: unitless*

        x0: Optional[Tensor]
            The x-coordinate of the center of the grid.

            *Unit: arcsec*

        y0: Optional[Tensor]
            The y-coordinate of the center of the grid.

            *Unit: arcsec*

        potential_map: Optional[Tensor]
            A 2D tensor representing the potential map.

            *Unit: unitless*

        shape: Optional[tuple[int, ...]]
            The shape of the potential map.

        """

        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        if potential_map is not None and potential_map.ndim != 2:
            raise ValueError(
                f"potential_map must be 2D. Received a {potential_map.ndim}D tensor)"
            )
        elif shape is not None and len(shape) != 2:
            raise ValueError(f"shape must specify a 2D tensor. Received shape={shape}")

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.potential_map = Param(
            "potential_map", potential_map, shape, units="unitless"
        )
        self.scale = Param("scale", scale, units="flux", valid=(0, None))

        self.pixelscale = pixelscale
        if potential_map is not None:
            self.n_pix = potential_map.shape[0]
        elif shape is not None:
            self.n_pix = shape[0]
        else:
            raise ValueError("Either potential_map or shape must be provided")
        self.fov = self.n_pix * self.pixelscale

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        potential_map: Annotated[Tensor, "Param"],
        scale: Annotated[Tensor, "Param"],
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the deflection angles at the specified positions using the given convergence map.

        Parameters
        ----------
        x: Tensor
            The x-coordinates of the positions to compute the deflection angles for.

            *Unit: arcsec*

        y: Tensor
            The y-coordinates of the positions to compute the deflection angles for.

            *Unit: arcsec*

        Returns
        -------
        x_component: Tensor
            Deflection Angle in the x-direction.

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle in the y-direction.

            *Unit: arcsec*

        """
        # TODO: rescale from fov units to arcsec
        return tuple(
            alpha.reshape(x.shape) / self.pixelscale
            for alpha in interp_bicubic(
                (x - x0).view(-1) / self.fov * 2,
                (y - y0).view(-1) / self.fov * 2,
                potential_map * scale,
                get_Y=False,
                get_dY=True,
                get_ddY=False,
            )
        )

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        potential_map: Annotated[Tensor, "Param"],
        scale: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Compute the lensing potential at the specified positions using the given convergence map.

        Parameters
        ----------
        x: Tensor
            The x-coordinates of the positions to compute the lensing potential for.

            *Unit: arcsec*

        y: Tensor
            The y-coordinates of the positions to compute the lensing potential for.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The lensing potential at the specified positions.

            *Unit: arcsec^2*

        """
        return interp_bicubic(
            (x - x0).view(-1) / self.fov * 2,
            (y - y0).view(-1) / self.fov * 2,
            potential_map * scale,
            get_Y=True,
            get_dY=False,
            get_ddY=False,
        )[0].reshape(x.shape)

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        potential_map: Annotated[Tensor, "Param"],
        scale: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Compute the convergence at the specified positions.

        Parameters
        ----------
        x: Tensor
            The x-coordinates of the positions to compute the convergence for.

            *Unit: arcsec*

        y: Tensor
            The y-coordinates of the positions to compute the convergence for.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The convergence at the specified positions.

            *Unit: unitless*

        """
        # TODO: rescale from fov units to arcsec
        _, dY11, dY22 = interp_bicubic(
            (x - x0).view(-1) / self.fov * 2,
            (y - y0).view(-1) / self.fov * 2,
            potential_map * scale,
            get_Y=False,
            get_dY=False,
            get_ddY=True,
        )
        return (dY11 + dY22).reshape(x.shape) / (2 * self.pixelscale**2)
