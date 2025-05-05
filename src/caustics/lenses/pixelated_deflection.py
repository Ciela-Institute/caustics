# mypy: disable-error-code="index,dict-item"
from typing import Optional, Annotated, Union

import torch
from torch import Tensor
import numpy as np
from caskade import forward, Param

from ..utils import interp2d
from .base import ThinLens, CosmologyType, NameType, ZType

__all__ = ("PixelatedDeflection",)


class PixelatedDeflection(ThinLens):
    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "deflection_map": np.linspace(-0.1, 0.1, 100, dtype=np.float32).reshape(10, 10),
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
        deflection_map: Annotated[
            Optional[Tensor],
            "A 3D tensor (2, nx, ny) representing the reduced deflection angle map",
            True,
        ] = None,
        scale: Annotated[
            Optional[Tensor], "A scale factor to multiply by the deflection map", True
        ] = 1.0,
        shape: Annotated[
            Optional[tuple[int, ...]], "The shape of the deflection map"
        ] = None,
        name: NameType = None,
    ):
        """Strong lensing with user provided deflection map

        This class enables the computation of deflection angles by interpolating
        the user-provided deflection map.

        Attributes
        ----------
        name: string
            The name of the PixelatedDeflection object.

        cosmology: Cosmology
            An instance of the cosmological parameters.

        z_l: Optional[Tensor]
            The redshift of the lens.

            *Unit: unitless*

        z_s: Optional[Tensor]
            The redshift of the source.

            *Unit: unitless*

        x0: Optional[Tensor]
            The x-coordinate of the center of the grid.

            *Unit: arcsec*

        y0: Optional[Tensor]
            The y-coordinate of the center of the grid.

            *Unit: arcsec*

        deflection_map: Optional[Tensor]
            A 2D tensor representing the deflection map.

            *Unit: unitless*

        shape: Optional[tuple[int, ...]]
            The shape of the deflection map.

        """

        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        if deflection_map is not None and deflection_map.ndim != 3:
            raise ValueError(
                f"deflection_map must be 3D (2, nx, ny). Received a {deflection_map.ndim}D tensor)"
            )
        elif shape is not None and len(shape) != 3:
            raise ValueError(
                f"shape must specify a 3D tensor (2, nx, ny). Received shape={shape}"
            )

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.deflection_map = Param(
            "deflection_map", deflection_map, shape, units="unitless"
        )
        self.scale = Param("scale", scale, units="flux", valid=(0, None))

        self.pixelscale = pixelscale

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        deflection_map: Annotated[Tensor, "Param"],
        scale: Annotated[Tensor, "Param"],
    ) -> Tensor:
        """
        Compute the deflection at the specified positions.

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
            The deflection at the specified positions.

            *Unit: unitless*

        """
        fov_x = deflection_map.shape[2] * self.pixelscale
        fov_y = deflection_map.shape[1] * self.pixelscale
        return (
            interp2d(
                deflection_map[0] * scale,
                (x - x0).view(-1) / fov_x * 2,
                (y - y0).view(-1) / fov_y * 2,
            ).reshape(x.shape),
            interp2d(
                deflection_map[1] * scale,
                (x - x0).view(-1) / fov_x * 2,
                (y - y0).view(-1) / fov_y * 2,
            ).reshape(x.shape),
        )

    @forward
    def potential(self, x, y, **kwargs):
        raise NotImplementedError(
            "Potential is not implemented for PixelatedDeflection."
        )

    @forward
    def convergence(self, x, y, **kwargs):
        raise NotImplementedError(
            "Convergence is not implemented for PixelatedDeflection."
        )
