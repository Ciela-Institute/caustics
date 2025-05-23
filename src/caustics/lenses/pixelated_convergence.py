# mypy: disable-error-code="index,dict-item"
from typing import Optional, Annotated, Union, Literal

import torch
from torch import Tensor
import numpy as np
from caskade import forward, Param

from ..utils import interp2d
from .base import ThinLens, CosmologyType, NameType, ZType
from . import func

__all__ = ("PixelatedConvergence",)


class PixelatedConvergence(ThinLens):
    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "convergence_map": np.logspace(0, 1, 100, dtype=np.float32).reshape(10, 10),
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
        convergence_map: Annotated[
            Optional[Tensor],
            "A 2D tensor representing the convergence map",
            True,
        ] = None,
        scale: Annotated[
            Optional[Tensor], "A scale factor to multiply by the convergence map", True
        ] = 1.0,
        shape: Annotated[
            Optional[tuple[int, ...]], "The shape of the convergence map"
        ] = None,
        convolution_mode: Annotated[
            Literal["fft", "conv2d"],
            "The convolution mode for calculating deflection angles and lensing potential",
        ] = "fft",
        use_next_fast_len: Annotated[
            bool,
            "If True, adds additional padding to speed up the FFT by calling `scipy.fft.next_fast_len`",
        ] = True,
        padding: Annotated[
            Literal["zero", "circular", "reflect", "tile"],
            "Specifies the type of padding",
        ] = "zero",
        window_kernel: Annotated[float, "Amount of kernel to be windowed"] = 1.0 / 8.0,
        name: NameType = None,
    ):
        """Strong lensing with user provided kappa map

        PixelatedConvergence is a class for strong gravitational lensing with a
        user-provided kappa map. It inherits from the ThinLens class. This class
        enables the computation of deflection angles and lensing potential by
        applying the user-provided kappa map to a grid using either Fast Fourier
        Transform (FFT) or a 2D convolution.

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

        z_s: Optional[Tensor]
            The redshift of the source.

            *Unit: unitless*

        x0: Optional[Tensor]
            The x-coordinate of the center of the grid.

            *Unit: arcsec*

        y0: Optional[Tensor]
            The y-coordinate of the center of the grid.

            *Unit: arcsec*

        convergence_map: Optional[Tensor]
            A 2D tensor representing the convergence map.

            *Unit: unitless*

        shape: Optional[tuple[int, ...]]
            The shape of the convergence map.

        convolution_mode: str, optional
            The convolution mode for calculating deflection angles and lensing
            potential. It can be either "fft" (Fast Fourier Transform) or
            "conv2d" (2D convolution). Default is "fft".

        use_next_fast_len: bool, optional
            If True, adds additional padding to speed up the FFT by calling
            `scipy.fft.next_fast_len`. The speed boost can be substantial when
            `n_pix` is a multiple of a small prime number. Default is True.

        padding: { "zero", "circular", "reflect", "tile" }

            Specifies the type of padding to use: "zero" will do zero padding,
            "circular" will do cyclic boundaries. "reflect" will do reflection
            padding. "tile" will tile the image at 2x2 which basically identical
            to circular padding, but is easier. Use zero padding to represent an
            overdensity, the other padding schemes represent a mass distribution
            embedded in a field of similar mass distributions.

            Generally you should use either "zero" or "tile".

        window_kernel: float, optional
            Amount of kernel to be windowed, specify the fraction of the kernel
            size from which a linear window scaling will ensure the edges go to
            zero for the purpose of FFT stability. Set to 0 for no windowing.
            Default is 1/8.

        """

        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        if convergence_map is not None and convergence_map.ndim != 2:
            raise ValueError(
                f"convergence_map must be 2D. Received a {convergence_map.ndim}D tensor)"
            )
        elif shape is not None and len(shape) != 2:
            raise ValueError(f"shape must specify a 2D tensor. Received shape={shape}")

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.convergence_map = Param(
            "convergence_map", convergence_map, shape, units="unitless"
        )
        self.scale = Param("scale", scale, units="flux", valid=(0, None))
        self.pixelscale = pixelscale

        assert (
            self.convergence_map.shape[0] == self.convergence_map.shape[1]
        ), f"Convergence map must be square, not {self.convergence_map.shape}"
        self.n_pix = self.convergence_map.shape[0]
        self.use_next_fast_len = use_next_fast_len
        self.padding = padding

        # Construct kernels
        self.ax_kernel, self.ay_kernel, self.potential_kernel = (
            func.build_kernels_pixelated_convergence(pixelscale, self.n_pix)
        )
        # Window the kernels if needed
        if padding != "zero" and convolution_mode == "fft" and window_kernel > 0:
            window = func.build_window_pixelated_convergence(
                window_kernel, self.ax_kernel.shape
            )
            self.ax_kernel = self.ax_kernel * window
            self.ay_kernel = self.ay_kernel * window

        self.potential_kernel_tilde = None
        self.ax_kernel_tilde = None
        self.ay_kernel_tilde = None
        self._s = None

        # Triggers creation of FFTs of kernels
        self.convolution_mode = convolution_mode

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """
        Move the ConvergenceGrid object and all its tensors to the specified device and dtype.

        Parameters
        ----------
        device: Optional[torch.device]
            The target device to move the tensors to.

        dtype: Optional[torch.dtype]
            The target data type to cast the tensors to.

        """
        super().to(device, dtype)
        self.potential_kernel = self.potential_kernel.to(device=device, dtype=dtype)
        self.ax_kernel = self.ax_kernel.to(device=device, dtype=dtype)
        self.ay_kernel = self.ay_kernel.to(device=device, dtype=dtype)
        if self.potential_kernel_tilde is not None:
            self.potential_kernel_tilde = self.potential_kernel_tilde.to(device=device)
        if self.ax_kernel_tilde is not None:
            self.ax_kernel_tilde = self.ax_kernel_tilde.to(device=device)
        if self.ay_kernel_tilde is not None:
            self.ay_kernel_tilde = self.ay_kernel_tilde.to(device=device)

    @property
    def convolution_mode(self):
        """
        Get the convolution mode of the ConvergenceGrid object.

        Returns
        -------
        string
            The convolution mode, either "fft" or "conv2d".

        """
        return self._convolution_mode

    @convolution_mode.setter
    def convolution_mode(self, convolution_mode: str):
        """
        Set the convolution mode of the ConvergenceGrid object.

        Parameters
        ----------
        mode: string
            The convolution mode to be set, either "fft" or "conv2d".

        """
        if convolution_mode == "fft":
            # Create FFTs of kernels
            self.potential_kernel_tilde = torch.fft.rfft2(
                self.potential_kernel, func._fft_size(self.n_pix)
            )
            self.ax_kernel_tilde = torch.fft.rfft2(
                self.ax_kernel, func._fft_size(self.n_pix)
            )
            self.ay_kernel_tilde = torch.fft.rfft2(
                self.ay_kernel, func._fft_size(self.n_pix)
            )
        elif convolution_mode == "conv2d":
            # Drop FFTs of kernels
            self.potential_kernel_tilde = self.potential_kernel
            self.ax_kernel_tilde = self.ax_kernel
            self.ay_kernel_tilde = self.ay_kernel
        else:
            raise ValueError("invalid convolution convolution_mode")

        self._convolution_mode = convolution_mode

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        convergence_map: Annotated[Tensor, "Param"],
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
        return func.reduced_deflection_angle_pixelated_convergence(
            x0,
            y0,
            convergence_map * scale,
            x,
            y,
            self.ax_kernel_tilde,
            self.ay_kernel_tilde,
            self.pixelscale,
            self.n_pix * self.pixelscale,
            self.n_pix,
            self.padding,
            self.convolution_mode,
        )

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        convergence_map: Annotated[Tensor, "Param"],
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
        return func.potential_pixelated_convergence(
            x0,
            y0,
            convergence_map * scale,
            x,
            y,
            self.potential_kernel_tilde,
            self.pixelscale,
            self.n_pix * self.pixelscale,
            self.n_pix,
            self.padding,
            self.convolution_mode,
        )

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        convergence_map: Annotated[Tensor, "Param"],
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
        fov_x = convergence_map.shape[1] * self.pixelscale
        fov_y = convergence_map.shape[0] * self.pixelscale
        return interp2d(
            convergence_map * scale,
            (x - x0).view(-1) / fov_x * 2,
            (y - y0).view(-1) / fov_y * 2,
        ).reshape(x.shape)
