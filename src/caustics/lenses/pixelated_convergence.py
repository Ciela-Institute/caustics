# mypy: disable-error-code="index"
from math import pi
from typing import Optional

import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len
from torch import Tensor
import numpy as np

from ..cosmology import Cosmology
from ..utils import get_meshgrid, interp2d, safe_divide, safe_log
from .base import ThinLens
from ..parametrized import unpack
from ..packed import Packed

__all__ = ("PixelatedConvergence",)


class PixelatedConvergence(ThinLens):
    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "convergence_map": np.logspace(0, 1, 100, dtype=np.float32).reshape(10, 10),
    }

    def __init__(
        self,
        pixelscale: float,
        n_pix: int,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = torch.tensor(0.0),
        y0: Optional[Tensor] = torch.tensor(0.0),
        convergence_map: Optional[Tensor] = None,
        shape: Optional[tuple[int, ...]] = None,
        convolution_mode: str = "fft",
        use_next_fast_len: bool = True,
        padding: str = "zero",
        name: Optional[str] = None,
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

        n_pix: int
            The number of pixels on each side of the grid.


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

        convergence_map: Optional[Tensor]
            A 2D tensor representing the convergence map.

            *Unit: unitless*

        shape: Optional[tuple[int, ...]]
            The shape of the convergence map.

        convolution_mode: str, optional
            The convolution mode for calculating deflection angles and lensing potential.
            It can be either "fft" (Fast Fourier Transform) or "conv2d" (2D convolution).
            Default is "fft".

        use_next_fast_len: bool, optional
            If True, adds additional padding to speed up the FFT by calling
            `scipy.fft.next_fast_len`.
            The speed boost can be substantial when `n_pix` is a multiple of a
            small prime number. Default is True.

        padding: { "zero", "circular", "reflect", "tile" }

            Specifies the type of padding to use:
            "zero" will do zero padding,
            "circular" will do cyclic boundaries.
            "reflect" will do reflection padding.
            "tile" will tile the image at 2x2 which
            basically identical to circular padding, but is easier.

            Generally you should use either "zero" or "tile".

        """

        super().__init__(cosmology, z_l, name=name)

        if convergence_map is not None and convergence_map.ndim != 2:
            raise ValueError(
                f"convergence_map must be 2D. Received a {convergence_map.ndim}D tensor)"
            )
        elif shape is not None and len(shape) != 2:
            raise ValueError(f"shape must specify a 2D tensor. Received shape={shape}")

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("convergence_map", convergence_map, shape)

        self.n_pix = n_pix
        self.pixelscale = pixelscale
        self.fov = self.n_pix * self.pixelscale
        self.use_next_fast_len = use_next_fast_len
        self.padding = padding

        # Construct kernels
        x_mg, y_mg = get_meshgrid(self.pixelscale, 2 * self.n_pix, 2 * self.n_pix)
        # Shift to center kernels within pixel at index n_pix
        x_mg = x_mg - self.pixelscale / 2
        y_mg = y_mg - self.pixelscale / 2
        d2 = x_mg**2 + y_mg**2
        self.potential_kernel = safe_log(d2.sqrt())
        self.ax_kernel = safe_divide(x_mg, d2)
        self.ay_kernel = safe_divide(y_mg, d2)
        # Set centers of kernels to zero
        self.potential_kernel[..., self.n_pix, self.n_pix] = 0
        self.ax_kernel[..., self.n_pix, self.n_pix] = 0
        self.ay_kernel[..., self.n_pix, self.n_pix] = 0

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

    def _fft2_padded(self, x: Tensor) -> Tensor:
        """
        Compute the 2D Fast Fourier Transform (FFT) of a tensor with zero-padding.

        Parameters
        x: Tensor
            The input tensor to be transformed.

        Returns
        -------
        Tensor
            The 2D FFT of the input tensor with zero-padding.

        """
        pad = 2 * self.n_pix
        if self.use_next_fast_len:
            pad = next_fast_len(pad)
        self._s = (pad, pad)

        if self.padding == "zero":
            pass
        elif self.padding in ["reflect", "circular"]:
            x = F.pad(
                x[None, None], (0, self.n_pix - 1, 0, self.n_pix - 1), mode=self.padding
            ).squeeze()
        elif self.padding == "tile":
            x = torch.tile(x, (2, 2))

        return torch.fft.rfft2(x, self._s)

    def _unpad_fft(self, x: Tensor) -> Tensor:
        """
        Remove padding from the result of a 2D FFT.

        Parameters
        ----------
        x: Tensor
            The input tensor with padding.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The input tensor without padding.

        """
        return torch.roll(x, (-self._s[0] // 2, -self._s[1] // 2), dims=(-2, -1))[..., : self.n_pix, : self.n_pix]  # fmt: skip

    def _unpad_conv2d(self, x: Tensor) -> Tensor:
        """
        Remove padding from the result of a 2D convolution.

        Parameters
        ----------
        x: Tensor
            The input tensor with padding.

        Returns
        -------
        Tensor
            The input tensor without padding.

        """
        return x  # noqa: E501 torch.roll(x, (-self.padding_range * self.ax_kernel.shape[0]//4,-self.padding_range * self.ax_kernel.shape[1]//4), dims = (-2,-1))[..., :self.n_pix, :self.n_pix] #[..., 1:, 1:]

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
            self.potential_kernel_tilde = self._fft2_padded(self.potential_kernel)
            self.ax_kernel_tilde = self._fft2_padded(self.ax_kernel)
            self.ay_kernel_tilde = self._fft2_padded(self.ay_kernel)
        elif convolution_mode == "conv2d":
            # Drop FFTs of kernels
            self.potential_kernel_tilde = None
            self.ax_kernel_tilde = None
            self.ay_kernel_tilde = None
        else:
            raise ValueError("invalid convolution convolution_mode")

        self._convolution_mode = convolution_mode

    @unpack
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        convergence_map: Optional[Tensor] = None,
        **kwargs,
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

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: Packed, optional
            A dictionary containing additional parameters.

        Returns
        -------
        x_component: Tensor
            Deflection Angle in the x-direction.

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle in the y-direction.

            *Unit: arcsec*

        """
        if self.convolution_mode == "fft":
            deflection_angle_x_map, deflection_angle_y_map = self._deflection_angle_fft(
                convergence_map
            )
        else:
            (
                deflection_angle_x_map,
                deflection_angle_y_map,
            ) = self._deflection_angle_conv2d(convergence_map)
        # Scale is distance from center of image to center of pixel on the edge
        scale = self.fov / 2
        deflection_angle_x = interp2d(
            deflection_angle_x_map, (x - x0).view(-1) / scale, (y - y0).view(-1) / scale
        ).reshape(x.shape)
        deflection_angle_y = interp2d(
            deflection_angle_y_map, (x - x0).view(-1) / scale, (y - y0).view(-1) / scale
        ).reshape(x.shape)
        return deflection_angle_x, deflection_angle_y

    def _deflection_angle_fft(self, convergence_map: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the deflection angles using the Fast Fourier Transform (FFT) method.

        Parameters
        ----------
        convergence_map: Tensor
            The 2D tensor representing the convergence map.

            *Unit: unitless*

        Returns
        -------
        x_component: Tensor
            Deflection Angle in x-component.

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle in y-component.

            *Unit: arcsec*

        """
        convergence_tilde = self._fft2_padded(convergence_map)
        deflection_angle_x = torch.fft.irfft2(
            convergence_tilde * self.ax_kernel_tilde, self._s
        ).real * (self.pixelscale**2 / pi)
        deflection_angle_y = torch.fft.irfft2(
            convergence_tilde * self.ay_kernel_tilde, self._s
        ).real * (self.pixelscale**2 / pi)
        return self._unpad_fft(deflection_angle_x), self._unpad_fft(deflection_angle_y)

    def _deflection_angle_conv2d(
        self, convergence_map: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the deflection angles using the 2D convolution method.

        Parameters
        ----------
        convergence_map: Tensor
            The 2D tensor representing the convergence map.

            *Unit: unitless*

        Returns
        -------
        x_component: Tensor
            Deflection Angle

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle

            *Unit: arcsec*

        """
        # Use convergence_map as kernel since the kernel is twice as large. Flip since
        # we actually want the cross-correlation.

        2 * self.n_pix
        convergence_map_flipped = convergence_map.flip((-1, -2))[None, None]
        # noqa: E501 F.pad(, ((pad - self.n_pix)//2, (pad - self.n_pix)//2, (pad - self.n_pix)//2, (pad - self.n_pix)//2), mode = self.padding_mode)
        deflection_angle_x = F.conv2d(
            self.ax_kernel[None, None], convergence_map_flipped, padding="same"
        ).squeeze() * (self.pixelscale**2 / pi)
        deflection_angle_y = F.conv2d(
            self.ay_kernel[None, None], convergence_map_flipped, padding="same"
        ).squeeze() * (self.pixelscale**2 / pi)
        return self._unpad_conv2d(deflection_angle_x), self._unpad_conv2d(
            deflection_angle_y
        )

    @unpack
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        convergence_map: Optional[Tensor] = None,
        **kwargs,
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

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: (Packed, optional)
            A dictionary containing additional parameters.

        Returns
        -------
        Tensor
            The lensing potential at the specified positions.

            *Unit: arcsec^2*

        """
        if self.convolution_mode == "fft":
            potential_map = self._potential_fft(convergence_map)
        else:
            potential_map = self._potential_conv2d(convergence_map)

        # Scale is distance from center of image to center of pixel on the edge
        scale = self.fov / 2
        return interp2d(
            potential_map, (x - x0).view(-1) / scale, (y - y0).view(-1) / scale
        ).reshape(x.shape)

    def _potential_fft(self, convergence_map: Tensor) -> Tensor:
        """
        Compute the lensing potential using the Fast Fourier Transform (FFT) method.

        Parameters
        ----------
        convergence_map: Tensor
            The 2D tensor representing the convergence map.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        convergence_tilde = self._fft2_padded(convergence_map)
        potential = torch.fft.irfft2(
            convergence_tilde * self.potential_kernel_tilde, self._s
        ) * (self.pixelscale**2 / pi)
        return self._unpad_fft(potential)

    def _potential_conv2d(self, convergence_map: Tensor) -> Tensor:
        """
        Compute the lensing potential using the 2D convolution method.

        Parameters
        ----------
        convergence_map: Tensor
            The 2D tensor representing the convergence map.

            *Unit: unitless*

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        # Use convergence_map as kernel since the kernel is twice as large. Flip since
        # we actually want the cross-correlation.
        convergence_map_flipped = convergence_map.flip((-1, -2))[None, None]
        potential = F.conv2d(
            self.potential_kernel[None, None], convergence_map_flipped, padding="same"
        ).squeeze() * (self.pixelscale**2 / pi)
        return self._unpad_conv2d(potential)

    @unpack
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        convergence_map: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the convergence at the specified positions. This method is not implemented.

        Parameters
        ----------
        x: Tensor
            The x-coordinates of the positions to compute the convergence for.

            *Unit: arcsec*

        y: Tensor
            The y-coordinates of the positions to compute the convergence for.

            *Unit: arcsec*

        z_s: Tensor
            The source redshift.

            *Unit: unitless*

        params: (Packed, optional)
            A dictionary containing additional parameters.

        Returns
        -------
        Tensor
            The convergence at the specified positions.

            *Unit: unitless*

        Raises
        ------
        NotImplementedError
            This method is not implemented.

        """
        return interp2d(
            convergence_map,
            (x - x0).view(-1) / self.fov * 2,
            (y - y0).view(-1) / self.fov * 2,
        ).reshape(x.shape)
