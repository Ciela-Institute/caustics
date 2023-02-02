from math import pi
from typing import Optional

import torch
import torch.nn.functional as F

from ..utils import get_meshgrid, interpolate_image, safe_divide, safe_log
from .base import ThinLens


class KappaGrid(ThinLens):
    def __init__(
        self,
        fov,
        n_pix,
        mode="fft",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device, dtype)

        self.n_pix = n_pix
        self.fov = fov
        self.res = fov / n_pix

        # Construct kernels
        x_mg, y_mg = get_meshgrid(
            self.res, 2 * self.n_pix, 2 * self.n_pix, device, dtype
        )
        # Shift to center kernels within pixel at index n_pix
        x_mg = x_mg - self.res / 2
        y_mg = y_mg - self.res / 2
        d2 = x_mg**2 + y_mg**2
        self.Psi_kernel = safe_log(d2.sqrt())[None, None, :, :]
        self.ax_kernel = safe_divide(x_mg, d2)[None, None, :, :]
        self.ay_kernel = safe_divide(y_mg, d2)[None, None, :, :]
        # Set centers of kernels to zero
        self.Psi_kernel[..., self.n_pix, self.n_pix] = 0
        self.ax_kernel[..., self.n_pix, self.n_pix] = 0
        self.ay_kernel[..., self.n_pix, self.n_pix] = 0

        # FTs of kernels
        self.Psi_kernel_tilde = None
        self.ax_kernel_tilde = None
        self.ay_kernel_tilde = None

        self.mode = mode

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        self.Psi_kernel = self.Psi_kernel.to(device=device, dtype=dtype)
        self.alpha_x_kernel = self.alpha_x_kernel.to(device=device, dtype=dtype)
        self.alpha_y_kernel = self.alpha_y_kernel.to(device=device, dtype=dtype)
        if self.Psi_kernel_tilde is not None:
            self.Psi_kernel_tilde = self.Psi_kernel_tilde.to(device=device, dtype=dtype)
            self.alpha_x_kernel_tilde = self.alpha_x_kernel_tilde.to(
                device=device, dtype=dtype
            )
            self.alpha_y_kernel_tilde = self.alpha_y_kernel_tilde.to(
                device=device, dtype=dtype
            )

    def _fft2_padded(self, x):
        # TODO: next_fast_len
        pad = 2 * self.n_pix
        return torch.fft.fft2(x, (pad, pad))

    def _unpad_fft(self, x):
        return x[..., self.n_pix :, self.n_pix :]

    def _unpad_conv2d(self, x):
        return x[..., 1:, 1:]

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode == "fft":
            self.Psi_kernel_tilde = self._fft2_padded(self.Psi_kernel)
            self.ax_kernel_tilde = self._fft2_padded(self.ax_kernel)
            self.ay_kernel_tilde = self._fft2_padded(self.ay_kernel)
        elif mode != "conv2d":
            raise ValueError("invalid convolution mode")

        self._mode = mode

    def _check_kappa_map_shape(self, kappa_map):
        if kappa_map.ndim != 4:
            raise ValueError("kappa map must have four dimensions")

        expected_shape = (1, self.n_pix, self.n_pix)
        if kappa_map.shape[1:] != expected_shape:
            raise ValueError(
                f"kappa map shape does not have the expected shape of {expected_shape}"
            )

    def alpha(self, thx, thy, z_l, z_s, cosmology, kappa_map, thx0=0.0, thy0=0.0):
        if self.mode == "fft":
            alpha_x_map, alpha_y_map = self._alpha_fft(kappa_map)
        else:
            alpha_x_map, alpha_y_map = self._alpha_conv2d(kappa_map)

        # Scale is distance from center of image to center of pixel on the edge
        alpha_x = interpolate_image(
            thx, thy, thx0, thy0, alpha_x_map, (self.fov - self.res) / 2
        )
        alpha_y = interpolate_image(
            thx, thy, thx0, thy0, alpha_y_map, (self.fov - self.res) / 2
        )
        return alpha_x, alpha_y

    def _alpha_fft(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        kappa_tilde = self._fft2_padded(kappa_map)
        alpha_x = torch.fft.ifft2(kappa_tilde * self.ax_kernel_tilde).real * (
            self.res**2 / pi
        )
        alpha_y = torch.fft.ifft2(kappa_tilde * self.ay_kernel_tilde).real * (
            self.res**2 / pi
        )
        return self._unpad_fft(alpha_x), self._unpad_fft(alpha_y)

    def _alpha_conv2d(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        alpha_x = F.conv2d(self.ax_kernel, kappa_map) * (self.res**2 / pi)
        alpha_y = F.conv2d(self.ay_kernel, kappa_map) * (self.res**2 / pi)
        return self._unpad_conv2d(alpha_x), self._unpad_conv2d(alpha_y)

    def Psi(self, thx, thy, z_l, z_s, cosmology, kappa_map, thx0=0.0, thy0=0.0):
        if self.mode == "fft":
            Psi_map = self._Psi_fft(kappa_map)
        else:
            Psi_map = self._Psi_conv2d(kappa_map)

        # Scale is distance from center of image to center of pixel on the edge
        Psi = interpolate_image(
            thx, thy, thx0, thy0, Psi_map, (self.fov - self.res) / 2
        )
        return Psi

    def _Psi_fft(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        kappa_tilde = self._fft2_padded(kappa_map)
        Psi = torch.fft.ifft2(kappa_tilde * self.Psi_kernel_tilde).real * (
            self.res**2 / pi
        )
        return self._unpad_fft(Psi)

    def _Psi_conv2d(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        Psi = F.conv2d(self.Psi_kernel, kappa_map) * (self.res**2 / pi)
        return self._unpad_conv2d(Psi)

    def kappa(self, thx, thy, z_s):
        ...
