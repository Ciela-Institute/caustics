from math import pi

import torch
import torch.nn.functional as F

from ..interpolate_image import interpolate_image
from ..utils import get_meshgrid, safe_divide, safe_log
from .base import AbstractThinLens


class KappaGrid(AbstractThinLens):
    def __init__(
        self, fov, n_pix, dtype=torch.float32, mode="fft", z_l=None, device=None
    ):
        super().__init__(device)

        self.n_pix = n_pix  # kappa_map.shape[2]
        self.fov = fov
        self.res = fov / n_pix
        self.dx_kap = fov / (n_pix - 1)  # dx on image grid
        self.z_l = z_l

        x_mg, y_mg = get_meshgrid(
            self.res, 2 * self.n_pix, 2 * self.n_pix, self.device, dtype
        )
        # Shift to center kernel within pixel
        x_mg = x_mg - self.res / 2
        y_mg = y_mg - self.res / 2
        d2 = x_mg**2 + y_mg**2
        self.Psi_kernel = safe_log(d2.sqrt())[None, None, :, :]
        self.ax_kernel = -safe_divide(x_mg, d2)[None, None, :, :]
        self.ay_kernel = -safe_divide(y_mg, d2)[None, None, :, :]

        self.mode = mode

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
            self.ax_kernel_tilde = self._fft2_padded(-self.ax_kernel)
            self.ay_kernel_tilde = self._fft2_padded(-self.ay_kernel)
        elif mode != "conv2d":
            raise ValueError("invalid convolution mode")

        self._mode = mode

    def alpha(self, thx, thy, z_l, z_s, cosmology, kappa_map, thx0=0.0, thy0=0.0):
        if z_l != self.z_l:
            raise ValueError(
                "dynamically setting the lens redshift is not yet supported"
            )

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

    def Psi(self, thx, thy, z_l, z_s, cosmology, kappa_map, thx0=0.0, thy0=0.0):
        if z_l != self.z_l:
            raise ValueError(
                "dynamically setting the lens redshift is not yet supported"
            )

        if self.mode == "fft":
            Psi_map = self._Psi_fft(kappa_map)
        else:
            Psi_map = self._Psi_conv2d(kappa_map)

        # Scale is distance from center of image to center of pixel on the edge
        Psi = interpolate_image(
            thx, thy, thx0, thy0, Psi_map, (self.fov - self.res) / 2
        )
        return Psi

    def kappa(self, thx, thy, z_s):
        ...

    def _check_kappa_map_shape(self, kappa_map):
        if kappa_map.ndim != 4:
            raise ValueError("kappa map must have four dimensions")

        expected_shape = (1, self.n_pix, self.n_pix)
        if kappa_map.shape[1:] != expected_shape:
            raise ValueError(
                f"kappa map shape does not have the expected shape of {expected_shape}"
            )

    def _alpha_fft(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        kappa_tilde = self._fft2_padded(kappa_map)
        alpha_x = torch.fft.ifft2(kappa_tilde * self.ax_kernel_tilde).real * (
            self.dx_kap**2 / pi
        )
        alpha_y = torch.fft.ifft2(kappa_tilde * self.ay_kernel_tilde).real * (
            self.dx_kap**2 / pi
        )
        return self._unpad_fft(alpha_x), self._unpad_fft(alpha_y)

    def _alpha_conv2d(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        alpha_x = F.conv2d(kappa_map, self.ax_kernel, padding="same") * (
            self.dx_kap**2 / pi
        )
        alpha_y = F.conv2d(kappa_map, self.ay_kernel, padding="same") * (
            self.dx_kap**2 / pi
        )
        return self._unpad_conv2d(alpha_x), self._unpad_conv2d(alpha_y)

    def _Psi_fft(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        kappa_tilde = self._fft2_padded(kappa_map)
        Psi = torch.fft.ifft2(kappa_tilde * self.Psi_kernel_tilde).real * (
            self.dx_kap**2 / pi
        )
        return self._unpad_fft(Psi)

    def _Psi_conv2d(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        print(kappa_map.shape, self.Psi_kernel.shape)
        Psi = F.conv2d(self.Psi_kernel, kappa_map) * (self.dx_kap**2 / pi)
        return self._unpad_conv2d(Psi)
