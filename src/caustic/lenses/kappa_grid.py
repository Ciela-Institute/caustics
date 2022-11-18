from math import pi

import torch
import torch.nn.functional as F

from ..interpolatedimage import InterpolatedImage
from .base import AbstractLens


class KappaGrid(AbstractLens):
    def __init__(
        self, fov, n_pix, dtype=torch.float32, mode="fft", z_l=None, device=None
    ):
        super().__init__(device)

        self.n_pix = n_pix  # kappa_map.shape[2]
        self.fov = fov
        self.interpolator = InterpolatedImage(fov, device)
        self.dx_kap = fov / (n_pix - 1)  # dx on image grid
        self.z_l = z_l

        self.Psi_kernel = self.get_Psi_kernel(n_pix, fov, dtype)
        self.ax_kernel, self.ay_kernel = self.get_alpha_kernels(n_pix, fov, dtype)

        if mode == "fft":
            # Get (padded) Fourier transforms of kernels
            self.Psi_kernel_tilde = self._fft2_padded(self.Psi_kernel)
            self.ax_kernel_tilde = self._fft2_padded(-self.ax_kernel)
            self.ay_kernel_tilde = self._fft2_padded(-self.ay_kernel)

            self._kappa_to_Psi = self._kappa_to_Psi_fft
            self._kappa_to_alpha = self._kappa_to_alpha_fft
        elif mode == "conv2d":
            self._kappa_to_Psi = self._kappa_to_Psi_conv2d
            self._kappa_to_alpha = self._kappa_to_alpha_conv2d
        else:
            raise ValueError("invalid convolution mode")

        self._mode = mode

    def _fft2_padded(self, x):
        pad = 3 * self.n_pix
        return torch.fft.fft2(x, (pad, pad))

    def _unpad(self, x):
        return x[..., self.n_pix : -self.n_pix, self.n_pix : -self.n_pix]

    @property
    def mode(self):
        return self._mode

    @staticmethod
    def _safe_divide(num, denominator):
        out = torch.zeros_like(num)
        where = denominator != 0
        out[where] = num[where] / denominator[where]
        return out

    @staticmethod
    def get_alpha_kernels(n_pix, fov, dtype):
        # Shapes are (in_channels, out_channels, filter_height, filter_width)
        grid = torch.linspace(-1, 1, 2 * n_pix, dtype=dtype) * fov
        x_mg, y_mg = torch.meshgrid(grid, grid, indexing="xy")
        d2 = x_mg**2 + y_mg**2
        ax_kernel = -KappaGrid._safe_divide(x_mg, d2)[None, None, :, :]
        ay_kernel = -KappaGrid._safe_divide(y_mg, d2)[None, None, :, :]
        return ax_kernel, ay_kernel

    @staticmethod
    def get_Psi_kernel(n_pix, fov, dtype):
        grid = torch.linspace(-1, 1, 2 * n_pix, dtype=dtype) * fov
        x_mg, y_mg = torch.meshgrid(grid, grid, indexing="xy")
        d2 = x_mg**2 + y_mg**2
        return d2.sqrt().log()[None, None, :, :]

    def alpha(self, thx, thy, z_l, z_s, cosmology, kappa_map, thx0=0.0, thy0=0.0):
        if z_l != self.z_l:
            raise ValueError(
                "dynamically setting the lens redshift is not yet supported"
            )

        alpha_x_map, alpha_y_map = self._kappa_to_alpha(kappa_map)
        alpha_x = self.interpolator(thx, thy, thx0, thy0, alpha_x_map)
        alpha_y = self.interpolator(thx, thy, thx0, thy0, alpha_y_map)
        return alpha_x, alpha_y

    def Psi(self, thx, thy, z_l, z_s, cosmology, kappa_map, thx0=0.0, thy0=0.0):
        if z_l != self.z_l:
            raise ValueError(
                "dynamically setting the lens redshift is not yet supported"
            )

        Psi_map = self._kappa_to_Psi(kappa_map)
        Psi = self.interpolator(thx, thy, thx0, thy0, Psi_map)
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

    def _kappa_to_alpha_fft(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        kappa_tilde = self._fft2_padded(kappa_map)
        alpha_x = torch.fft.ifft2(kappa_tilde * self.ax_kernel_tilde).real * (
            self.dx_kap**2 / pi
        )
        alpha_y = torch.fft.ifft2(kappa_tilde * self.ay_kernel_tilde).real * (
            self.dx_kap**2 / pi
        )
        return self._unpad(alpha_x), self._unpad(alpha_y)

    def _kappa_to_alpha_conv2d(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        alpha_x = F.conv2d(kappa_map, self.ax_kernel, padding="same") * (
            self.dx_kap**2 / pi
        )
        alpha_y = F.conv2d(kappa_map, self.ay_kernel, padding="same") * (
            self.dx_kap**2 / pi
        )
        return alpha_x, alpha_y

    def _kappa_to_Psi_fft(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        kappa_tilde = self._fft2_padded(kappa_map)
        Psi = torch.fft.ifft2(kappa_tilde * self.Psi_kernel_tilde).real * (
            self.dx_kap**2 / pi
        )
        return self._unpad(Psi)

    def _kappa_to_Psi_conv2d(self, kappa_map):
        self._check_kappa_map_shape(kappa_map)
        Psi = F.conv2d(kappa_map, self.Psi_kernel, padding="same") * (
            self.dx_kap**2 / pi
        )
        return Psi


if __name__ == "__main__":
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np

    hf = h5py.File("../../../../data/data_1.h5", "r")
    kap = hf["kappa"][0][None, None]
    kappa = KappaGrid(kap)
    # alpha_x, alpha_y = kappa._fft_mode()
    # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # axs[0].imshow(alpha_x[0, 0], cmap="seismic")
    # axs[1].imshow(alpha_y[0, 0], cmap="seismic")
    alpha_x, alpha_y = kappa._kappa_to_alpha_fft()
    alpha_x_true, alpha_y_true = kappa._kappa_to_alpha_conv2d()
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))
    axs[0, 0].imshow(alpha_x[0, 0], cmap="seismic")
    axs[0, 1].imshow(alpha_y[0, 0], cmap="seismic")
    axs[1, 0].imshow(alpha_x_true[0, 0], cmap="seismic")
    axs[1, 1].imshow(alpha_y_true[0, 0], cmap="seismic")
    axs[2, 0].imshow(alpha_x_true[0, 0] - alpha_x[0, 0], cmap="seismic")
    im = axs[2, 1].imshow(alpha_y_true[0, 0] - alpha_y[0, 0], cmap="seismic")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    print(np.abs(alpha_x_true[0, 0] - alpha_x[0, 0]).max())
    print(np.abs(alpha_y_true[0, 0] - alpha_y[0, 0]).max())

    plt.show()
