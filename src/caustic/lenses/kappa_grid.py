from math import pi

import torch
import torch.nn.functional as F

from .base import AbstractLens


class KappaGrid(AbstractLens):
    def __init__(self, kappa, thx0=0.0, thy0=0.0, fov=1.0, method="fft", cosmology=None, device=None):
        super().__init__(cosmology, device)
        self.kappa = torch.as_tensor(kappa, dtype=torch.float32, device=device)
        if kappa.ndim != 4:
            raise ValueError("Kappa map must have four dimensions")
        self.thx0 = torch.as_tensor(thx0, dtype=torch.float32, device=device)
        self.thy0 = torch.as_tensor(thy0, dtype=torch.float32, device=device)
        self.fov = torch.as_tensor(fov,   dtype=torch.float32, device=device)
        if method == "fft":
            self._method = self._fft_method
        elif method == "conv2d":
            self._method = self._conv2d_method
        else:
            raise
        self.pixels = pixels = kappa.shape[2]
        self.dx_kap = fov / (pixels - 1)  # dx on image grid
        # Convolution kernel
        x = torch.linspace(-1, 1, 2 * pixels + 1) * fov
        xx, yy = torch.meshgrid(x, x)
        rho = xx**2 + yy**2
        xconv_kernel = -self._safe_divide(xx, rho)
        yconv_kernel = -self._safe_divide(yy, rho)
        # reshape to [in_channels, out_channels, filter_height, filter_width]
        self.xconv_kernel = xconv_kernel.unsqueeze(0).unsqueeze(0)
        self.yconv_kernel = yconv_kernel.unsqueeze(0).unsqueeze(0)

    def alpha_hat(self, thx, thy, z_l):
        ...

    def alpha(self, thx, thy, z_l, z_s):
        ...

    def Psi_hat(self, thx, thy, z_l, z_s):
        ...

    def Sigma(self, xix, xiy, z_l):
        ...

    def _fft_method(self):
        x_kernel_tilde = torch.fft.fft2(-self.xconv_kernel)
        y_kernel_tilde = torch.fft.fft2(-self.yconv_kernel)
        kap = F.pad(self.kappa, [self.pixels + 1, 0, self.pixels+1, 0, 0, 0, 0, 0])
        kappa_tilde = torch.fft.fft2(kap)
        alpha_x = torch.fft.ifft2(kappa_tilde * x_kernel_tilde).real * (self.dx_kap ** 2 / pi)
        alpha_y = torch.fft.ifft2(kappa_tilde * y_kernel_tilde).real * (self.dx_kap ** 2 / pi)
        return alpha_x[..., :self.pixels, :self.pixels], alpha_y[..., :self.pixels, :self.pixels]

    def _conv2d_method(self):
        alpha_x = F.conv2d(self.kappa, self.xconv_kernel, padding="same") * (self.dx_kap ** 2 / pi)
        alpha_y = F.conv2d(self.kappa, self.yconv_kernel, padding="same") * (self.dx_kap ** 2 / pi)
        return alpha_x, alpha_y

    @staticmethod
    def _safe_divide(num, denominator):
        out = torch.zeros_like(num)
        where = denominator != 0
        out[where] = num[where] / denominator[where]
        return out


if __name__ == '__main__':
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    hf = h5py.File("../../../../data/data_1.h5", "r")
    kap = hf["kappa"][0][None, None]
    kappa = KappaGrid(kap)
    # alpha_x, alpha_y = kappa._fft_method()
    # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # axs[0].imshow(alpha_x[0, 0], cmap="seismic")
    # axs[1].imshow(alpha_y[0, 0], cmap="seismic")
    alpha_x, alpha_y = kappa._fft_method()
    alpha_x_true, alpha_y_true = kappa._conv2d_method()
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
