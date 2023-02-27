from collections import defaultdict
from math import pi
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import get_meshgrid, interpolate_image, safe_divide, safe_log
from .base import ThinLens

__all__ = ("KappaGrid",)


class KappaGrid(ThinLens):
    def __init__(
        self,
        name: str,
        fov: float,
        n_pix: int,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        thx0: Optional[Tensor] = torch.tensor(0.0),
        thy0: Optional[Tensor] = torch.tensor(0.0),
        kappa_map: Optional[Tensor] = None,
        kappa_map_shape: Optional[Tuple[int, ...]] = None,
        mode: str = "fft",
        use_next_fast_len: bool = True,
    ):
        """
        Args:
            use_next_fast_len: if true, add additional padding to speed up the FFT
                by calling `scipy.fft.next_fast_len <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.next_fast_len.html#scipy.fft.next_fast_len>`_. The speed boost can be substantial
                when `n_pix` is prime.
        """
        if kappa_map is None and kappa_map_shape is None:
            raise ValueError("if kappa_map is dynamic, its shape must be given")
        if kappa_map is not None and kappa_map_shape is not None:
            raise ValueError("if kappa_map is static, its shape must not be given")

        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        if kappa_map is None and kappa_map_shape is not None:
            self.add_param("kappa_map", shape=kappa_map_shape)
        else:
            self.add_param("kappa_map", kappa_map, kappa_map_shape)

        self.n_pix = n_pix
        self.fov = fov
        self.res = fov / n_pix
        self.use_next_fast_len = use_next_fast_len

        # Construct kernels
        x_mg, y_mg = get_meshgrid(self.res, 2 * self.n_pix, 2 * self.n_pix)
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

        self.Psi_kernel_tilde = None
        self.ax_kernel_tilde = None
        self.ay_kernel_tilde = None

        # Triggers creation of FFTs of kernels
        self.mode = mode

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        super().to(device, dtype)

        # Move kernels
        kwargs = {"device": device, "dtype": dtype}
        self.Psi_kernel = self.Psi_kernel.to(**kwargs)
        self.alpha_x_kernel = self.alpha_x_kernel.to(**kwargs)
        self.alpha_y_kernel = self.alpha_y_kernel.to(**kwargs)
        if self.Psi_kernel_tilde is not None:
            self.Psi_kernel_tilde = self.Psi_kernel_tilde.to(**kwargs)
            self.alpha_x_kernel_tilde = self.alpha_x_kernel_tilde.to(**kwargs)
            self.alpha_y_kernel_tilde = self.alpha_y_kernel_tilde.to(**kwargs)

    def _fft2_padded(self, x):
        pad = 2 * self.n_pix
        if self.use_next_fast_len:
            pad = next_fast_len(pad)
        return torch.fft.fft2(x, (pad, pad))

    def _unpad_fft(self, x):
        return x[..., -self.n_pix :, -self.n_pix :]

    def _unpad_conv2d(self, x):
        return x[..., 1:, 1:]

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode == "fft":
            # Create FFTs of kernels
            self.Psi_kernel_tilde = self._fft2_padded(self.Psi_kernel)
            self.ax_kernel_tilde = self._fft2_padded(self.ax_kernel)
            self.ay_kernel_tilde = self._fft2_padded(self.ay_kernel)
        elif mode == "conv2d":
            # Drop FFTs of kernels
            self.Psi_kernel_tilde = None
            self.ax_kernel_tilde = None
            self.ay_kernel_tilde = None
        else:
            raise ValueError("invalid convolution mode")

        self._mode = mode

    def _check_kappa_map_shape(self, kappa_map):
        if kappa_map.ndim != 4:
            raise ValueError("kappa map must have four dimensions")

        expected_shape = (1, self.n_pix, self.n_pix)
        if kappa_map.shape[-3:] != expected_shape:
            raise ValueError(
                f"kappa map shape does not have the expected shape of {expected_shape}"
            )

    def alpha(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tuple[Tensor, Tensor]:
        z_l, thx0, thy0, kappa_map = self.unpack(x)

        self._check_kappa_map_shape(kappa_map)
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
        kappa_tilde = self._fft2_padded(kappa_map)
        alpha_x = torch.fft.ifft2(kappa_tilde * self.ax_kernel_tilde).real * (
            self.res**2 / pi
        )
        alpha_y = torch.fft.ifft2(kappa_tilde * self.ay_kernel_tilde).real * (
            self.res**2 / pi
        )
        return self._unpad_fft(alpha_x), self._unpad_fft(alpha_y)

    def _alpha_conv2d(self, kappa_map):
        alpha_x = F.conv2d(self.ax_kernel, kappa_map) * (self.res**2 / pi)
        alpha_y = F.conv2d(self.ay_kernel, kappa_map) * (self.res**2 / pi)
        return self._unpad_conv2d(alpha_x), self._unpad_conv2d(alpha_y)

    def Psi(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tensor:
        z_l, thx0, thy0, kappa_map = self.unpack(x)

        self._check_kappa_map_shape(kappa_map)
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

    def kappa(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tensor:
        raise NotImplementedError()
