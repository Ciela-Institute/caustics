from math import pi
from typing import Any, Optional

import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import get_meshgrid, interp2d, safe_divide, safe_log
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
        kappa_map_shape: Optional[tuple[int, ...]] = None,
        mode: str = "fft",
        use_next_fast_len: bool = True,
    ):
        """Strong lensing with user provided kappa map

        KappaGrid is a class for strong gravitational lensing with a
        user-provided kappa map. It inherits from the ThinLens class.
        This class enables the computation of deflection angles and
        lensing potential by applying the user-provided kappa map to a
        grid using either Fast Fourier Transform (FFT) or a 2D
        convolution.

        Attributes:
            name (str): The name of the KappaGrid object.
            fov (float): The field of view in arcseconds.
            n_pix (int): The number of pixels on each side of the grid.
            cosmology (Cosmology): An instance of the cosmological parameters.
            z_l (Optional[Tensor]): The redshift of the lens.
            thx0 (Optional[Tensor]): The x-coordinate of the center of the grid.
            thy0 (Optional[Tensor]): The y-coordinate of the center of the grid.
            kappa_map (Optional[Tensor]): A 2D tensor representing the kappa map.
            kappa_map_shape (Optional[tuple[int, ...]]): The shape of the kappa map.
            mode (str, optional): The convolution mode for calculating deflection angles and lensing potential.
                It can be either "fft" (Fast Fourier Transform) or "conv2d" (2D convolution). Default is "fft".
            use_next_fast_len (bool, optional): If True, adds additional padding to speed up the FFT by calling
                `scipy.fft.next_fast_len`. The speed boost can be substantial when `n_pix` is prime. Default is True.

        """
        
        super().__init__(name, cosmology, z_l)

        if kappa_map is not None and kappa_map.ndim != 2:
            raise ValueError(
                f"kappa_map must be 2D (received {kappa_map.ndim}D tensor)"
            )
        elif kappa_map_shape is not None and len(kappa_map_shape) != 2:
            raise ValueError(
                f"kappa_map_shape must be 2D (received {len(kappa_map_shape)}D)"
            )

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
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
        self.Psi_kernel = safe_log(d2.sqrt())
        self.ax_kernel = safe_divide(x_mg, d2)
        self.ay_kernel = safe_divide(y_mg, d2)
        # Set centers of kernels to zero
        self.Psi_kernel[..., self.n_pix, self.n_pix] = 0
        self.ax_kernel[..., self.n_pix, self.n_pix] = 0
        self.ay_kernel[..., self.n_pix, self.n_pix] = 0

        self.Psi_kernel_tilde = None
        self.ax_kernel_tilde = None
        self.ay_kernel_tilde = None
        self._s = None

        # Triggers creation of FFTs of kernels
        self.mode = mode

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """
        Move the KappaGrid object and all its tensors to the specified device and dtype.

        Args:
            device (Optional[torch.device]): The target device to move the tensors to.
            dtype (Optional[torch.dtype]): The target data type to cast the tensors to.
        """        
        super().to(device, dtype)
        self.Psi_kernel = self.Psi_kernel.to(device=device, dtype=dtype)
        self.ax_kernel = self.ax_kernel.to(device=device, dtype=dtype)
        self.ay_kernel = self.ay_kernel.to(device=device, dtype=dtype)
        if self.Psi_kernel_tilde is not None:
            self.Psi_kernel_tilde = self.Psi_kernel_tilde.to(device=device)
        if self.ax_kernel_tilde is not None:
            self.ax_kernel_tilde = self.ax_kernel_tilde.to(device=device)
        if self.ay_kernel_tilde is not None:
            self.ay_kernel_tilde = self.ay_kernel_tilde.to(device=device)

    def _fft2_padded(self, x: Tensor) -> Tensor:
        """
        Compute the 2D Fast Fourier Transform (FFT) of a tensor with zero-padding.

        Args:
            x (Tensor): The input tensor to be transformed.

        Returns:
            Tensor: The 2D FFT of the input tensor with zero-padding.
        """
        pad = 2 * self.n_pix
        if self.use_next_fast_len:
            pad = next_fast_len(pad)
        self._s = (pad, pad)
        return torch.fft.rfft2(x, self._s)

    def _unpad_fft(self, x: Tensor) -> Tensor:
        """
        Remove padding from the result of a 2D FFT.

        Args:
            x (Tensor): The input tensor with padding.

        Returns:
            Tensor: The input tensor without padding.
        """
        return x[..., -self.n_pix :, -self.n_pix :]

    def _unpad_conv2d(self, x: Tensor) -> Tensor:
        """
        Remove padding from the result of a 2D convolution.

        Args:
            x (Tensor): The input tensor with padding.

        Returns:
            Tensor: The input tensor without padding.
        """
        return x[..., 1:, 1:]

    @property
    def mode(self):
        """
        Get the convolution mode of the KappaGrid object.

        Returns:
            str: The convolution mode, either "fft" or "conv2d".
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        """
        Set the convolution mode of the KappaGrid object.

        Args:
            mode (str): The convolution mode to be set, either "fft" or "conv2d".
        """
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

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the deflection angles at the specified positions using the given kappa map.

        Args:
            thx (Tensor): The x-coordinates of the positions to compute the deflection angles for.
            thy (Tensor): The y-coordinates of the positions to compute the deflection angles for.
            z_s (Tensor): The source redshift.
            x (Optional[dict[str, Any]]): A dictionary containing additional parameters.

        Returns:
            tuple[Tensor, Tensor]: The x and y components of the deflection angles at the specified positions.
        """
        z_l, thx0, thy0, kappa_map = self.unpack(x)

        if self.mode == "fft":
            alpha_x_map, alpha_y_map = self._alpha_fft(kappa_map)
        else:
            alpha_x_map, alpha_y_map = self._alpha_conv2d(kappa_map)

        # Scale is distance from center of image to center of pixel on the edge
        scale = self.fov / 2
        alpha_x = interp2d(
            alpha_x_map, (thx - thx0).view(-1) / scale, (thy - thy0).view(-1) / scale
        ).reshape(thx.shape)
        alpha_y = interp2d(
            alpha_y_map, (thx - thx0).view(-1) / scale, (thy - thy0).view(-1) / scale
        ).reshape(thx.shape)
        return alpha_x, alpha_y

    def _alpha_fft(self, kappa_map: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the deflection angles using the Fast Fourier Transform (FFT) method.

        Args:
            kappa_map (Tensor): The 2D tensor representing the kappa map.

        Returns:
            tuple[Tensor, Tensor]: The x and y components of the deflection angles.
        """
        kappa_tilde = self._fft2_padded(kappa_map)
        alpha_x = torch.fft.irfft2(kappa_tilde * self.ax_kernel_tilde, self._s) * (
            self.res**2 / pi
        )
        alpha_y = torch.fft.irfft2(kappa_tilde * self.ay_kernel_tilde, self._s) * (
            self.res**2 / pi
        )
        return self._unpad_fft(alpha_x), self._unpad_fft(alpha_y)

    def _alpha_conv2d(self, kappa_map: Tensor) -> tuple[Tensor, Tensor]::
        """
        Compute the deflection angles using the 2D convolution method.

        Args:
            kappa_map (Tensor): The 2D tensor representing the kappa map.

        Returns:
            tuple[Tensor, Tensor]: The x and y components of the deflection angles.
        """
        # Use kappa_map as kernel since the kernel is twice as large. Flip since
        # we actually want the cross-correlation.
        kappa_map_flipped = kappa_map.flip((-1, -2))[None, None]
        alpha_x = F.conv2d(self.ax_kernel[None, None], kappa_map_flipped)[0, 0] * (
            self.res**2 / pi
        )
        alpha_y = F.conv2d(self.ay_kernel[None, None], kappa_map_flipped)[0, 0] * (
            self.res**2 / pi
        )
        return self._unpad_conv2d(alpha_x), self._unpad_conv2d(alpha_y)

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Compute the lensing potential at the specified positions using the given kappa map.

        Args:
        thx (Tensor): The x-coordinates of the positions to compute the lensing potential for.
        thy (Tensor): The y-coordinates of the positions to compute the lensing potential for.
        z_s (Tensor): The source redshift.
        x (Optional[dict[str, Any]]): A dictionary containing additional parameters.

        Returns:
            Tensor: The lensing potential at the specified positions.
        """
        z_l, thx0, thy0, kappa_map = self.unpack(x)

        if self.mode == "fft":
            Psi_map = self._Psi_fft(kappa_map)
        else:
            Psi_map = self._Psi_conv2d(kappa_map)

        # Scale is distance from center of image to center of pixel on the edge
        scale = self.fov / 2
        return interp2d(
            Psi_map, (thx - thx0).view(-1) / scale, (thy - thy0).view(-1) / scale
        ).reshape(thx.shape)

    def _Psi_fft(self, kappa_map: Tensor) -> Tensor:
        """
        Compute the lensing potential using the Fast Fourier Transform (FFT) method.
    
        Args:
            kappa_map (Tensor): The 2D tensor representing the kappa map.
    
        Returns:
            Tensor: The lensing potential.
        """
        kappa_tilde = self._fft2_padded(kappa_map)
        Psi = torch.fft.irfft2(kappa_tilde * self.Psi_kernel_tilde, self._s) * (
            self.res**2 / pi
        )
        return self._unpad_fft(Psi)

    def _Psi_conv2d(self, kappa_map: Tensor) -> Tensor:
        """
        Compute the lensing potential using the 2D convolution method.
    
        Args:
            kappa_map (Tensor): The 2D tensor representing the kappa map.
    
        Returns:
            Tensor: The lensing potential.
        """
        # Use kappa_map as kernel since the kernel is twice as large. Flip since
        # we actually want the cross-correlation.
        kappa_map_flipped = kappa_map.flip((-1, -2))[None, None]
        Psi = F.conv2d(self.Psi_kernel[None, None], kappa_map_flipped)[0, 0] * (
            self.res**2 / pi
        )
        return self._unpad_conv2d(Psi)

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Compute the convergence (kappa) at the specified positions. This method is not implemented.
    
        Args:
            thx (Tensor): The x-coordinates of the positions to compute the convergence for.
            thy (Tensor): The y-coordinates of the positions to compute the convergence for.
            z_s (Tensor): The source redshift.
            x (Optional[dict[str, Any]]): A dictionary containing additional parameters.
    
        Returns:
            Tensor: The convergence at the specified positions.
    
        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError()
