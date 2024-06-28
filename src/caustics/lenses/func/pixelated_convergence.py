import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len

from ...utils import safe_divide, safe_log, meshgrid, interp2d


def build_kernels_pixelated_convergence(pixelscale, n_pix):
    """
    Build the kernels for the pixelated convergence.

    Parameters
    ----------
    pixelscale: float
        The pixel scale of the convergence map.

        *Unit: arcsec/pixel*

    n_pix: int
        The number of pixels in the convergence map.

        *Unit: number*

    Returns
    -------
    x_kernel: Tensor
        The x-component of the kernel.

        *Unit: unitless*

    y_kernel: Tensor
        The y-component of the kernel.

        *Unit: unitless*

    """
    x_mg, y_mg = meshgrid(pixelscale, 2 * n_pix)
    # Shift to center kernels within pixel at index n_pix
    x_mg = x_mg - pixelscale / 2
    y_mg = y_mg - pixelscale / 2
    d2 = x_mg**2 + y_mg**2
    potential_kernel = safe_log(d2.sqrt())
    ax_kernel = safe_divide(x_mg, d2)
    ay_kernel = safe_divide(y_mg, d2)
    # Set centers of kernels to zero
    potential_kernel[..., n_pix, n_pix] = 0
    ax_kernel[..., n_pix, n_pix] = 0
    ay_kernel[..., n_pix, n_pix] = 0

    return ax_kernel, ay_kernel, potential_kernel


def _fft_size(n_pix):
    pad = 2 * n_pix
    pad = next_fast_len(pad)
    return pad, pad


def _fft2_padded(x, n_pix, padding: str):
    """
    Compute the 2D FFT of a tensor with padding.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    padding: str
        The type of padding to use.

    Returns
    -------
    Tensor
        The 2D FFT of the input tensor.

    """

    if padding == "zero":
        pass
    elif padding in ["reflect", "circular"]:
        x = F.pad(x[None, None], (0, n_pix - 1, 0, n_pix - 1), mode=padding).squeeze()
    elif padding == "tile":
        x = torch.tile(x, (2, 2))
    else:
        raise ValueError(f"Invalid padding type: {padding}")

    return torch.fft.rfft2(x, _fft_size(n_pix))


def _unpad_fft(x, n_pix):
    """
    Unpad the FFT of a tensor.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The unpaded FFT of the input tensor.

    """
    _s = _fft_size(n_pix)
    return torch.roll(x, (-_s[0] // 2, -_s[1] // 2), dims=(-2, -1))[..., : n_pix, : n_pix]  # fmt: skip


def reduced_deflection_angle_pixelated_convergence(
    x0,
    y0,
    convergence_map,
    x,
    y,
    ax_kernel,
    ay_kernel,
    pixelscale,
    fov,
    n_pix,
    padding,
    convolution_mode="fft",
):
    """
    Compute the reduced deflection angle for a pixelated convergence map. This
    follows from the basic formulas for deflection angle, namely that it is the
    convolution of the convergence with a unit vector pointing towards the
    origin. For more details see the Meneghetti lecture notes equation 2.32

    Parameters
    ----------
    x0: float
        The x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: float
        The y-coordinate of the center of the lens.

        *Unit: arcsec*

    convergence_map: Tensor
        The pixelated convergence map.

        *Unit: unitless*

    x: Tensor
        The x-coordinate in the lens plane at which to compute the deflection.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate in the lens plane at which to compute the deflection.

        *Unit: arcsec*

    ax_kernel: Tensor
        The x-component of the kernel for convolution.

        *Unit: unitless*

    ay_kernel: Tensor
        The y-component of the kernel for convolution.

        *Unit: unitless*

    pixelscale: float
        The pixel scale of the convergence map.

        *Unit: arcsec/pixel*

    fov: float
        The field of view of the convergence map.

        *Unit: arcsec*

    n_pix: int
        The number of pixels in the convergence map.

        *Unit: number*

    padding: str
        The type of padding to use. Either "zero", "reflect", "circular", or "tile".

    convolution_mode: str
        The mode of convolution to use. Either "fft" or "conv2d".
    """
    _s = _fft_size(n_pix)
    if convolution_mode == "fft":
        convergence_tilde = _fft2_padded(convergence_map, n_pix, padding)
        deflection_angle_x = torch.fft.irfft2(
            convergence_tilde * ax_kernel, _s
        ).real * (pixelscale**2 / torch.pi)
        deflection_angle_y = torch.fft.irfft2(
            convergence_tilde * ay_kernel, _s
        ).real * (pixelscale**2 / torch.pi)
        deflection_angle_x_map = _unpad_fft(deflection_angle_x, n_pix)
        deflection_angle_y_map = _unpad_fft(deflection_angle_y, n_pix)
    elif convolution_mode == "conv2d":
        convergence_map_flipped = convergence_map.flip((-1, -2))[None, None]
        # noqa: E501 F.pad(, ((pad - self.n_pix)//2, (pad - self.n_pix)//2, (pad - self.n_pix)//2, (pad - self.n_pix)//2), mode = self.padding_mode)
        deflection_angle_x_map = F.conv2d(
            ax_kernel[None, None], convergence_map_flipped, padding="same"
        ).squeeze() * (pixelscale**2 / torch.pi)
        deflection_angle_y_map = F.conv2d(
            ay_kernel[None, None], convergence_map_flipped, padding="same"
        ).squeeze() * (
            pixelscale**2 / torch.pi
        )  # noqa: E501 torch.roll(x, (-self.padding_range * self.ax_kernel.shape[0]//4,-self.padding_range * self.ax_kernel.shape[1]//4), dims = (-2,-1))[..., :self.n_pix, :self.n_pix] #[..., 1:, 1:]
    else:
        raise ValueError(f"Invalid convolution mode: {convolution_mode}")
    # Scale is distance from center of image to center of pixel on the edge
    scale = fov / 2
    deflection_angle_x = interp2d(
        deflection_angle_x_map, (x - x0).view(-1) / scale, (y - y0).view(-1) / scale
    ).reshape(x.shape)
    deflection_angle_y = interp2d(
        deflection_angle_y_map, (x - x0).view(-1) / scale, (y - y0).view(-1) / scale
    ).reshape(x.shape)
    return deflection_angle_x, deflection_angle_y


def potential_pixelated_convergence(
    x0,
    y0,
    convergence_map,
    x,
    y,
    potential_kernel,
    pixelscale,
    fov,
    n_pix,
    padding,
    convolution_mode="fft",
):
    """
    Compute the lensing potential for a pixelated convergence map. This follows
    from the basic formulas for potential, namely that it is the convolution of
    the convergence with the logarithm of a vector pointing towards the origin.
    For more details see the Meneghetti lecture notes equation 2.31

    Parameters
    ----------
    x0: float
        The x-coordinate of the center of the lens.

        *Unit: arcsec*

    y0: float
        The y-coordinate of the center of the lens.

        *Unit: arcsec*

    convergence_map: Tensor
        The pixelated convergence map.

        *Unit: unitless*

    x: Tensor
        The x-coordinate in the lens plane at which to compute the deflection.

        *Unit: arcsec*

    y: Tensor
        The y-coordinate in the lens plane at which to compute the deflection.

        *Unit: arcsec*

    potential_kernel: Tensor
        The kernel for convolution.

        *Unit: unitless*

    pixelscale: float
        The pixel scale of the convergence map.

        *Unit: arcsec/pixel*

    fov: float
        The field of view of the convergence map.

        *Unit: arcsec*

    n_pix: int
        The number of pixels in the convergence map.

        *Unit: number*

    padding: str
        The type of padding to use. Either "zero", "reflect", "circular", or "tile".

    convolution_mode: str
        The mode of convolution to use. Either "fft" or "conv2d".
    """
    _s = _fft_size(n_pix)
    if convolution_mode == "fft":
        convergence_tilde = _fft2_padded(convergence_map, n_pix, padding)
        potential = torch.fft.irfft2(convergence_tilde * potential_kernel, _s) * (
            pixelscale**2 / torch.pi
        )
        potential_map = _unpad_fft(potential, n_pix)
    elif convolution_mode == "conv2d":
        convergence_map_flipped = convergence_map.flip((-1, -2))[None, None]
        potential_map = F.conv2d(
            potential_kernel[None, None], convergence_map_flipped, padding="same"
        ).squeeze() * (pixelscale**2 / torch.pi)
    else:
        raise ValueError(f"Invalid convolution mode: {convolution_mode}")
    scale = fov / 2
    return interp2d(
        potential_map, (x - x0).view(-1) / scale, (y - y0).view(-1) / scale
    ).reshape(x.shape)
