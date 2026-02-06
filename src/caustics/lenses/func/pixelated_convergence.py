from scipy.fft import next_fast_len

from ...backend_obj import backend
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
    x_kernel: ArrayLike
        The x-component of the kernel.

        *Unit: unitless*

    y_kernel: ArrayLike
        The y-component of the kernel.

        *Unit: unitless*

    """
    x_mg, y_mg = meshgrid(pixelscale, 2 * n_pix)

    d2 = x_mg**2 + y_mg**2
    potential_kernel = safe_log(backend.sqrt(d2))
    ax_kernel = safe_divide(x_mg, d2)
    ay_kernel = safe_divide(y_mg, d2)

    return ax_kernel, ay_kernel, potential_kernel


def build_window_pixelated_convergence(window, kernel_shape):
    """
    Window the kernel for stable FFT.

    Parameters
    ----------
    window: float
        The window to apply as a fraction of the image width. For example a
        window of 1/4 will set the kernel to start decreasing at 1/4 of the
        image width, and then linearly go to zero.

    kernel_shape: tuple
        The shape of the kernel to be windowed.

    Returns
    -------
    ArrayLike
        The window to multiply with the kernel.

    """
    x, y = backend.meshgrid(
        backend.linspace(-1, 1, kernel_shape[-1]),
        backend.linspace(-1, 1, kernel_shape[-2]),
        indexing="xy",
    )
    r = backend.sqrt(x**2 + y**2)
    return backend.clamp((1 - r) / window, 0, 1)


def _fft_size(n_pix):
    pad = 2 * n_pix
    pad = next_fast_len(pad)
    return pad, pad


def _fft2_padded(x, n_pix, padding: str):
    """
    Compute the 2D FFT of a tensor with padding.

    Parameters
    ----------
    x: ArrayLike
        The input tensor.

    padding: str
        The type of padding to use.

    Returns
    -------
    ArrayLike
        The 2D FFT of the input tensor.

    """

    if padding == "zero":
        pass
    elif padding in ["reflect", "circular"]:
        x = backend.pad(
            x[None, None], (0, n_pix - 1, 0, n_pix - 1), mode=padding
        ).squeeze()
    elif padding == "tile":
        x = backend.tile(x, (2, 2))
    else:
        raise ValueError(f"Invalid padding type: {padding}")

    return backend.fft.rfft2(x, _fft_size(n_pix))


def _unpad_fft(x, n_pix):
    """
    Unpad the FFT of a tensor.

    Parameters
    ----------
    x: ArrayLike
        The input tensor.

    Returns
    -------
    ArrayLike
        The unpaded FFT of the input tensor.

    """
    _s = _fft_size(n_pix)
    return backend.roll(x, (-_s[0] // 2, -_s[1] // 2), dims=(-2, -1))[..., : n_pix, : n_pix]  # fmt: skip


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

    convergence_map: ArrayLike
        The pixelated convergence map.

        *Unit: unitless*

    x: ArrayLike
        The x-coordinate in the lens plane at which to compute the deflection.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinate in the lens plane at which to compute the deflection.

        *Unit: arcsec*

    ax_kernel: ArrayLike
        The x-component of the kernel for convolution.

        *Unit: unitless*

    ay_kernel: ArrayLike
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
    _pixelscale_pi = pixelscale**2 / backend.pi
    kernels = backend.stack((ax_kernel, ay_kernel), dim=0)
    if convolution_mode == "fft":
        convergence_tilde = _fft2_padded(convergence_map, n_pix, padding)
        deflection_angles = (
            backend.fft.irfft2(convergence_tilde * kernels, _s).real * _pixelscale_pi
        )
        deflection_angle_maps = _unpad_fft(deflection_angles, n_pix)
    elif convolution_mode == "conv2d":
        convergence_map_flipped = backend.flip(convergence_map, (-1, -2))[None, None]
        # noqa: E501 F.pad(, ((pad - self.n_pix)//2, (pad - self.n_pix)//2, (pad - self.n_pix)//2, (pad - self.n_pix)//2), mode = self.padding_mode)
        deflection_angle_maps = (
            backend.conv2d(
                backend.unsqueeze(kernels, 1), convergence_map_flipped, padding="same"
            ).squeeze()
            * _pixelscale_pi
        )
        # noqa: E501 torch.roll(x, (-self.padding_range * self.ax_kernel.shape[0]//4,-self.padding_range * self.ax_kernel.shape[1]//4), dims = (-2,-1))[..., :self.n_pix, :self.n_pix] #[..., 1:, 1:]
    else:
        raise ValueError(f"Invalid convolution mode: {convolution_mode}")
    # Scale is distance from center of image to center of pixel on the edge
    scale = fov / 2
    _x_view_scale = backend.view(x - x0, -1) / scale
    _y_view_scale = backend.view(y - y0, -1) / scale
    deflection_angle_x = interp2d(
        deflection_angle_maps[0], _x_view_scale, _y_view_scale
    ).reshape(x.shape)
    deflection_angle_y = interp2d(
        deflection_angle_maps[1], _x_view_scale, _y_view_scale
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

    convergence_map: ArrayLike
        The pixelated convergence map.

        *Unit: unitless*

    x: ArrayLike
        The x-coordinate in the lens plane at which to compute the deflection.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinate in the lens plane at which to compute the deflection.

        *Unit: arcsec*

    potential_kernel: ArrayLike
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
        potential = backend.fft.irfft2(convergence_tilde * potential_kernel, _s) * (
            pixelscale**2 / backend.pi
        )
        potential_map = _unpad_fft(potential, n_pix)
    elif convolution_mode == "conv2d":
        convergence_map_flipped = backend.flip(convergence_map, (-1, -2))[None, None]
        potential_map = backend.conv2d(
            potential_kernel[None, None], convergence_map_flipped, padding="same"
        ).squeeze() * (pixelscale**2 / backend.pi)
    else:
        raise ValueError(f"Invalid convolution mode: {convolution_mode}")
    scale = fov / 2
    return interp2d(
        potential_map,
        backend.view(x - x0, -1) / scale,
        backend.view(y - y0, -1) / scale,
    ).reshape(x.shape)
