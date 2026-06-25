# mypy: disable-error-code="misc", disable-error-code="attr-defined"
from math import pi, ceil
from typing import Callable, Optional, Tuple, Dict, Union, Any, Literal
from importlib import import_module
from functools import partial, lru_cache

from scipy.special import roots_legendre

from .constants import rad_to_deg, deg_to_rad
from .backend_obj import backend, ArrayLike


def _import_func_or_class(module_path: str) -> Any:
    """
    Import a function or class from a module path

    Parameters
    ----------
    module_path : str
        The module path to import from

    Returns
    -------
    Callable
        The imported function or class
    """
    module_name, name = module_path.rsplit(".", 1)
    mod = import_module(module_name)
    return getattr(mod, name)  # type: ignore


def _eval_expression(input_string: str) -> Union[int, float]:
    """
    Evaluates a string expression to create an integer or float

    Parameters
    ----------
    input_string : str
        The string expression to evaluate

    Returns
    -------
    Union[int, float]
        The result of the evaluation

    Raises
    ------
    NameError
        If a disallowed constant is used
    """
    # Allowed modules to use string evaluation
    allowed_names = {"pi": pi}
    # Compile the input string
    code = compile(input_string, "<string>", "eval")
    # Check for disallowed names
    for name in code.co_names:
        if name not in allowed_names:
            # Throw an error if a disallowed name is used
            raise NameError(f"Use of {name} not allowed")
    # Evaluate the input string without using builtins
    # for security
    return eval(code, {"__builtins__": {}}, allowed_names)


def flip_axis_ratio(q, phi):
    """
    Makes the value of 'q' positive, then swaps x and y axes if 'q' is larger than 1.

    Parameters
    ----------
    q: ArrayLike
        ArrayLike containing values to be processed.
    phi: ArrayLike
        ArrayLike containing the phi values for the orientation of the axes.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Tuple containing the processed 'q' and 'phi' ArrayLikes.
    """
    q = backend.abs(q)
    return backend.where(q > 1, 1 / q, q), backend.where(q > 1, phi + pi / 2, phi)


def translate_rotate(x, y, x0, y0, phi: Optional[ArrayLike] = None):
    """
    Translates and rotates the points (x, y) by subtracting (x0, y0)
    and applying rotation angle phi.

    Parameters
    ----------
    x: ArrayLike
        ArrayLike containing the x-coordinates.
    y: ArrayLike
        ArrayLike containing the y-coordinates.
    x0: ArrayLike
        ArrayLike containing the x-coordinate translation values.
    y0: ArrayLike
        ArrayLike containing the y-coordinate translation values.
    phi: Optional[ArrayLike], optional)
        ArrayLike containing the rotation angles. If None, no rotation is applied. Defaults to None.

    Returns
    -------
    Tuple: [ArrayLike, ArrayLike]
        Tuple containing the translated and rotated x and y coordinates.
    """
    xt = x - x0
    yt = y - y0

    if phi is not None:
        # Apply R(-phi)
        c_phi = backend.cos(phi)
        s_phi = backend.sin(phi)
        # Simultaneous assignment
        return xt * c_phi + yt * s_phi, yt * c_phi - xt * s_phi  # fmt: skip

    return xt, yt


def derotate(vx, vy, phi: Optional[ArrayLike] = None):
    """
    Applies inverse rotation to the velocity components (vx, vy) using the rotation angle phi.

    Parameters
    ----------
    vx: ArrayLike
        ArrayLike containing the x-component of velocity.
    vy: ArrayLike
        ArrayLike containing the y-component of velocity.
    phi: Optional[ArrayLike], optional)
        ArrayLike containing the rotation angles. If None, no rotation is applied. Defaults to None.

    Returns
    -------
    Tuple: [ArrayLike, ArrayLike]
        Tuple containing the derotated x and y components of velocity.
    """
    if phi is None:
        return vx, vy

    c_phi = backend.cos(phi)
    s_phi = backend.sin(phi)
    return vx * c_phi - vy * s_phi, vx * s_phi + vy * c_phi  # fmt: skip


def to_elliptical(x, y, q: ArrayLike):
    """
    Converts Cartesian coordinates to elliptical coordinates.

    Parameters
    ----------
    x: ArrayLike
        ArrayLike containing the x-coordinates.
    y: ArrayLike
        ArrayLike containing the y-coordinates.
    q: ArrayLike
        ArrayLike containing the elliptical parameters.

    Returns
    -------
    Tuple: ArrayLike, ArrayLike
        Tuple containing the x and y coordinates in elliptical form.
    """
    return x, y / q


def meshgrid(
    pixelscale, nx, ny=None, device=None, dtype=backend.float32
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Generates a 2D meshgrid based on the provided pixelscale and dimensions.

    Parameters
    ----------
    pixelscale: float
        The scale of the meshgrid in each dimension.
    nx: int
        The number of grid points along the x-axis.
    ny: int
        The number of grid points along the y-axis.
    device: optional
        The device on which to create the tensor. Defaults to None.
    dtype: optional
        The desired data type of the tensor. Defaults to torch.float32.

    Returns
    -------
    Tuple: [ArrayLike, ArrayLike]
        The generated meshgrid as a tuple of ArrayLikes.
    """
    if ny is None:
        ny = nx
    xs = backend.linspace(-1, 1, nx, device=device, dtype=dtype) * pixelscale * (nx - 1) / 2  # fmt: skip
    ys = backend.linspace(-1, 1, ny, device=device, dtype=dtype) * pixelscale * (ny - 1) / 2  # fmt: skip
    return backend.meshgrid([xs, ys], indexing="xy")


def plane_to_world_gnomonic(px, py, crval):
    """
    Perform a gnomonic projection from a tangent plane to the celestial sphere
    world coordinates.

    Parameters
    ----------
    px: ArrayLike
        The x-coordinate of the point on the tangent plane in degrees.
    py: ArrayLike
        The y-coordinate of the point on the tangent plane in degrees.
    crval: ArrayLike
        The celestial sphere world coordinates in degrees where the tangent
        plane meets the celestial sphere, should be a shape (2,) tensor. It is
        assumed that the tangent plane is centered at (0,0) for these
        coordinates. Thus ``crval`` matches the standard FITS convention.

    Returns
    -------
    Tuple: [ArrayLike, ArrayLike]
        Tuple containing the right ascension and declination in degrees.
    """
    plane = backend.stack((px, py), -1) * deg_to_rad
    rho = backend.sqrt(backend.sum(plane**2, dim=-1))
    c = backend.arctan(rho)

    # Convert to sky coordinates
    ra = crval[0] + rad_to_deg * backend.arctan2(
        plane[..., 0] * backend.sin(c),
        rho * backend.cos(crval[1] * deg_to_rad) * backend.cos(c)
        - plane[..., 1] * backend.sin(crval[1] * deg_to_rad) * backend.sin(c),
    )

    dec = backend.where(
        rho == 0,
        crval[1],
        rad_to_deg
        * backend.arcsin(
            backend.cos(c) * backend.sin(crval[1] * deg_to_rad)
            + plane[..., 1] * backend.sin(c) * backend.cos(crval[1] * deg_to_rad) / rho
        ),
    )
    return ra, dec


def pixel_to_plane(i, j, crpix, CD, sip_powers=[], sip_coefs=[], crplane=None):
    """
    Convert pixel coordinates to a tangent plane using the WCS information. This
    matches the FITS convention for SIP transformations.

    For more information see:

    * FITS World Coordinate System (WCS):
      https://fits.gsfc.nasa.gov/fits_wcs.html
    * Representations of world coordinates in FITS, 2002, by Geisen and
      Calabretta
    * The SIP Convention for Representing Distortion in FITS Image Headers,
      2008, by Shupe and Hook

    Parameters
    ----------
    i: ArrayLike
        The first coordinate of the pixel in pixel units. The origin may be
        either 0 indexed (python convention) or 1 indexed (FITS convention),
        simply ensure that ``crpix`` has the same convention.
    j: ArrayLike
        The second coordinate of the pixel in pixel units. The origin may be
        either 0 indexed (python convention) or 1 indexed (FITS convention),
        simply ensure that ``crpix`` has the same convention.
    crpix: ArrayLike
        The reference pixel in pixel units, should be a shape (2,) tensor. This
        is the point that will be placed at ``crval`` in the world coordinates.
        The origin may be either 0 indexed (python convention) or 1 indexed
        (FITS convention), simply ensure that ``i`` and ``j`` have the same
        convention.
    CD: ArrayLike
        The CD matrix in degrees per pixel. This 2x2 matrix is used to convert
        from pixel to degree units and also handles rotation/skew.
    sip_powers: ArrayLike
        The powers of the pixel coordinates for the SIP distortion, should be a
        shape (N orders, 2) tensor. ``N orders`` is the number of non-zero
        polynomial coefficients. The second axis has the powers in order ``i,
        j``.
    sip_coefs: ArrayLike
        The coefficients of the pixel coordinates for the SIP distortion, should
        be a shape (N orders, 2) tensor. ``N orders`` is the number of non-zero
        polynomial coefficients. The second axis has the coefficients in order
        ``delta_x, delta_y``.
    crplane: Optional[ArrayLike], optional
        The reference plane coordinates in degrees, should be a shape (2,)
        tensor. This is the point that will be placed at ``crpix`` in the pixel
        coordinates. If None, it is assumed to be (0, 0). Defaults to None.

    Note
    ----
    The representation of the SIP powers and coefficients assumes that the SIP
    polynomial will use the same orders for both the x and y coordinates. If
    this is not the case you may use zeros for the coefficients to ensure all
    polynomial combinations are evaluated. However, it is very common to have
    the same orders for both.

    Note
    ----
    While it is not perfect, an approximate inverse for the SIP distortion can
    be determined by taking the negative of the coefficients (and using the
    ``plane_to_pixel`` function).

    Returns
    -------
    Tuple: [ArrayLike, ArrayLike]
        Tuple containing the x and y tangent plane coordinates in degrees.
    """
    if crplane is None:
        crplane = backend.zeros_like(crpix)

    pixel = backend.stack((i, j), -1) - crpix
    delta_p = backend.zeros_like(pixel)
    for p in range(len(sip_powers)):
        delta_p += sip_coefs[p] * backend.unsqueeze(
            backend.prod(pixel ** sip_powers[p], dim=-1), -1
        )
    plane = backend.einsum("ij,...j->...i", CD, pixel + delta_p) + crplane
    return plane[..., 0], plane[..., 1]


def pixel_to_world(
    i,
    j,
    crpix,
    crval,
    CD,
    sip_powers=[],
    sip_coefs=[],
    crplane=None,
):
    """
    Convert pixel coordinates to world coordinates using the WCS information.
    This matches the FITS convention for SIP transformations.

    For more information see:

    * FITS World Coordinate System (WCS):
      https://fits.gsfc.nasa.gov/fits_wcs.html
    * Representations of world coordinates in FITS, 2002, by Geisen and
      Calabretta
    * The SIP Convention for Representing Distortion in FITS Image Headers,
      2008, by Shupe and Hook

    Parameters
    ----------
    i: ArrayLike
        The first coordinate of the pixel in pixel units. The origin may be either 0
        indexed (python convention) or 1 indexed (FITS convention), simply
        ensure that ``crpix`` has the same convention.
    j: ArrayLike
        The second coordinate of the pixel in pixel units. The origin may be either 0
        indexed (python convention) or 1 indexed (FITS convention), simply
        ensure that ``crpix`` has the same convention.
    crpix: ArrayLike
        The reference pixel in pixel units, should be a shape (2,) tensor. This
        is the point that will be placed at ``crval`` in the world coordinates.
        The origin may be either 0 indexed (python convention) or 1 indexed
        (FITS convention), simply ensure that ``i`` and ``j`` have the same
        convention.
    crval: ArrayLike
        The reference world coordinates in degrees, should be a shape (2,)
        tensor. This is the point that will be placed at ``crpix`` in the pixel
        coordinates.
    CD: ArrayLike
        The CD matrix in degrees per pixel. This 2x2 matrix is used to convert
        from pixel to world units and also handles rotation/skew.
    sip_powers: ArrayLike
        The powers of the pixel coordinates for the SIP distortion, should be a
        shape (N orders, 2) tensor. ``N orders`` is the number of non-zero
        polynomial coefficients. The second axis has the powers in order ``i,
        j``.
    sip_coefs: ArrayLike
        The coefficients of the pixel coordinates for the SIP distortion, should
        be a shape (N orders, 2) tensor. ``N orders`` is the number of non-zero
        polynomial coefficients. The second axis has the coefficients in order
        ``delta_x, delta_y``.
    crplane: Optional[ArrayLike], optional
        The reference plane coordinates in degrees, should be a shape (2,)
        tensor. This is the point that will be placed at ``crpix`` in the pixel
        coordinates. If None, it is assumed to be (0, 0). Defaults to None.

    Note
    ----
    The representation of the SIP powers and coefficients assumes that the SIP
    polynomial will use the same orders for both the x and y coordinates. If
    this is not the case you may use zeros for the coefficients to ensure all
    polynomial combinations are evaluated. However, it is very common to have
    the same orders for both.

    Note
    ----
    While it is not perfect, an approximate inverse for the SIP distortion can
    be determined by taking the negative of the coefficients (and using the
    ``world_to_pixel`` function).

    Returns
    -------
    Tuple: [ArrayLike, ArrayLike]
        Tuple containing the right ascension and declination in degrees.
    """

    px, py = pixel_to_plane(i, j, crpix, CD, sip_powers, sip_coefs, crplane)
    ra, dec = plane_to_world_gnomonic(px, py, crval)
    return ra, dec


def world_to_plane_gnomonic(ra, dec, crval):
    """
    Perform a gnomonic projection from the celestial sphere
    world coordinates to a tangent plane.

    Parameters
    ----------
    ra: ArrayLike
        The right ascension in degrees.
    dec: ArrayLike
        The declination in degrees.
    crval: ArrayLike
        The celestial sphere world coordinates in degrees where the tangent
        plane meets the celestial sphere, should be a shape (2,) tensor. It is
        assumed that the tangent plane is centered at (0,0) for these
        coordinates. Thus ``crval`` matches the standard FITS convention.

    Returns
    -------
    Tuple: [ArrayLike, ArrayLike]
        Tuple containing the x and y tangent plane coordinates in degrees.
    """
    ra = ra * deg_to_rad
    dec = dec * deg_to_rad

    cosc = backend.sin(crval[1] * deg_to_rad) * backend.sin(dec) + backend.cos(
        crval[1] * deg_to_rad
    ) * backend.cos(dec) * backend.cos(ra - crval[0] * deg_to_rad)

    x = backend.cos(dec) * backend.sin(ra - crval[0] * deg_to_rad) / cosc

    y = (
        backend.cos(crval[1] * deg_to_rad) * backend.sin(dec)
        - backend.sin(crval[1] * deg_to_rad)
        * backend.cos(dec)
        * backend.cos(ra - crval[0] * deg_to_rad)
    ) / cosc

    return x * rad_to_deg, y * rad_to_deg


def plane_to_pixel(px, py, crpix, CD, sip_powers=[], sip_coefs=[], crplane=None):
    """
    Convert tangent plane coordinates to pixel coordinates using the WCS
    information. This matches the FITS convention for SIP transformations.

    For more information see:

    * FITS World Coordinate System (WCS):
      https://fits.gsfc.nasa.gov/fits_wcs.html
    * Representations of world coordinates in FITS, 2002, by Geisen and
      Calabretta
    * The SIP Convention for Representing Distortion in FITS Image Headers,
      2008, by Shupe and Hook

    Parameters
    ----------
    px: ArrayLike
        The x-coordinate of the point on the tangent plane in degrees.
    py: ArrayLike
        The y-coordinate of the point on the tangent plane in degrees.
    crpix: ArrayLike
        The reference pixel in pixel units, should be a shape (2,) tensor. This
        is the point that will be placed at ``crval`` in the world coordinates.
        The origin may be either 0 indexed (python convention) or 1 indexed
        (FITS convention), ``i`` and ``j`` will have the same convention.
    CD: ArrayLike
        The CD matrix in degrees per pixel. This 2x2 matrix is used to convert
        from pixel to world units and also handles rotation/skew.
    sip_powers: ArrayLike
        The powers of the pixel coordinates for the SIP distortion, should be a
        shape (N orders, 2) tensor. ``N orders`` is the number of non-zero
        polynomial coefficients. The second axis has the powers in order ``px,
        py``.
    sip_coefs: ArrayLike
        The coefficients of the pixel coordinates for the SIP distortion, should
        be a shape (N orders, 2) tensor. ``N orders`` is the number of non-zero
        polynomial coefficients. The second axis has the coefficients in order
        ``delta_x, delta_y``.
    crplane: Optional[ArrayLike], optional
        The reference plane coordinates in degrees, should be a shape (2,)
        tensor. This is the point that will be placed at ``crpix`` in the pixel
        coordinates. If None, it is assumed to be (0, 0). Defaults to None.

    Note
    ----
    The representation of the SIP powers and coefficients assumes that the SIP
    polynomial will use the same orders for both the x and y coordinates. If
    this is not the case you may use zeros for the coefficients to ensure all
    polynomial combinations are evaluated. However, it is very common to have
    the same orders for both.

    Note
    ----
    While it is not perfect, an approximate inverse for the SIP distortion can
    be determined by taking the negative of the coefficients (and using the
    ``pixel_to_plane`` function).

    Returns
    -------
    Tuple: [ArrayLike, ArrayLike]
        Tuple containing the ``i`` and ``j`` pixel coordinates (in pixel units).

    """
    if crplane is None:
        crplane = backend.zeros_like(crpix)

    plane = backend.stack((px, py), -1) - crplane
    iCD = backend.linalg.inv(CD)
    pixel = backend.einsum("ij,...j->...i", iCD, plane)
    delta_w = backend.zeros_like(plane)
    for i in range(len(sip_powers)):
        delta_w += sip_coefs[i] * backend.unsqueeze(
            backend.prod(pixel ** sip_powers[i], dim=-1), -1
        )
    pixel += delta_w + crpix
    return pixel[..., 0], pixel[..., 1]


def world_to_pixel(
    ra,
    dec,
    crpix,
    crval,
    CD,
    sip_powers=[],
    sip_coefs=[],
    crplane=None,
):
    """
    Convert world coordinates to pixel coordinates using the WCS information.
    This matches the FITS convention for SIP transformations.

    For more information see:

    * FITS World Coordinate System (WCS):
      https://fits.gsfc.nasa.gov/fits_wcs.html
    * Representations of world coordinates in FITS, 2002, by Geisen and
      Calabretta
    * The SIP Convention for Representing Distortion in FITS Image Headers,
      2008, by Shupe and Hook

    Parameters
    ----------
    ra: ArrayLike
        The right ascension in degrees.
    dec: ArrayLike
        The declination in degrees.
    crpix: ArrayLike
        The reference pixel in pixel units, should be a shape (2,) tensor. This
        is the point that will be placed at ``crval`` in the world coordinates.
        The origin may be either 0 indexed (python convention) or 1 indexed
        (FITS convention), ``i`` and ``j`` will have the same convention.
    crval: ArrayLike
        The reference world coordinates in degrees, should be a shape (2,)
        tensor. This is the point that will be placed at ``crpix`` in the pixel
        coordinates (unless ``crplane`` is non-zero).
    CD: ArrayLike
        The CD matrix in degrees per pixel. This 2x2 matrix is used to convert
        from pixel to world units and also handles rotation/skew.
    powers: ArrayLike
        The powers of the pixel coordinates for the SIP distortion, should be a
        shape (N orders, 2) tensor. ``N orders`` is the number of non-zero
        polynomial coefficients. The second axis has the powers in order ``i,
        j``.
    coefs: ArrayLike
        The coefficients of the pixel coordinates for the SIP distortion, should
        be a shape (N orders, 2) tensor. ``N orders`` is the number of non-zero
        polynomial coefficients. The second axis has the coefficients in order
        ``delta_x, delta_y``.

    Note
    ----
    The representation of the SIP powers and coefficients assumes that the SIP
    polynomial will use the same orders for both the x and y coordinates. If
    this is not the case you may use zeros for the coefficients to ensure all
    polynomial combinations are evaluated. However, it is very common to have
    the same orders for both.

    Note
    ----
    While it is not perfect, an approximate inverse for the SIP distortion can
    be determined by taking the negative of the coefficients (and using the
    ``pixel_to_world`` function).

    Returns
    -------
    Tuple: [ArrayLike, ArrayLike]
        Tuple containing the x and y pixel coordinates (in pixels).
    """
    px, py = world_to_plane_gnomonic(ra, dec, crval)
    i, j = plane_to_pixel(px, py, crpix, CD, sip_powers, sip_coefs, crplane)
    return i, j


@lru_cache(maxsize=32)
def _quad_table(n, p, dtype, device):
    """
    Generate a meshgrid for quadrature points using Legendre-Gauss quadrature.

    Parameters
    ----------
    n : int
        The number of quadrature points in each dimension.
    p : torch.ArrayLike
        The pixelscale.
    dtype : torch.dtype
        The desired data type of the tensor.
    device : torch.device
        The device on which to create the tensor.

    Returns
    -------
    Tuple[torch.ArrayLike, torch.ArrayLike, torch.ArrayLike]
        The generated meshgrid as a tuple of ArrayLikes.
    """
    abscissa, weights = roots_legendre(n)

    w = backend.as_array(weights, dtype=dtype, device=device)
    a = p * backend.as_array(abscissa, dtype=dtype, device=device) / 2.0
    X, Y = backend.meshgrid(a, a, indexing="xy")

    W = backend.outer(w, w) / 4.0

    X, Y = X.reshape(-1), Y.reshape(-1)  # flatten
    return X, Y, W.reshape(-1)


def gaussian_quadrature_grid(
    pixelscale,
    X,
    Y,
    quad_level=3,
):
    """
    Generates a 2D meshgrid for Gaussian quadrature based on the provided pixelscale and dimensions.

    Parameters
    ----------
    pixelscale : float
        The scale of the meshgrid in each dimension.
    X : ArrayLike
        The x-coordinates of the pixel centers.
    Y : ArrayLike
        The y-coordinates of the pixel centers.
    quad_level : int, optional
        The number of quadrature points in each dimension. Default is 3.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        The generated meshgrid as a tuple of ArrayLikes.

    Example
    -------
    Usage would look something like:: python

        X, Y = meshgrid(pixelscale, nx, ny)
        Xs, Ys, weight = gaussian_quadrature_grid(pixelscale, X, Y, quad_level)
        F = your_brightness_function(Xs, Ys, other, parameters)
        res = gaussian_quadrature_integrator(F, weight)
    """

    # collect gaussian quadrature weights
    abscissaX, abscissaY, weight = _quad_table(
        quad_level, pixelscale, dtype=X.dtype, device=backend.device(X)
    )

    # Gaussian quadrature evaluation points
    Xs = backend.repeat(X[..., None], quad_level**2, -1) + abscissaX
    Ys = backend.repeat(Y[..., None], quad_level**2, -1) + abscissaY
    return Xs, Ys, weight


def gaussian_quadrature_integrator(
    F: ArrayLike,
    weight: ArrayLike,
):
    """
    Performs a pixel-wise integration using Gaussian quadrature.
    It takes the brightness function evaluated at the quadrature points `F`
    and the quadrature weights `weight` as input.
    The result is the integrated brightness function at each pixel.


    Parameters
    ----------
    F : ArrayLike
        The brightness function evaluated at the quadrature points.
    weight : ArrayLike
        The quadrature weights as provided by the get_pixel_quad_integrator_grid function.

    Returns
    -------
    ArrayLike
        The integrated brightness function at each pixel.


    Example
    -------
    Usage would look something like:: python

        X, Y = meshgrid(pixelscale, nx, ny)
        Xs, Ys, weight = gaussian_quadrature_grid(pixelscale, X, Y, quad_level)
        F = your_brightness_function(Xs, Ys, other, parameters)
        res = gaussian_quadrature_integrator(F, weight)
    """

    return backend.sum(F * weight, dim=-1)


def quad(
    F: Callable,
    pixelscale: float,
    X: ArrayLike,
    Y: ArrayLike,
    args: Tuple = (),
    quad_level: int = 3,
):
    """
    Performs a pixel-wise integration on a function using Gaussian quadrature.

    Parameters
    ----------
    F : Callable
        The brightness function to be evaluated at the quadrature points.
        The function should take as input: F(X, Y, *args).
    pixelscale : float
        The scale of each pixel.
    X : ArrayLike
        The x-coordinates of the pixels.
    Y : ArrayLike
        The y-coordinates of the pixels.
    args : Optional[Tuple], optional
        Additional arguments to be passed to the brightness function, by default None.
    quad_level : int, optional
        The level of quadrature to use, by default 3.

    Returns
    -------
    ArrayLike
        The integrated brightness function at each pixel.
    """
    X, Y, weight = gaussian_quadrature_grid(pixelscale, X, Y, quad_level)
    F = F(X, Y, *args)
    return gaussian_quadrature_integrator(F, weight)


def safe_divide(num, denom):
    """
    Safely divides two tensors, returning zero where the denominator is zero.

    Parameters
    ----------
    num: ArrayLike
        The numerator tensor.
    denom: ArrayLike
        The denominator tensor.

    Returns
    -------
    ArrayLike
        The result of the division, with zero where the denominator was zero.
    """
    return backend.where(denom != 0, num / denom, backend.zeros_like(num))


def safe_log(x):
    """
    Safely applies the logarithm to a tensor, returning zero where the tensor is zero.

    Parameters
    ----------
    x: ArrayLike
        The input tensor.

    Returns
    -------
    ArrayLike
        The result of applying the logarithm, with zero where the input was zero.
    """
    return backend.where(x != 0, backend.log(x), backend.zeros_like(x))


def _h_poly(t):
    """Helper function to compute the 'h' polynomial matrix used in the
    cubic spline.

    Parameters
    ----------
    t: ArrayLike
        A 1D tensor representing the normalized x values.

    Returns
    -------
    ArrayLike
        A 2D tensor of size (4, len(t)) representing the 'h' polynomial matrix.

    """

    tt = t[None, :] ** (backend.arange(4, device=backend.device(t))[:, None])
    A = backend.as_array(
        [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
        dtype=t.dtype,
        device=backend.device(t),
    )
    return A @ tt


def interp1d(
    x: ArrayLike,
    y: ArrayLike,
    xs: ArrayLike,
    extend: Literal["extrapolate", "const", "linear"] = "extrapolate",
) -> ArrayLike:
    """Compute the 1D cubic spline interpolation for the given data points
    using PyTorch.

    Parameters
    ----------
    x: ArrayLike
        A 1D tensor representing the x-coordinates of the known data points.
    y: ArrayLike
        A 1D tensor representing the y-coordinates of the known data points.
    xs: ArrayLike
        A 1D tensor representing the x-coordinates of the positions where
        the cubic spline function should be evaluated.
    extend: (str, optional)
        The method for handling extrapolation, either "const", "extrapolate", or "linear".
        Default is "extrapolate".
        "const": Use the value of the last known data point for extrapolation.
        "linear": Use linear extrapolation based on the last two known data points.
        "extrapolate": Use cubic extrapolation of data.

    Returns
    -------
    ArrayLike
        A 1D tensor representing the interpolated values at the specified positions (xs).

    """
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = backend.cat([m[0:1], (m[1:] + m[:-1]) / 2, m[-1:]])
    idxs = backend.searchsorted(x[:-1], xs) - 1
    dx = x[idxs + 1] - x[idxs]
    hh = _h_poly((xs - x[idxs]) / dx)
    ret = hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx  # fmt: skip
    if extend == "const":
        ret[xs > x[-1]] = y[-1]
    elif extend == "linear":
        indices = xs > x[-1]
        ret[indices] = y[-1] + (xs[indices] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return ret


def interp2d(
    im: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    method: Literal["linear", "nearest"] = "linear",
    padding_mode: str = "zeros",
) -> ArrayLike:
    """
    Interpolates a 2D image at specified coordinates. Similar to
    `torch.nn.functional.grid_sample` with `align_corners=False`.

    Parameters
    ----------
    im: ArrayLike
        A 2D tensor representing the image.
    x: ArrayLike
        A 0D or 1D tensor of x coordinates at which to interpolate.
    y: ArrayLike
        A 0D or 1D tensor of y coordinates at which to interpolate.
    method: (str, optional)
        Interpolation method. Either 'nearest' or 'linear'. Defaults to
        'linear'.
    padding_mode:  (str, optional)
        Defines the padding mode when out-of-bound indices are encountered.
        Either 'zeros', 'clamp', or 'extrapolate'. Defaults to 'zeros' which
        fills padded coordinates with zeros. The 'clamp' mode clamps the
        coordinates to the image boundaries (essentially taking the border
        values out to infinity). The 'extrapolate' mode extrapolates the outer
        linear interpolation beyond the last pixel boundary.

    Raises
    ------
    ValueError
        If `im` is not a 2D tensor.
    ValueError
        If `x` is not a 0D or 1D tensor.
    ValueError
        If `y` is not a 0D or 1D tensor.
    ValueError
        If `padding_mode` is not 'extrapolate' or 'zeros'.
    ValueError
        If `method` is not 'nearest' or 'linear'.

    Returns
    -------
    ArrayLike
        ArrayLike with the same shape as `x` and `y` containing the interpolated
        values.
    """
    if im.ndim != 2:
        raise ValueError(f"im must be 2D (received {im.ndim}D tensor)")
    if x.ndim > 1:
        raise ValueError(f"x must be 0 or 1D (received {x.ndim}D tensor)")
    if y.ndim > 1:
        raise ValueError(f"y must be 0 or 1D (received {y.ndim}D tensor)")
    if padding_mode not in ["extrapolate", "clamp", "zeros"]:
        raise ValueError(f"{padding_mode} is not a valid padding mode")

    if padding_mode == "clamp":
        x = backend.clamp(x, -1, 1)
        y = backend.clamp(y, -1, 1)
    else:
        idxs_out_of_bounds = (y < -1) | (y > 1) | (x < -1) | (x > 1)

    # Convert coordinates to pixel indices
    h, w = im.shape
    x = 0.5 * ((x + 1) * w - 1)
    y = 0.5 * ((y + 1) * h - 1)

    if method == "nearest":
        result = im[
            backend.clamp(backend.long(backend.round(y)), 0, h - 1),
            backend.clamp(backend.long(backend.round(x)), 0, w - 1),
        ]
    elif method == "linear":
        x0 = backend.clamp(backend.long(backend.floor(x)), 0, w - 2)
        y0 = backend.clamp(backend.long(backend.floor(y)), 0, h - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        fa = im[y0, x0]
        fb = im[y1, x0]
        fc = im[y0, x1]
        fd = im[y1, x1]

        dx1 = x1 - x
        dx0 = x - x0
        dy1 = y1 - y
        dy0 = y - y0

        result = fa * dx1 * dy1 + fb * dx1 * dy0 + fc * dx0 * dy1 + fd * dx0 * dy0  # fmt: skip
    else:
        raise ValueError(f"{method} is not a valid interpolation method")

    if padding_mode == "zeros":  # else padding_mode == "extrapolate"
        result = backend.where(idxs_out_of_bounds, backend.zeros_like(result), result)

    return result


def interp3d(
    cu: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    t: ArrayLike,
    method: Literal["linear", "nearest"] = "linear",
    padding_mode: Literal["zeros", "extrapolate"] = "zeros",
) -> ArrayLike:
    """
    Interpolates a 3D image at specified coordinates.
    Similar to `torch.nn.functional.grid_sample` with `align_corners=False`.

    Parameters
    ----------
    cu: ArrayLike
        A 3D tensor representing the cube.
    x: ArrayLike
        A 0D or 1D tensor of x coordinates at which to interpolate.
    y: ArrayLike
        A 0D or 1D tensor of y coordinates at which to interpolate.
    t: ArrayLike
        A 0D or 1D tensor of t coordinates at which to interpolate.
    method: (str, optional)
        Interpolation method. Either 'nearest' or 'linear'. Defaults to 'linear'.
    padding_mode:  (str, optional)
        Defines the padding mode when out-of-bound indices are encountered.
        Either 'zeros' or 'extrapolate'. Defaults to 'zeros'.

    Raises
    ------
    ValueError
        If `cu` is not a 3D tensor.
    ValueError
        If `x` is not a 0D or 1D tensor.
    ValueError
        If `y` is not a 0D or 1D tensor.
    ValueError
        If `t` is not a 0D or 1D tensor.
    ValueError
        If `padding_mode` is not 'extrapolate' or 'zeros'.
    ValueError
        If `method` is not 'nearest' or 'linear'.

    Returns
    -------
    ArrayLike
        ArrayLike with the same shape as `x` and `y` containing the interpolated values.
    """
    if cu.ndim != 3:
        raise ValueError(f"im must be 3D (received {cu.ndim}D tensor)")

    if t.ndim > 1:
        raise ValueError(f"t must be 0 or 1D (received {t.ndim}D tensor)")
    if padding_mode not in ["extrapolate", "zeros"]:
        raise ValueError(f"{padding_mode} is not a valid padding mode")

    idxs_out_of_bounds = (y < -1) | (y > 1) | (x < -1) | (x > 1) | (t < -1) | (t > 1)
    # Convert coordinates to pixel indices
    d, h, w = cu.shape
    x = 0.5 * ((x + 1) * w - 1)
    y = 0.5 * ((y + 1) * h - 1)
    t = 0.5 * ((t + 1) * d - 1)

    if method == "nearest":
        result = cu[
            backend.clamp(backend.long(backend.round(t)), 0, d - 1),
            backend.clamp(backend.long(backend.round(y)), 0, h - 1),
            backend.clamp(backend.long(backend.round(x)), 0, w - 1),
        ]
    elif method == "linear":
        x0 = backend.clamp(backend.long(backend.floor(x)), 0, w - 2)
        y0 = backend.clamp(backend.long(backend.floor(y)), 0, h - 2)
        t0 = backend.clamp(backend.long(backend.floor(t)), 0, d - 2)
        x1 = x0 + 1
        y1 = y0 + 1
        t1 = t0 + 1

        fa = cu[t0, y0, x0]
        fb = cu[t0, y1, x0]
        fc = cu[t0, y0, x1]
        fd = cu[t0, y1, x1]
        fe = cu[t1, y0, x0]
        ff = cu[t1, y1, x0]
        fg = cu[t1, y0, x1]
        fh = cu[t1, y1, x1]

        xd = x - x0
        yd = y - y0
        td = t - t0

        c00 = fa * (1 - xd) + fc * xd
        c01 = fe * (1 - xd) + fg * xd
        c10 = fb * (1 - xd) + fd * xd
        c11 = ff * (1 - xd) + fh * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        result = c0 * (1 - td) + c1 * td
    else:
        raise ValueError(f"{method} is not a valid interpolation method")

    if padding_mode == "zeros":  # else padding_mode == "extrapolate"
        result = backend.where(idxs_out_of_bounds, backend.zeros_like(result), result)

    return result


# Bicubic interpolation coefficients
# These are the coefficients for the bicubic interpolation kernel.
# To quote numerical recipes:
#     The formulas that obtain the c’s from the function and derivative values
#     are just a complicated linear transformation, with coefficients which,
#     having been determined once in the mists of numerical history, can be
#     tabulated and forgotten
BC = (
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    (-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0),
    (2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1),
    (0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1),
    (-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0),
    (9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2),
    (-6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2),
    (2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0),
    (-6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1),
    (4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1),
)


def bicubic_kernels(Z, d1, d2):
    """
    This is just a quick script to compute the necessary derivatives using
    finite differences. This is not the most accurate way to compute the
    derivatives, but it is good enough for most purposes.
    """
    dZ1 = backend.zeros_like(Z)
    dZ2 = backend.zeros_like(Z)
    dZ12 = backend.zeros_like(Z)

    # First derivatives on first axis
    # df/dx = (f(x+h, y) - f(x-h, y)) / 2h
    dZ1 = backend.fill_at_indices(dZ1, slice(1, -1), (Z[:-2] - Z[2:]) / (2 * d1))
    dZ1 = backend.fill_at_indices(dZ1, 0, (Z[0] - Z[1]) / d1)
    dZ1 = backend.fill_at_indices(dZ1, -1, (Z[-2] - Z[-1]) / d1)
    # First derivatives on second axis
    # df/dy = (f(x,y+h) - f(x,y-h)) / h
    dZ2 = backend.fill_at_indices(
        dZ2, (slice(None), slice(1, -1)), (Z[:, :-2] - Z[:, 2:]) / (2 * d2)
    )
    dZ2 = backend.fill_at_indices(dZ2, (slice(None), 0), (Z[:, 0] - Z[:, 1]) / d2)
    dZ2 = backend.fill_at_indices(dZ2, (slice(None), -1), (Z[:, -2] - Z[:, -1]) / d2)

    # Second derivatives across both axes
    # d2f/dxdy = (f(x-h, y-k) - f(x-h, y+k) - f(x+h, y-k) + f(x+h, y+k)) / (4hk)
    dZ12 = backend.fill_at_indices(
        dZ12,
        (slice(1, -1), slice(1, -1)),
        (Z[:-2, :-2] - Z[:-2, 2:] - Z[2:, :-2] + Z[2:, 2:]) / (4 * d1 * d2),
    )
    return dZ1, dZ2, dZ12


def interp_bicubic(
    x,
    y,
    Z,
    dZ1=None,
    dZ2=None,
    dZ12=None,
    get_Y: bool = True,
    get_dY: bool = False,
    get_ddY: bool = False,
):
    """
    Compute bicubic interpolation of a 2D grid at arbitrary locations. This will
    smoothly interpolate a grid of points, including smooth first derivatives
    and smooth cross derivative (d^2Y/dxdy). For the derivatives, continuity is
    enforced, though the transition may be sharp as higher order derivatives are
    not considered.

    The interpolation requires knowing the values of the first derivative in
    each axis and the cross derivative. If these are not provided, they will be
    estimated using central differences. For this function, the derivatives
    should be provided in pixel units. The interpolation will be more accurate
    if an analytic value is available for the derivatives.

    See Numerical Recipes in C, Chapter 3 (specifically: "Higher Order for
    Smoothness: Bicubic Interpolation") for more details.

    Parameters
    ----------
    x : torch.ArrayLike
        x-coordinates of the points to interpolate. Must be a 0D or 1D tensor.
        It should be in (-1,1) fov units, meaning that -1 is the left edge of
        the left pixel, and 1 is the right edge of the right pixel.
    y : torch.ArrayLike
        y-coordinates of the points to interpolate. Must be a 0D or 1D tensor.
        It should be in (-1,1) fov units, meaning that -1 is the bottom edge of
        the bottom pixel, and 1 is the top edge of the top pixel.
    Z : torch.ArrayLike
        2D grid of values to interpolate. The first axis corresponds to the
        y-axis and the second axis to the x-axis. The values in Z correspond to
        pixel center values, so Z[0,0] is the value at the center of the bottom
        left corner pixel of the grid. The grid should be at least 2x2 so the
        bicubic interpolation can go between the values.
    dZ1 : torch.ArrayLike or None
        First derivative of Z along the x-axis. If None, it will be estimated
        using central differences. Note that the derivative should be computed
        in pixel units, meaning that the distance from one pixel to the next is
        considered "1" in these units.
    dZ2 : torch.ArrayLike or None
        First derivative of Z along the y-axis. If None, it will be estimated
        using central differences. Note that the derivative should be computed
        in pixel units, meaning that the distance from one pixel to the next is
        considered "1" in these units.
    dZ12 : torch.ArrayLike or None
        Second derivative of Z along both axes. If None, it will be estimated
        using central differences. Note that the derivative should be computed
        in pixel units, meaning that the distance from one pixel to the next is
        considered "1" in these units.
    get_Y : bool
        Whether to return the interpolated values. This will add the estimated Y
        values to the return tuple
    get_dY : bool
        Whether to return the interpolated first derivatives. This will add dY1
        and dY2 to the return tuple
    get_ddY : bool
        Whether to return the interpolated second derivatives. This will add
        dY12, dY11, and dY22 to the return tuple

    Returns
    -------
    Y : torch.ArrayLike or None
        Interpolated values at the given locations. Only returned if get_Y is
        True
    dY1 : torch.ArrayLike or None
        Interpolated first derivative along the x-axis. Only returned if get_dY
        is True
    dY2 : torch.ArrayLike or None
        Interpolated first derivative along the y-axis. Only returned if get_dY
        is True
    dY12 : torch.ArrayLike or None
        Interpolated second derivative along both axes. Only returned if get_ddY
        is True
    dY11 : torch.ArrayLike or None
        Interpolated second derivative along the x-axis. Only returned if
        get_ddY is True
    dY22 : torch.ArrayLike or None
        Interpolated second derivative along the y-axis. Only returned if
        get_ddY is True
    """

    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D (received {Z.ndim}D tensor)")

    if x.ndim > 1:
        raise ValueError(f"x must be 0 or 1D (received {x.ndim}D tensor)")
    if y.ndim > 1:
        raise ValueError(f"y must be 0 or 1D (received {y.ndim}D tensor)")

    # Convert coordinates to pixel indices
    h, w = Z.shape
    x = 0.5 * ((x + 1) * w - 1)
    x = backend.clamp(x, -0.5, w - 0.5)
    y = 0.5 * ((y + 1) * h - 1)
    y = backend.clamp(y, -0.5, h - 0.5)

    # Compute bicubic kernels if not provided
    if dZ1 is None or dZ2 is None or dZ12 is None:
        _dZ1, _dZ2, _dZ12 = bicubic_kernels(Z, 1.0, 1.0)
    if dZ1 is None:
        dZ1 = _dZ1
    if dZ2 is None:
        dZ2 = _dZ2
    if dZ12 is None:
        dZ12 = _dZ12

    # Extract pixel values
    x0 = backend.long(backend.floor(x))
    y0 = backend.long(backend.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = backend.clamp(x0, 0, w - 2)
    x1 = backend.clamp(x1, 1, w - 1)
    y0 = backend.clamp(y0, 0, h - 2)
    y1 = backend.clamp(y1, 1, h - 1)

    # Build interpolation vector
    v = []
    v.append(Z[y0, x0])
    v.append(Z[y0, x1])
    v.append(Z[y1, x1])
    v.append(Z[y1, x0])
    v.append(dZ1[y0, x0])
    v.append(dZ1[y0, x1])
    v.append(dZ1[y1, x1])
    v.append(dZ1[y1, x0])
    v.append(dZ2[y0, x0])
    v.append(dZ2[y0, x1])
    v.append(dZ2[y1, x1])
    v.append(dZ2[y1, x0])
    v.append(dZ12[y0, x0])
    v.append(dZ12[y0, x1])
    v.append(dZ12[y1, x1])
    v.append(dZ12[y1, x0])
    v = backend.stack(v, dim=-1)

    # Compute interpolation coefficients
    c = (
        backend.as_array(BC, dtype=v.dtype, device=backend.device(v))
        @ backend.unsqueeze(v, -1)
    ).reshape(-1, 4, 4)

    # Compute interpolated values
    return_interp = []
    t = backend.where(
        (x < 0), (x % 1) - 1, backend.where(x >= w - 1, x % 1 + 1, x % 1)
    )  # TODO: change to x - x0
    u = backend.where((y < 0), (y % 1) - 1, backend.where(y >= h - 1, y % 1 + 1, y % 1))
    if get_Y:
        Y = backend.zeros_like(x)
        for i in range(4):
            for j in range(4):
                Y = Y + c[:, i, j] * t**i * u**j
        return_interp.append(Y)
    if get_dY:
        dY1 = backend.zeros_like(x)
        dY2 = backend.zeros_like(x)
        for i in range(4):
            for j in range(4):
                if i > 0:
                    dY1 = dY1 + i * c[:, i, j] * t ** (i - 1) * u**j
                if j > 0:
                    dY2 = dY2 + j * c[:, i, j] * t**i * u ** (j - 1)
        return_interp.append(dY1)
        return_interp.append(dY2)
    if get_ddY:
        dY12 = backend.zeros_like(x)
        dY11 = backend.zeros_like(x)
        dY22 = backend.zeros_like(x)
        for i in range(4):
            for j in range(4):
                if i > 0 and j > 0:
                    dY12 = dY12 + i * j * c[:, i, j] * t ** (i - 1) * u ** (j - 1)
                if i > 1:
                    dY11 = dY11 + i * (i - 1) * c[:, i, j] * t ** (i - 2) * u**j
                if j > 1:
                    dY22 = dY22 + j * (j - 1) * c[:, i, j] * t**i * u ** (j - 2)
        return_interp.append(dY12)
        return_interp.append(dY11)
        return_interp.append(dY22)
    return tuple(return_interp)


def vmap_n(
    func: Callable,
    depth: int = 1,
    in_dims: Union[int, Tuple] = 0,
    out_dims: Union[int, Tuple[int, ...]] = 0,
    randomness: str = "error",
) -> Callable:
    """
    Transforms a function `depth` times using `torch.vmap` with the same arguments passed each time.
    Returns `func` transformed `depth` times by `vmap`, with the same arguments
    passed to `vmap` each time.

    Parameters
    ----------
    func: Callable
        The function to transform.
    depth: (int, optional)
        The number of times to apply `torch.vmap`. Defaults to 1.
    in_dims: (Union[int, Tuple], optional)
        The dimensions to vectorize over in the input. Defaults to 0.
    out_dims: (Union[int, Tuple[int, ...]], optional):
        The dimensions to vectorize over in the output. Defaults to 0.
    randomness: (str, optional)
        How to handle randomness. Defaults to 'error'.

    Raises
    ------
    ValueError
        If `depth` is less than 1.

    Returns
    -------
    Callable
        The transformed function.

    TODO: test.
    """
    if depth < 1:
        raise ValueError("vmap_n depth must be >= 1")

    vmapd_func = func
    for _ in range(depth):
        vmapd_func = backend.vmap(vmapd_func, in_dims, out_dims, randomness)

    return vmapd_func


def _chunk_input(x, k, in_dims, chunk_size):
    if isinstance(in_dims, tuple):
        if chunk_size is None:
            n_chunks = 1
        else:
            i = 0
            while in_dims[i] is None:
                i += 1
            B = x[i].shape[in_dims[i]]
            n_chunks = ceil(B / chunk_size)

        # Break data into chunks
        chunks = [[] for _ in range(n_chunks)]
        for subx, in_dim in zip(x, in_dims):
            if in_dim is None:
                subchunking = [subx] * n_chunks
            else:
                subchunking = backend.chunk(subx, n_chunks, dim=in_dim)
            for j, subchunk in enumerate(subchunking):
                chunks[j].append(subchunk)
    else:  # isinstance(in_dims, dict)
        if chunk_size is None:
            n_chunks = 1
        else:
            for key, value in in_dims.items():
                if value is not None:
                    B = k[key].shape[value]
                    n_chunks = ceil(B / chunk_size)
                    break

        # Break data into chunks
        chunks = [{} for _ in range(n_chunks)]
        for key, value in in_dims.items():
            if value is None:
                subchunking = [k[key]] * n_chunks
            else:
                subchunking = backend.chunk(k[key], n_chunks, dim=value)
            for j, subchunk in enumerate(subchunking):
                chunks[j][key] = subchunk
    return chunks


def vmap_reduce(
    func: Callable,
    reduce_func: Callable = lambda x: backend.sum(x, dim=0),
    chunk_size: Optional[int] = None,
    in_dims: Union[Tuple[int, ...], Dict[str, int]] = (0,),
    out_dims: Union[int, Tuple[int, ...]] = 0,
    **kwargs,
) -> Callable:
    """
    Applies `torch.vmap` to `func` and then reduces the output using
    `reduce_func` along the appropriate dimensions. This saves on memory
    management if the dimension being reduced can cause the intermediate tensor
    (before reduction) to be large.

    Note
    ----
    The chunking and reduction is only "one level deep". If the output of `func`
    is still large even after chunking, this function will not completely solve
    the problem. Essentially if the batch dimension divided by chunk_size is
    still larger than chunk_size, then you will still have a large intermediate
    tensor.

    Parameters
    ----------
    func: Callable
        The function to transform.
    reduce_func: Callable
        The function to reduce the output of `func`.
    in_dims: Tuple[int,...]
        The dimensions to vectorize over in the input.
    out_dims: Tuple[int,...]
        The dimension to stack the output over.
    chunk_size: (Optional[int])
        The size of the chunks to process. If None, the entire input is
        processed at once.
    kwargs: Dict
        Additional keyword arguments to pass to `torch.vmap`.

    Returns
    -------
    ArrayLike
        The reduced output.
    """
    if isinstance(in_dims, tuple):
        vfunc = backend.vmap(func, in_dims, **kwargs)
    else:  # isinstance(in_dims, dict)
        vfunc = backend.vmap(func, (in_dims,), **kwargs)

    def wrapped(*x, **k):
        # Determine chunks
        chunks = _chunk_input(x, k, in_dims, chunk_size)

        # Process and reduce the chunks
        if isinstance(in_dims, tuple):
            out = tuple(reduce_func(vfunc(*chunk)) for chunk in chunks)
        else:  # isinstance(in_dims, dict)
            out = tuple(reduce_func(vfunc(chunk)) for chunk in chunks)

        # Stack the output
        if isinstance(out_dims, int):
            out = backend.stack(out, dim=out_dims)
        else:
            out = tuple(
                backend.stack([o[i] for o in out], dim=d)
                for i, d in enumerate(out_dims)
            )

        # Reduce the output
        return reduce_func(out)

    return wrapped


def cluster_means(xs: ArrayLike, k: int, key=None):
    """
    Computes cluster means using the k-means++ initialization algorithm.

    Parameters
    ----------
    xs: ArrayLike
        A tensor of data points.
    k: int
        The number of clusters.
    key: Optional[Union[None, jax.random.key]]
        A jax.random.key if using the Jax backend.

    Returns
    -------
    ArrayLike
        A tensor of cluster means.
    """
    b = len(xs)
    mean_idxs = [
        int(backend.randint(high=b, size=(), device=backend.device(xs), key=key).item())
    ]
    means = [xs[mean_idxs[0]]]
    for _ in range(1, k):
        unselected_xs = backend.stack(
            [x for i, x in enumerate(xs) if i not in mean_idxs]
        )

        # Distances to all means
        d2s = backend.sum(
            (unselected_xs[:, None, :] - backend.stack(means)[None, :, :]) ** 2, -1
        )

        # Distances to closest mean
        d2s_closest = backend.as_array(
            [d2s[i, m] for i, m in enumerate(d2s.argmin(-1))]
        )

        # Add point furthest from closest mean as next mean
        new_idx = int(backend.argmax(d2s_closest).item())
        means.append(unselected_xs[new_idx])
        mean_idxs.append(new_idx)

    return backend.stack(means)


def _lm_step(f, X, Y, Cinv, L, Lup, Ldn, epsilon, L_min, L_max):
    # Forward
    fY = f(X)
    dY = Y - fY

    # Jacobian
    J = backend.jacfwd(f)(X)
    J = backend.to(J, dtype=X.dtype)
    if Cinv.ndim == 1:
        chi2 = backend.sum(dY**2 * Cinv, -1)
    else:
        chi2 = backend.sum(dY @ Cinv @ dY, -1)

    # Gradient
    if Cinv.ndim == 1:
        grad = J.T @ (dY * Cinv)
    else:
        grad = J.T @ Cinv @ dY

    # Hessian
    if Cinv.ndim == 1:
        hess = J.T @ (J * Cinv.reshape(-1, 1))
    else:
        hess = J.T @ Cinv @ J
    hess_perturb = L * backend.eye(hess.shape[0], device=backend.device(hess))
    hess = hess + hess_perturb

    # Step
    h = backend.linalg.solve(hess, grad)

    # New chi^2
    fYnew = f(X + h)
    dYnew = Y - fYnew
    if Cinv.ndim == 1:
        chi2_new = backend.sum(dYnew**2 * Cinv, -1)
    else:
        chi2_new = backend.sum(dYnew @ Cinv @ dYnew, -1)

    # Test
    expected_improvement = backend.dot(h, hess @ h) + 2 * backend.dot(h, grad)
    rho = (chi2 - chi2_new) / backend.abs(expected_improvement)  # fmt: skip

    # Update
    X = backend.where(rho >= epsilon, X + h, X)
    chi2 = backend.where(rho > epsilon, chi2_new, chi2)
    L = backend.clamp(backend.where(rho >= epsilon, L / Ldn, L * Lup), L_min, L_max)

    return X, L, chi2


def batch_lm(
    X,  # B, Din
    Y,  # B, Dout
    f,  # Din -> Dout
    C=None,  # B, Dout, Dout !or! B, Dout
    epsilon=1e-1,
    L=1e0,
    L_dn=11.0,
    L_up=9.0,
    max_iter=50,
    L_min=1e-9,
    L_max=1e9,
    stopping=1e-4,
    f_args=(),
    f_kwargs={},
):
    B, Din = X.shape
    B, Dout = Y.shape

    if len(X) != len(Y):
        raise ValueError("x and y must having matching batch dimension")

    if C is None:
        C = backend.ones_like(Y)
    if C.ndim == 2:
        Cinv = 1 / C
    else:
        Cinv = backend.linalg.inv(C)
    Cinv = backend.to(Cinv, dtype=X.dtype)

    v_lm_step = backend.vmap(
        partial(
            _lm_step,
            lambda x: f(x, *f_args, **f_kwargs),
            Lup=L_up,
            Ldn=L_dn,
            epsilon=epsilon,
            L_min=L_min,
            L_max=L_max,
        )
    )
    L = L * backend.ones(B, device=backend.device(X), dtype=X.dtype)
    for _ in range(max_iter):
        Xnew, L, C = v_lm_step(X, Y, Cinv, L)
        if (
            backend.all(backend.abs(Xnew - X) < stopping)
            and backend.sum(L < 1e-2).item() > B / 3
        ):
            break
        if backend.all(L >= L_max):
            break
        X = Xnew

    return X, L, C


def gaussian(pixelscale, nx, ny, sigma, upsample=1, dtype=backend.float32, device=None):
    X, Y = backend.meshgrid(
        backend.linspace(
            -(nx * upsample - 1) * pixelscale / 2,
            (nx * upsample - 1) * pixelscale / 2,
            nx * upsample,
            dtype=dtype,
            device=device,
        ),
        backend.linspace(
            -(ny * upsample - 1) * pixelscale / 2,
            (ny * upsample - 1) * pixelscale / 2,
            ny * upsample,
            dtype=dtype,
            device=device,
        ),
        indexing="xy",
    )

    Z = backend.exp(-0.5 * (X**2 + Y**2) / sigma**2)

    Z = backend.sum(Z.reshape(ny, upsample, nx, upsample), dim=(1, 3))

    return Z / backend.sum(Z)
