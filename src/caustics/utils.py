# mypy: disable-error-code="misc"
from math import pi
from typing import Callable, Optional, Tuple, Union, Any
from importlib import import_module
from functools import partial, lru_cache

import torch
from torch import Tensor
from torch.func import jacfwd
import numpy as np
from scipy.special import roots_legendre


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
    q: Tensor
        Tensor containing values to be processed.
    phi: Tensor
        Tensor containing the phi values for the orientation of the axes.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Tuple containing the processed 'q' and 'phi' Tensors.
    """
    q = q.abs()
    return torch.where(q > 1, 1 / q, q), torch.where(q > 1, phi + pi / 2, phi)


def translate_rotate(x, y, x0, y0, phi: Optional[Tensor] = None):
    """
    Translates and rotates the points (x, y) by subtracting (x0, y0)
    and applying rotation angle phi.

    Parameters
    ----------
    x: Tensor
        Tensor containing the x-coordinates.
    y: Tensor
        Tensor containing the y-coordinates.
    x0: Tensor
        Tensor containing the x-coordinate translation values.
    y0: Tensor
        Tensor containing the y-coordinate translation values.
    phi: Optional[Tensor], optional)
        Tensor containing the rotation angles. If None, no rotation is applied. Defaults to None.

    Returns
    -------
    Tuple: [Tensor, Tensor]
        Tuple containing the translated and rotated x and y coordinates.
    """
    xt = x - x0
    yt = y - y0

    if phi is not None:
        # Apply R(-phi)
        c_phi = phi.cos()
        s_phi = phi.sin()
        # Simultaneous assignment
        xt, yt = xt * c_phi + yt * s_phi, -xt * s_phi + yt * c_phi  # fmt: skip

    return xt, yt


def derotate(vx, vy, phi: Optional[Tensor] = None):
    """
    Applies inverse rotation to the velocity components (vx, vy) using the rotation angle phi.

    Parameters
    ----------
    vx: Tensor
        Tensor containing the x-component of velocity.
    vy: Tensor
        Tensor containing the y-component of velocity.
    phi: Optional[Tensor], optional)
        Tensor containing the rotation angles. If None, no rotation is applied. Defaults to None.

    Returns
    -------
    Tuple: [Tensor, Tensor]
        Tuple containing the derotated x and y components of velocity.
    """
    if phi is None:
        return vx, vy

    c_phi = phi.cos()
    s_phi = phi.sin()
    return vx * c_phi - vy * s_phi, vx * s_phi + vy * c_phi  # fmt: skip


def to_elliptical(x, y, q: Tensor):
    """
    Converts Cartesian coordinates to elliptical coordinates.

    Parameters
    ----------
    x: Tensor
        Tensor containing the x-coordinates.
    y: Tensor
        Tensor containing the y-coordinates.
    q: Tensor
        Tensor containing the elliptical parameters.

    Returns
    -------
    Tuple: Tensor, Tensor
        Tuple containing the x and y coordinates in elliptical form.
    """
    return x, y / q


def get_meshgrid(
    pixelscale, nx, ny, device=None, dtype=torch.float32
) -> Tuple[Tensor, Tensor]:
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
    device: torch.device, optional
        The device on which to create the tensor. Defaults to None.
    dtype: torch.dtype, optional
        The desired data type of the tensor. Defaults to torch.float32.

    Returns
    -------
    Tuple: [Tensor, Tensor]
        The generated meshgrid as a tuple of Tensors.
    """
    xs = torch.linspace(-1, 1, nx, device=device, dtype=dtype) * pixelscale * (nx - 1) / 2  # fmt: skip
    ys = torch.linspace(-1, 1, ny, device=device, dtype=dtype) * pixelscale * (ny - 1) / 2  # fmt: skip
    return torch.meshgrid([xs, ys], indexing="xy")


@lru_cache(maxsize=32)
def _quad_table(n, p, dtype, device):
    """
    Generate a meshgrid for quadrature points using Legendre-Gauss quadrature.

    Parameters
    ----------
    n : int
        The number of quadrature points in each dimension.
    p : torch.Tensor
        The pixelscale.
    dtype : torch.dtype
        The desired data type of the tensor.
    device : torch.device
        The device on which to create the tensor.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The generated meshgrid as a tuple of Tensors.
    """
    abscissa, weights = roots_legendre(n)

    w = torch.tensor(weights, dtype=dtype, device=device)
    a = p * torch.tensor(abscissa, dtype=dtype, device=device) / 2.0
    X, Y = torch.meshgrid(a, a, indexing="xy")

    W = torch.outer(w, w) / 4.0

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
    X : Tensor
        The x-coordinates of the pixel centers.
    Y : Tensor
        The y-coordinates of the pixel centers.
    quad_level : int, optional
        The number of quadrature points in each dimension. Default is 3.

    Returns
    -------
    Tuple[Tensor, Tensor]
        The generated meshgrid as a tuple of Tensors.

    Example
    -------
    Usage would look something like:: python

        X, Y = get_meshgrid(pixelscale, nx, ny)
        Xs, Ys, weight = gaussian_quadrature_grid(pixelscale, X, Y, quad_level)
        F = your_brightness_function(Xs, Ys, other, parameters)
        res = gaussian_quadrature_integrator(F, weight)
    """

    # collect gaussian quadrature weights
    abscissaX, abscissaY, weight = _quad_table(
        quad_level, pixelscale, dtype=X.dtype, device=X.device
    )

    # Gaussian quadrature evaluation points
    Xs = torch.repeat_interleave(X[..., None], quad_level**2, -1) + abscissaX
    Ys = torch.repeat_interleave(Y[..., None], quad_level**2, -1) + abscissaY
    return Xs, Ys, weight


def gaussian_quadrature_integrator(
    F: Tensor,
    weight: Tensor,
):
    """
    Performs a pixel-wise integration using Gaussian quadrature.
    It takes the brightness function evaluated at the quadrature points `F`
    and the quadrature weights `weight` as input.
    The result is the integrated brightness function at each pixel.


    Parameters
    ----------
    F : Tensor
        The brightness function evaluated at the quadrature points.
    weight : Tensor
        The quadrature weights as provided by the get_pixel_quad_integrator_grid function.

    Returns
    -------
    Tensor
        The integrated brightness function at each pixel.


    Example
    -------
    Usage would look something like:: python

        X, Y = get_meshgrid(pixelscale, nx, ny)
        Xs, Ys, weight = gaussian_quadrature_grid(pixelscale, X, Y, quad_level)
        F = your_brightness_function(Xs, Ys, other, parameters)
        res = gaussian_quadrature_integrator(F, weight)
    """

    return (F * weight).sum(axis=-1)


def quad(
    F: Callable,
    pixelscale: float,
    X: Tensor,
    Y: Tensor,
    args: Optional[Tuple] = None,
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
    X : Tensor
        The x-coordinates of the pixels.
    Y : Tensor
        The y-coordinates of the pixels.
    args : Optional[Tuple], optional
        Additional arguments to be passed to the brightness function, by default None.
    quad_level : int, optional
        The level of quadrature to use, by default 3.

    Returns
    -------
    Tensor
        The integrated brightness function at each pixel.
    """
    X, Y, weight = gaussian_quadrature_grid(pixelscale, X, Y, quad_level)
    F = F(X, Y, *args)
    return gaussian_quadrature_integrator(F, weight)


def safe_divide(num, denom, places=7):
    """
    Safely divides two tensors, returning zero where the denominator is zero.

    Parameters
    ----------
    num: Tensor
        The numerator tensor.
    denom: Tensor
        The denominator tensor.

    Returns
    -------
    Tensor
        The result of the division, with zero where the denominator was zero.
    """
    out = torch.zeros_like(num)
    where = denom != 0
    out[where] = num[where] / denom[where]
    return out


def safe_log(x):
    """
    Safely applies the logarithm to a tensor, returning zero where the tensor is zero.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The result of applying the logarithm, with zero where the input was zero.
    """
    out = torch.zeros_like(x)
    where = x != 0
    out[where] = x[where].log()
    return out


def _h_poly(t):
    """Helper function to compute the 'h' polynomial matrix used in the
    cubic spline.

    Parameters
    ----------
    t: Tensor
        A 1D tensor representing the normalized x values.

    Returns
    -------
    Tensor
        A 2D tensor of size (4, len(t)) representing the 'h' polynomial matrix.

    """

    tt = t[None, :] ** (torch.arange(4, device=t.device)[:, None])
    A = torch.tensor(
        [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
        dtype=t.dtype,
        device=t.device,
    )
    return A @ tt


def interp1d(x: Tensor, y: Tensor, xs: Tensor, extend: str = "extrapolate") -> Tensor:
    """Compute the 1D cubic spline interpolation for the given data points
    using PyTorch.

    Parameters
    ----------
    x: Tensor
        A 1D tensor representing the x-coordinates of the known data points.
    y: Tensor
        A 1D tensor representing the y-coordinates of the known data points.
    xs: Tensor
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
    Tensor
        A 1D tensor representing the interpolated values at the specified positions (xs).

    """
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[:-1], xs) - 1
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
    im: Tensor,
    x: Tensor,
    y: Tensor,
    method: str = "linear",
    padding_mode: str = "zeros",
) -> Tensor:
    """
    Interpolates a 2D image at specified coordinates.
    Similar to `torch.nn.functional.grid_sample` with `align_corners=False`.

    Parameters
    ----------
    im: Tensor
        A 2D tensor representing the image.
    x: Tensor
        A 0D or 1D tensor of x coordinates at which to interpolate.
    y: Tensor
        A 0D or 1D tensor of y coordinates at which to interpolate.
    method: (str, optional)
        Interpolation method. Either 'nearest' or 'linear'. Defaults to 'linear'.
    padding_mode:  (str, optional)
        Defines the padding mode when out-of-bound indices are encountered.
        Either 'zeros' or 'extrapolate'. Defaults to 'zeros'.

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
    Tensor
        Tensor with the same shape as `x` and `y` containing the interpolated values.
    """
    if im.ndim != 2:
        raise ValueError(f"im must be 2D (received {im.ndim}D tensor)")
    if x.ndim > 1:
        raise ValueError(f"x must be 0 or 1D (received {x.ndim}D tensor)")
    if y.ndim > 1:
        raise ValueError(f"y must be 0 or 1D (received {y.ndim}D tensor)")
    if padding_mode not in ["extrapolate", "zeros"]:
        raise ValueError(f"{padding_mode} is not a valid padding mode")

    idxs_out_of_bounds = (y < -1) | (y > 1) | (x < -1) | (x > 1)
    # Convert coordinates to pixel indices
    h, w = im.shape
    x = 0.5 * ((x + 1) * w - 1)
    y = 0.5 * ((y + 1) * h - 1)

    if method == "nearest":
        result = im[y.round().long().clamp(0, h - 1), x.round().long().clamp(0, w - 1)]
    elif method == "linear":
        x0 = x.floor().long()
        y0 = y.floor().long()
        x1 = x0 + 1
        y1 = y0 + 1
        x0 = x0.clamp(0, w - 2)
        x1 = x1.clamp(1, w - 1)
        y0 = y0.clamp(0, h - 2)
        y1 = y1.clamp(1, h - 1)

        fa = im[y0, x0]
        fb = im[y1, x0]
        fc = im[y0, x1]
        fd = im[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        result = fa * wa + fb * wb + fc * wc + fd * wd
    else:
        raise ValueError(f"{method} is not a valid interpolation method")

    if padding_mode == "zeros":  # else padding_mode == "extrapolate"
        result = torch.where(idxs_out_of_bounds, torch.zeros_like(result), result)

    return result


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
        vmapd_func = torch.vmap(vmapd_func, in_dims, out_dims, randomness)

    return vmapd_func


def get_cluster_means(xs: Tensor, k: int):
    """
    Computes cluster means using the k-means++ initialization algorithm.

    Parameters
    ----------
    xs: Tensor
        A tensor of data points.
    k: int
        The number of clusters.

    Returns
    -------
    Tensor
        A tensor of cluster means.
    """
    b = len(xs)
    mean_idxs = [int(torch.randint(high=b, size=(), device=xs.device).item())]
    means = [xs[mean_idxs[0]]]
    for _ in range(1, k):
        unselected_xs = torch.stack([x for i, x in enumerate(xs) if i not in mean_idxs])

        # Distances to all means
        d2s = ((unselected_xs[:, None, :] - torch.stack(means)[None, :, :]) ** 2).sum(
            -1
        )

        # Distances to closest mean
        d2s_closest = torch.tensor([d2s[i, m] for i, m in enumerate(d2s.argmin(-1))])

        # Add point furthest from closest mean as next mean
        new_idx = int(d2s_closest.argmax().item())
        means.append(unselected_xs[new_idx])
        mean_idxs.append(new_idx)

    return torch.stack(means)


def _lm_step(f, X, Y, Cinv, L, Lup, Ldn, epsilon, L_min, L_max):
    # Forward
    fY = f(X)
    dY = Y - fY

    # Jacobian
    J = jacfwd(f)(X)
    J = J.to(dtype=X.dtype)
    chi2 = (dY @ Cinv @ dY).sum(-1)

    # Gradient
    grad = J.T @ Cinv @ dY

    # Hessian
    hess = J.T @ Cinv @ J
    hess_perturb = L * (
        torch.diag(hess) + 0.1 * torch.eye(hess.shape[0], device=hess.device)
    )
    hess = hess + hess_perturb

    # Step
    h = torch.linalg.solve(hess, grad)

    # New chi^2
    fYnew = f(X + h)
    dYnew = Y - fYnew
    chi2_new = (dYnew @ Cinv @ dYnew).sum(-1)

    # Test
    rho = (chi2 - chi2_new) / torch.abs(h @ (L * torch.dot(torch.diag(hess), h) + grad))  # fmt: skip

    # Update
    X = torch.where(rho >= epsilon, X + h, X)
    chi2 = torch.where(rho > epsilon, chi2_new, chi2)
    L = torch.clamp(torch.where(rho >= epsilon, L / Ldn, L * Lup), L_min, L_max)

    return X, L, chi2


def batch_lm(
    X,  # B, Din
    Y,  # B, Dout
    f,  # Din -> Dout
    C=None,  # B, Dout, Dout
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
        C = torch.eye(Dout, device=X.device).repeat(B, 1, 1)
    Cinv = torch.linalg.inv(C)

    v_lm_step = torch.vmap(
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
    L = L * torch.ones(B, device=X.device)
    for _ in range(max_iter):
        Xnew, L, C = v_lm_step(X, Y, Cinv, L)
        if (
            torch.all((Xnew - X).abs() < stopping)
            and torch.sum(L < 1e-2).item() > B / 3
        ):
            break
        X = Xnew

    return X, L, C


def gaussian(pixelscale, nx, ny, sigma, upsample=1, dtype=torch.float32, device=None):
    X, Y = np.meshgrid(
        np.linspace(
            -(nx * upsample - 1) * pixelscale / 2,
            (nx * upsample - 1) * pixelscale / 2,
            nx * upsample,
        ),
        np.linspace(
            -(ny * upsample - 1) * pixelscale / 2,
            (ny * upsample - 1) * pixelscale / 2,
            ny * upsample,
        ),
        indexing="xy",
    )

    Z = np.exp(-0.5 * (X**2 + Y**2) / sigma**2)

    Z = Z.reshape(ny, upsample, nx, upsample).sum(axis=(1, 3))

    return torch.tensor(Z / np.sum(Z), dtype=dtype, device=device)
