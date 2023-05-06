from math import pi
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor


def flip_axis_ratio(q, phi):
    """
    Makes q positive, then swaps x and y axes if it's larger than 1.
    """
    q = q.abs()
    return torch.where(q > 1, 1 / q, q), torch.where(q > 1, phi + pi / 2, phi)


def translate_rotate(x, y, x0, y0, phi: Optional[Tensor] = None):
    xt = x - x0
    yt = y - y0

    if phi is not None:
        # Apply R(-phi)
        c_phi = phi.cos()
        s_phi = phi.sin()
        # Simultaneous assignment
        xt, yt = xt * c_phi + yt * s_phi, -xt * s_phi + yt * c_phi

    return xt, yt


def derotate(vx, vy, phi: Optional[Tensor] = None):
    if phi is None:
        return vx, vy

    c_phi = phi.cos()
    s_phi = phi.sin()
    return vx * c_phi - vy * s_phi, vx * s_phi + vy * c_phi


def to_elliptical(x, y, q: Tensor):
    """
    Converts to elliptical Cartesian coordinates.
    """
    return x * q.sqrt(), y / q.sqrt()


def get_meshgrid(
    resolution, nx, ny, device=None, dtype=torch.float32
) -> Tuple[Tensor, Tensor]:
    xs = (
        torch.linspace(-1, 1, nx, device=device, dtype=dtype)
        * resolution
        * (nx - 1)
        / 2
    )
    ys = (
        torch.linspace(-1, 1, ny, device=device, dtype=dtype)
        * resolution
        * (ny - 1)
        / 2
    )
    return torch.meshgrid([xs, ys], indexing="xy")


def safe_divide(num, denom):
    """
    Differentiable version of `torch.where(denom != 0, num/denom, 0.0)`.

    Returns:
        `num / denom` where `denom != 0`; zero everywhere else.
    """
    out = torch.zeros_like(num)
    where = denom != 0
    out[where] = num[where] / denom[where]
    return out


def safe_log(x):
    """
    Differentiable version of `torch.where(denom != 0, num/denom, 0.0)`.

    Returns:
        `num / denom` where `denom != 0`; zero everywhere else.
    """
    out = torch.zeros_like(x)
    where = x != 0
    out[where] = x[where].log()
    return out



def _h_poly(t):
    """Helper function to compute the 'h' polynomial matrix used in the
    cubic spline.
    
    Args:
        t (Tensor): A 1D tensor representing the normalized x values.
    
    Returns:
        Tensor: A 2D tensor of size (4, len(t)) representing the 'h' polynomial matrix.

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

    Args:
        x (Tensor): A 1D tensor representing the x-coordinates of the known data points.
        y (Tensor): A 1D tensor representing the y-coordinates of the known data points.
        xs (Tensor): A 1D tensor representing the x-coordinates of the positions where
                     the cubic spline function should be evaluated.
        extend (str, optional): The method for handling extrapolation, either "const", "extrapolate", or "linear".
                                Default is "extrapolate".
                                "const": Use the value of the last known data point for extrapolation.
                                "linear": Use linear extrapolation based on the last two known data points.
                                "extrapolate": Use cubic extrapolation of data.
    
    Returns:
        Tensor: A 1D tensor representing the interpolated values at the specified positions (xs).

    """
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[:-1], xs) - 1
    dx = x[idxs + 1] - x[idxs]
    hh = _h_poly((xs - x[idxs]) / dx)
    ret = (
        hh[0] * y[idxs]
        + hh[1] * m[idxs] * dx
        + hh[2] * y[idxs + 1]
        + hh[3] * m[idxs + 1] * dx
    )
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
    Similar to `torch.nn.functional.grid_sample` with `align_corners=False`.

    Args:
        im: 2D tensor.
        x: 0D or 1D tensor of x coordinates at which to interpolate.
        y: 0D or 1D tensor of y coordinates at which to interpolate.

    Returns:
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
        result[idxs_out_of_bounds] = torch.zeros_like(result[idxs_out_of_bounds])

    return result


def vmap_n(
    func: Callable,
    depth: int = 1,
    in_dims: Union[int, Tuple] = 0,
    out_dims: Union[int, Tuple[int, ...]] = 0,
    randomness: str = "error",
) -> Callable:
    """
    Returns `func` transformed `depth` times by `vmap`, with the same arguments
    passed to `vmap` each time.

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
    Runs the k-means++ initialization algorithm.
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
