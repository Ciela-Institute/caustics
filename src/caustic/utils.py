from math import pi
from typing import Callable, Optional, Tuple, Union

import functorch
import torch
from torch import Tensor
from torch.nn.functional import grid_sample


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
    base_grid = torch.linspace(-1, 1, nx, device=device, dtype=dtype)
    xs = base_grid * resolution * (nx - 1) / 2
    ys = base_grid * resolution * (ny - 1) / 2
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


def interpolate_image(
    thx: Tensor,
    thy: Tensor,
    thx0: Union[float, Tensor],
    thy0: Union[float, Tensor],
    image: Tensor,
    scale: Union[float, Tensor],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
):
    """
    Shifts, scales and interpolates the image.

    Args:
        scale: distance from the origin to the center of a pixel on the edge of
            the image. For the common case of an image defined on a meshgrid of
            width `fov` and resolution `res`, this should be `0.5 * (fov - res)`.
    """
    if image.ndim != 4:
        raise ValueError("image must have four dimensions")

    # Batch grid to match image batching
    grid = (
        torch.stack((thx - thx0, thy - thy0), dim=-1).reshape(-1, *thx.shape[-2:], 2)
        / scale
    )
    grid = grid.repeat((len(image), 1, 1, 1))
    return grid_sample(image, grid, mode, padding_mode, align_corners)


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
        vmapd_func = functorch.vmap(vmapd_func, in_dims, out_dims, randomness)

    return vmapd_func


def get_cluster_means(xs, k):
    """
    Runs the k-means++ initialization algorithm.
    """
    b = len(xs)
    mean_idxs = [torch.randint(high=b, size=(), device=xs.device).item()]
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
        new_idx = d2s_closest.argmax().item()
        means.append(unselected_xs[new_idx])
        mean_idxs.append(new_idx)

    return torch.stack(means)
