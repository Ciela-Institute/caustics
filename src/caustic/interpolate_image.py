from typing import Union

import torch
from torch import Tensor
from torch.nn.functional import grid_sample


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
