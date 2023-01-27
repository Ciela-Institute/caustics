import torch
from torch.nn.functional import grid_sample


def interpolate_image(thx, thy, thx0, thy0, image, scale):
    """
    Shifts, scales and interpolates the image.
    """
    if image.ndim != 4:
        raise ValueError("image must have four dimensions")

    # Batch grid to match image batching
    grid = (
        torch.stack((thx - thx0, thy - thy0), dim=-1).reshape(-1, *thx.shape[-2:], 2)
        / scale
    )
    grid = grid.repeat((len(image), 1, 1, 1))
    return grid_sample(image, grid, align_corners=False)
