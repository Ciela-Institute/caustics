import numpy as np
from lenstronomy.Data.pixel_grid import PixelGrid

from caustics.utils import meshgrid


def test_meshgrid(device):
    res = 0.05
    nx = 200
    ny = 200
    thx, thy = meshgrid(res, nx, ny, device=device)

    res = 0.05  # size of pixel in angular coordinates #
    ra_at_xy_0, dec_at_xy_0 = (
        -5 + res / 2,
        -5 + res / 2,
    )  # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
    transform_pix2angle = (
        np.array([[1, 0], [0, 1]]) * res
    )  # linear translation matrix of a shift in pixel in a shift in coordinates
    kwargs_pixel = {
        "nx": nx,
        "ny": ny,  # number of pixels per axis
        "ra_at_xy_0": ra_at_xy_0,  # RA at pixel (0,0)
        "dec_at_xy_0": dec_at_xy_0,  # DEC at pixel (0,0)
        "transform_pix2angle": transform_pix2angle,
    }
    pixel_grid = PixelGrid(**kwargs_pixel)

    assert np.allclose(thx.cpu().numpy(), pixel_grid.coordinate_grid(nx, ny)[0])
    assert np.allclose(thy.cpu().numpy(), pixel_grid.coordinate_grid(nx, ny)[1])
