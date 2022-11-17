import numpy as np
from lenstronomy.Data.pixel_grid import PixelGrid

from caustic.utils import get_meshgrid


def test_get_meshgrid():
    res = 0.05
    nx = 200
    ny = 200
    thx, thy = get_meshgrid(res, nx, ny)

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

    assert np.allclose(thx.numpy(), pixel_grid.coordinate_grid(nx, ny)[0])
    assert np.allclose(thy.numpy(), pixel_grid.coordinate_grid(nx, ny)[1])
