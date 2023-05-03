import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from caustic.utils import get_meshgrid, interp2d


def test_random_inbounds():
    """
    Checks correctness against scipy at random in-bounds points.
    """
    nx = 57
    ny = 100
    n_pts = 7

    for method in ["nearest", "linear"]:
        image = torch.randn(ny, nx).double()
        y_max = 1 - 1 / ny
        x_max = 1 - 1 / nx
        ys = 2 * (torch.rand((n_pts,)).double() - 0.5) * y_max
        xs = 2 * (torch.rand((n_pts,)).double() - 0.5) * x_max
        points = np.linspace(-y_max, y_max, ny), np.linspace(-x_max, x_max, nx)
        rg = RegularGridInterpolator(points, image.double().numpy(), method)
        res_rg = torch.as_tensor(rg(torch.stack((ys, xs), 1).double().numpy()))

        res = interp2d(image, xs, ys, method)

        assert torch.allclose(res, res_rg)


def test_consistency():
    """
    Checks that interpolating at pixel positions gives back the original image.
    """
    torch.manual_seed(60)

    # Interpolation grid aligned with pixel centers
    nx = 50
    ny = 79
    res = 1.0
    thx, thy = get_meshgrid(res, nx, ny)
    thx = thx.double()
    thy = thy.double()
    scale_x = res * nx / 2
    scale_y = res * ny / 2

    for method in ["nearest", "linear"]:
        image = torch.randn(ny, nx).double()
        image_interpd = interp2d(
            image, thx.flatten() / scale_x, thy.flatten() / scale_y, method
        ).reshape(ny, nx)
        assert torch.allclose(image_interpd, image, atol=1e-5)


if __name__ == "__main__":
    test_random_inbounds()
    test_consistency()
