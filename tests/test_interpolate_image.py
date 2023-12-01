import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from caustics.utils import get_meshgrid, interp2d
from caustics.light import Pixelated


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


def test_pixelated_source():
    # Make sure pixelscale works as expected
    res = 0.05
    n = 32
    x, y = get_meshgrid(res, n, n)
    image = torch.ones(n, n)
    source = Pixelated(image=image, x0=0.0, y0=0.0, pixelscale=res)
    im = source.brightness(x, y)
    print(im)
    assert torch.all(im == image)

    # Check smaller res
    source = Pixelated(image=image, x0=0.0, y0=0.0, pixelscale=res / 2)
    im = source.brightness(x, y)
    expected_im = torch.nn.functional.pad(torch.ones(n // 2, n // 2), pad=[n // 4] * 4)
    print(im)
    assert torch.all(im == expected_im)
