import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from caustics.utils import meshgrid, interp2d, interp3d
from caustics.light import Pixelated


def test_random_inbounds(device):
    """
    Checks correctness against scipy at random in-bounds points.
    """
    nx = 57
    ny = 100
    n_pts = 7

    for method in ["nearest", "linear"]:
        image = torch.randn(ny, nx).double().to(device)
        y_max = 1 - 1 / ny
        x_max = 1 - 1 / nx
        ys = (2 * (torch.rand((n_pts,)).double() - 0.5) * y_max).to(device=device)
        xs = (2 * (torch.rand((n_pts,)).double() - 0.5) * x_max).to(device=device)
        points = np.linspace(-y_max, y_max, ny), np.linspace(-x_max, x_max, nx)
        rg = RegularGridInterpolator(points, image.double().cpu().numpy(), method)
        res_rg = torch.as_tensor(
            rg(torch.stack((ys, xs), 1).double().cpu().numpy()), device=device
        )

        res = interp2d(image, xs, ys, method)

        assert torch.allclose(res, res_rg)


def test_consistency(device):
    """
    Checks that interpolating at pixel positions gives back the original image.
    """
    torch.manual_seed(60)

    # Interpolation grid aligned with pixel centers
    nx = 50
    ny = 79
    res = 1.0
    thx, thy = meshgrid(res, nx, ny, device=device)
    thx = thx.double()
    thy = thy.double()
    scale_x = res * nx / 2
    scale_y = res * ny / 2

    for method in ["nearest", "linear"]:
        image = torch.randn(ny, nx).double().to(device)
        x = (thx.flatten() / scale_x).to(device=device)
        y = (thy.flatten() / scale_y).to(device=device)
        image_interpd = interp2d(image, x, y, method).reshape(ny, nx)
        assert torch.allclose(image_interpd, image, atol=1e-5)


def test_consistency_3d(device):
    """
    Checks that interpolating at pixel positions gives back the original image.
    """
    torch.manual_seed(60)

    # Interpolation grid aligned with pixel centers
    nx = 50
    ny = 79
    nt = 20
    res = 1.0
    xs = torch.linspace(-1, 1, nx, device=device, dtype=torch.float32) * res * (nx - 1) / 2  # fmt: skip
    ys = torch.linspace(-1, 1, ny, device=device, dtype=torch.float32) * res * (ny - 1) / 2  # fmt: skip
    ts = torch.linspace(-1, 1, nt, device=device, dtype=torch.float32)  # fmt: skip
    tht, thy, thx = torch.meshgrid((ts, ys, xs), indexing="ij")
    thx = thx.double()
    thy = thy.double()
    tht = tht.double()
    scale_x = res * nx / 2
    scale_y = res * ny / 2

    for method in ["nearest", "linear"]:
        print(method)
        cube = torch.randn(nt, ny, nx).double().to(device)
        x = (thx.flatten() / scale_x).to(device=device)
        y = (thy.flatten() / scale_y).to(device=device)
        t = (tht.flatten() * (nt - 1) / nt).to(device=device)
        image_interpd = interp3d(cube, x, y, t, method).reshape(nt, ny, nx)
        assert torch.allclose(image_interpd, cube, atol=1e-5)


def test_pixelated_source(device):
    # Make sure pixelscale works as expected
    res = 0.05
    n = 32
    x, y = meshgrid(res, n, device=device)
    image = torch.ones(n, n, device=device)
    source = Pixelated(image=image, x0=0.0, y0=0.0, pixelscale=res)
    source.to(device=device)
    im = source.brightness(x, y)
    print(im)
    assert torch.all(im == image)

    # Check smaller res
    source = Pixelated(image=image, x0=0.0, y0=0.0, pixelscale=res / 2)
    source.to(device=device)
    im = source.brightness(x, y)
    expected_im = torch.nn.functional.pad(
        torch.ones(n // 2, n // 2), pad=[n // 4] * 4
    ).to(device=device)
    print(im)
    assert torch.all(im == expected_im)
