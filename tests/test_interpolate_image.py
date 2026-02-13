import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from caustics.utils import meshgrid, interp2d, interp3d
from caustics.light import Pixelated
from caustics.backend_obj import backend


def test_random_inbounds(device):
    """
    Checks correctness against scipy at random in-bounds points.
    """
    nx = 57
    ny = 100
    n_pts = 7

    for method in ["nearest", "linear"]:
        image = backend.to(backend.randn(ny, nx), device=device, dtype=backend.float64)
        y_max = 1 - 1 / ny
        x_max = 1 - 1 / nx
        ys = backend.to(
            2
            * (backend.to(backend.rand((n_pts,)), dtype=backend.float64) - 0.5)
            * y_max,
            device=device,
        )
        xs = backend.to(
            2
            * (backend.to(backend.rand((n_pts,)), dtype=backend.float64) - 0.5)
            * x_max,
            device=device,
        )
        points = np.linspace(-y_max, y_max, ny), np.linspace(-x_max, x_max, nx)
        rg = RegularGridInterpolator(
            points, backend.to_numpy(backend.to(image, dtype=backend.float64)), method
        )
        res_rg = backend.as_array(
            rg(
                backend.to_numpy(
                    backend.to(backend.stack((ys, xs), 1), dtype=backend.float64)
                )
            ),
            device=device,
        )

        res = interp2d(image, xs, ys, method)

        assert backend.allclose(res, res_rg)


def test_consistency(device):
    """
    Checks that interpolating at pixel positions gives back the original image.
    """
    if backend.backend == "torch":
        torch.manual_seed(60)

    # Interpolation grid aligned with pixel centers
    nx = 50
    ny = 79
    res = 1.0
    thx, thy = meshgrid(res, nx, ny, device=device)
    thx = backend.to(thx, dtype=backend.float64)
    thy = backend.to(thy, dtype=backend.float64)
    scale_x = res * nx / 2
    scale_y = res * ny / 2

    for method in ["nearest", "linear"]:
        image = backend.to(backend.randn(ny, nx), device=device, dtype=backend.float64)
        x = backend.to(backend.flatten(thx) / scale_x, device=device)
        y = backend.to(backend.flatten(thy) / scale_y, device=device)
        image_interpd = interp2d(image, x, y, method).reshape(ny, nx)
        assert backend.allclose(image_interpd, image, atol=1e-4)


def test_consistency_3d(device):
    """
    Checks that interpolating at pixel positions gives back the original image.
    """
    if backend.backend == "torch":
        torch.manual_seed(60)

    # Interpolation grid aligned with pixel centers
    nx = 50
    ny = 79
    nt = 20
    res = 1.0
    xs = backend.linspace(-1, 1, nx, device=device, dtype=backend.float32) * res * (nx - 1) / 2  # fmt: skip
    ys = backend.linspace(-1, 1, ny, device=device, dtype=backend.float32) * res * (ny - 1) / 2  # fmt: skip
    ts = backend.linspace(-1, 1, nt, device=device, dtype=backend.float32)  # fmt: skip
    tht, thy, thx = backend.meshgrid((ts, ys, xs), indexing="ij")
    thx = backend.to(thx, dtype=backend.float64)
    thy = backend.to(thy, dtype=backend.float64)
    tht = backend.to(tht, dtype=backend.float64)
    scale_x = res * nx / 2
    scale_y = res * ny / 2

    for method in ["nearest", "linear"]:
        print(method)
        cube = backend.to(
            backend.randn(nt, ny, nx), device=device, dtype=backend.float64
        )
        x = backend.to(backend.flatten(thx) / scale_x, device=device)
        y = backend.to(backend.flatten(thy) / scale_y, device=device)
        t = backend.to(backend.flatten(tht) * (nt - 1) / nt, device=device)
        image_interpd = interp3d(cube, x, y, t, method).reshape(nt, ny, nx)
        assert backend.allclose(image_interpd, cube, atol=1e-4)


def test_pixelated_source(device):
    # Make sure pixelscale works as expected
    res = 0.05
    n = 32
    x, y = meshgrid(res, n, device=device)
    image = backend.ones((n, n), device=device)
    source = Pixelated(image=image, x0=0.0, y0=0.0, pixelscale=res)
    source.to(device=device)
    im = source.brightness(x, y)
    print(im)
    assert backend.all(im == image)

    # Check smaller res
    source = Pixelated(image=image, x0=0.0, y0=0.0, pixelscale=res / 2)
    source.to(device=device)
    im = source.brightness(x, y)
    expected_im = backend.to(
        backend.pad(backend.ones((n // 2, n // 2)), padding=[n // 4] * 4), device=device
    )
    print(im)
    assert backend.all(im == expected_im)
