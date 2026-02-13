import caustics
from caustics.backend_obj import backend


def test_pixelated_time():

    nx, ny = 64, 64

    X, Y = backend.meshgrid(
        backend.linspace(-5, 5, nx), backend.linspace(-5, 5, ny), indexing="ij"
    )
    R = backend.sqrt(X**2 + Y**2)
    T = backend.linspace(0, 9, 10)
    cube = backend.stack([backend.exp(-0.5 * R**2 / (0.5 + t)) for t in T])

    cubemodel = caustics.PixelatedTime(
        cube=cube, x0=0.1, y0=0.2, pixelscale=0.1, t_end=10
    )

    rescale_pixels = 0.1 * (nx - 1) / 10
    X = X * rescale_pixels
    Y = Y * rescale_pixels
    sample_exact = cubemodel.brightness(X + 0.1, Y + 0.2, backend.ones_like(X) * 4.5)
    assert backend.allclose(sample_exact, cube[4])

    sample_interp = cubemodel.brightness(X + 0.1, Y + 0.2, backend.ones_like(X) * 5)
    assert backend.allclose(sample_interp, 0.5 * (cube[4] + cube[5]))
