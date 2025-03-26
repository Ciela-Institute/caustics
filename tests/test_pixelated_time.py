import torch
import caustics


def test_pixelated_time():

    nx, ny = 64, 64

    X, Y = torch.meshgrid(
        torch.linspace(-5, 5, nx), torch.linspace(-5, 5, ny), indexing="ij"
    )
    R = torch.sqrt(X**2 + Y**2)
    T = torch.linspace(0, 9, 10)
    cube = torch.stack([torch.exp(-0.5 * R**2 / (0.5 + t)) for t in T])

    cubemodel = caustics.PixelatedTime(
        cube=cube, x0=0.1, y0=0.2, pixelscale=0.1, t_end=10
    )

    rescale_pixels = 0.1 * (nx - 1) / 10
    X = X * rescale_pixels
    Y = Y * rescale_pixels
    sample_exact = cubemodel.brightness(X + 0.1, Y + 0.2, torch.ones_like(X) * 4.5)
    assert torch.allclose(sample_exact, cube[4])

    sample_interp = cubemodel.brightness(X + 0.1, Y + 0.2, torch.ones_like(X) * 5)
    assert torch.allclose(sample_interp, 0.5 * (cube[4] + cube[5]))
