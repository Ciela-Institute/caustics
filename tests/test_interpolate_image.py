import torch

from caustics.utils import meshgrid
from caustics.light import Pixelated


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
