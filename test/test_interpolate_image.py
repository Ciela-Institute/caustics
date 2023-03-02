import torch

from caustic.utils import get_meshgrid, interpolate_image


def test_consistency():
    """
    Make sure corners are aligned properly.
    """
    torch.manual_seed(50)

    n_pix = 10
    fov = 10.0
    res = fov / n_pix
    thx, thy = get_meshgrid(res, n_pix, n_pix)

    image = torch.randn(1, 1, n_pix, n_pix)
    image_interpd = interpolate_image(
        thx,
        thy,
        torch.tensor(0.0),
        torch.tensor(0.0),
        image,
        (fov - res) / 2,
        mode="nearest",
        align_corners=True,
    )

    assert (image == image_interpd).all()
