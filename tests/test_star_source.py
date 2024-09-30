import caustics
import torch
from caustics.utils import meshgrid


def test_star_source():
    src = caustics.StarSource(name="source")

    rho = 1.5
    x = torch.tensor(
    [
        0.0,  # x0
        0.0,  # y0
        1 * rho,  # theta_s
        5.0,  # Ie,
        0.0,  # gamma
    ]
)
    theta_x = torch.tensor([0.1])
    theta_y = torch.tensor([0.2])
    z_s = torch.tensor([0.0])
    #Check if brightness is constant inside source with no limb darkening
    assert src.brightness(theta_x, theta_y, x) == 5.0
    #Check if brightness is zero outside source
    assert src.brightness(1.01*rho, theta_y, x) == 0.0
    #Check if function behaves as expected
    assert src.brightness(theta_x, theta_y, x) == caustics.light.func.brightness_star(0.0, 0.0, 1.5, 5.0, theta_x, theta_y, 0.0)

    #Check if brightness is finite everywhere over a grid
    n_pix = 10
    res = 0.05
    upsample_factor = 2
    thx, thy = meshgrid(
        res / upsample_factor,
        upsample_factor * n_pix,
        upsample_factor * n_pix,
        dtype=torch.float32,
    )

    assert torch.all(torch.isfinite(src.brightness(thx, thy, x)))