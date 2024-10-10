import torch

import caustics


def test_stack_sersic(device):
    res = 0.05
    nx = 100
    ny = 100
    thx, thy = caustics.utils.meshgrid(res, nx, ny, device=device)

    models = []
    params = []
    for i in range(3):
        sersic = caustics.Sersic(
            name=f"sersic_{i}",
        )
        sersic.to(device=device)
        models.append(sersic)
        params.append(
            torch.tensor([0.0 + 0.2 * i, 0.0, 0.5, 3.14 / 2, 2.0, 1.0 + 0.5 * i, 10.0])
        )

    stack = caustics.LightStack(light_models=models, name="stack")

    brightness = stack.brightness(thx, thy, params=params)

    assert brightness.shape == (nx, ny)
    assert torch.all(brightness >= 0.0).item()
    assert torch.all(torch.isfinite(brightness)).item()
