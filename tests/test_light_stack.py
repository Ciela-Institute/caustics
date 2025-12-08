import torch

import caustics


def test_stack_sersic(device):
    res = 0.05
    nx = 100
    ny = 100
    thx, thy = caustics.utils.meshgrid(res, nx, ny, device=device)

    models = []
    for i in range(3):
        sersic = caustics.Sersic(
            name=f"sersic_{i}",
            x0=0.2 * i,
            y0=0.0,
            q=0.5,
            phi=3.14 / 2,
            n=2.0,
            Re=1.0 + 0.5 * i,
            Ie=10.0,
        )
        sersic.to(device=device)
        models.append(sersic)

    stack = caustics.LightStack(light_models=models, name="stack")

    params = stack.get_values()
    brightness = stack.brightness(thx, thy, params=params)

    assert brightness.shape == (nx, ny)
    assert torch.all(brightness >= 0.0).item()
    assert torch.all(torch.isfinite(brightness)).item()
