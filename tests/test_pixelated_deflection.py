import torch
import caustics


def test_pixelated_deflection():

    n = 64

    dX, dY = torch.meshgrid(
        torch.linspace(1, -1, n), torch.linspace(1, -1, n), indexing="ij"
    )

    cosmology = caustics.FlatLambdaCDM(name="cosmo")
    model = caustics.PixelatedDeflection(
        deflection_map=torch.stack((dX, dY)),
        pixelscale=2 / n,
        cosmology=cosmology,
        x0=0.0,
        y0=0.0,
        z_s=1.0,
        z_l=0.5,
    )

    X, Y = torch.meshgrid(
        torch.linspace(-0.9, 0.9, 32), torch.linspace(-0.9, 0.9, 32), indexing="ij"
    )

    bx, by = model.raytrace(X, Y)
    print(bx)
    assert torch.all(torch.isfinite(bx))
    assert torch.allclose(bx, torch.zeros_like(bx))
    assert torch.all(torch.isfinite(by))
    assert torch.allclose(by, torch.zeros_like(by))
