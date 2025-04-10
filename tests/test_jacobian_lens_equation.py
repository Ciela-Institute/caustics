from math import pi

import torch

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Multiplane
from caustics.utils import meshgrid


def test_jacobian_autograd_vs_finitediff(device):
    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    z_s = torch.tensor(1.2, device=device)
    lens = SIE(name="sie", cosmology=cosmology, z_s=z_s)
    lens.to(device=device)
    thx, thy = meshgrid(0.01, 20, device=device)

    # Parameters
    x = torch.tensor([0.5, 0.912, -0.442, 0.7, pi / 3, 1.4], device=device)

    # Evaluate Jacobian
    J_autograd = lens.jacobian_lens_equation(thx, thy, x)
    J_finitediff = lens.jacobian_lens_equation(
        thx,
        thy,
        x,
        method="finitediff",
        pixelscale=torch.tensor(0.01, device=device),
    )

    assert (
        torch.sum(((J_autograd - J_finitediff) / J_autograd).abs() < 1e-3)
        > 0.8 * J_autograd.numel()
    )


def test_multiplane_jacobian(device):
    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32, device=device)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=torch.float32, device=device)

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = torch.tensor([p for _xs in xs for p in _xs], dtype=torch.float32, device=device)

    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[SIE(name=f"sie_{i}", cosmology=cosmology) for i in range(len(xs))],
        z_s=z_s,
    )
    lens.to(device=device)
    thx, thy = meshgrid(0.1, 10, device=device)

    # Parameters
    x = torch.tensor(xs, device=device).flatten()
    A = lens.jacobian_lens_equation(thx, thy, x)
    assert A.shape == (10, 10, 2, 2)


def test_multiplane_jacobian_autograd_vs_finitediff(device):
    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32, device=device)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=torch.float32, device=device)

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = torch.tensor([p for _xs in xs for p in _xs], dtype=torch.float32, device=device)

    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[SIE(name=f"sie_{i}", cosmology=cosmology) for i in range(len(xs))],
        z_s=z_s,
    )
    lens.to(device=device)
    thx, thy = meshgrid(0.01, 10, device=device)

    # Parameters
    x = torch.tensor(xs, device=device).flatten()

    # Evaluate Jacobian
    J_autograd = lens.jacobian_lens_equation(thx, thy, x)
    J_finitediff = lens.jacobian_lens_equation(
        thx, thy, x, method="finitediff", pixelscale=torch.tensor(0.01)
    )

    assert (
        torch.sum(((J_autograd - J_finitediff) / J_autograd).abs() < 1e-2)
        > 0.5 * J_autograd.numel()
    )


def test_multiplane_effective_convergence(device):
    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32, device=device)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=torch.float32, device=device)

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = torch.tensor([p for _xs in xs for p in _xs], dtype=torch.float32, device=device)

    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[SIE(name=f"sie_{i}", cosmology=cosmology) for i in range(len(xs))],
        z_s=z_s,
    )
    lens.to(device=device)
    thx, thy = meshgrid(0.1, 10, device=device)

    # Parameters
    x = torch.tensor(xs, device=device).flatten()
    C = lens.effective_convergence_div(thx, thy, x)
    assert C.shape == (10, 10)
    curl = lens.effective_convergence_curl(thx, thy, x)
    assert curl.shape == (10, 10)


if __name__ == "__main__":
    test_jacobian_autograd_vs_finitediff(None)
    test_multiplane_jacobian(None)
    test_multiplane_jacobian_autograd_vs_finitediff(None)
    test_multiplane_effective_convergence(None)
