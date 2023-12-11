from math import pi

import torch

from caustics.sims import Lens_Source
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Multiplane
from caustics.light import Sersic
from caustics.utils import gaussian, get_meshgrid

__all__ = ["test"]


def _test_simulator_runs():
    # Model
    cosmology = FlatLambdaCDM(name="cosmo")
    lensmass = SIE(
        name="lens",
        cosmology=cosmology,
        z_l=1.0,
        x0=0.0,
        y0=0.01,
        q=0.5,
        phi=pi / 3.0,
        b=1.0,
    )

    source = Sersic(
        name="source", x0=0.01, y0=-0.03, q=0.6, phi=-pi / 4, n=2.0, Re=0.5, Ie=1.0
    )
    lenslight = Sersic(
        name="lenslight", x0=0.0, y0=0.01, q=0.7, phi=pi / 4, n=3.0, Re=0.7, Ie=1.0
    )

    psf = gaussian(0.05, 11, 11, 0.2, upsample=2)

    sim = Lens_Source(
        lens=lensmass,
        source=source,
        pixelscale=0.05,
        pixels_x=50,
        lens_light=lenslight,
        psf=psf,
        z_s=2.0,
    )

    assert torch.all(torch.isfinite(sim()))
    assert torch.all(
        torch.isfinite(
            sim(
                {},
                source_light=True,
                lens_light=True,
                lens_source=True,
                psf_convolve=False,
            )
        )
    )
    assert torch.all(
        torch.isfinite(
            sim(
                {},
                source_light=True,
                lens_light=True,
                lens_source=False,
                psf_convolve=True,
            )
        )
    )
    assert torch.all(
        torch.isfinite(
            sim(
                {},
                source_light=True,
                lens_light=False,
                lens_source=True,
                psf_convolve=True,
            )
        )
    )
    assert torch.all(
        torch.isfinite(
            sim(
                {},
                source_light=False,
                lens_light=True,
                lens_source=True,
                psf_convolve=True,
            )
        )
    )


def _test_jacobian_autograd_vs_finitediff():
    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(name="sie", cosmology=cosmology)
    thx, thy = get_meshgrid(0.01, 20, 20)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, 0.912, -0.442, 0.7, pi / 3, 1.4])

    # Evaluate Jacobian
    J_autograd = lens.jacobian_lens_equation(thx, thy, z_s, lens.pack(x))
    J_finitediff = lens.jacobian_lens_equation(
        thx, thy, z_s, lens.pack(x), method="finitediff", pixelscale=torch.tensor(0.01)
    )

    assert (
        torch.sum(((J_autograd - J_finitediff) / J_autograd).abs() < 1e-3)
        > 0.8 * J_autograd.numel()
    )


def _test_multiplane_jacobian():
    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=torch.float32)

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = torch.tensor([p for _xs in xs for p in _xs], dtype=torch.float32)

    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[SIE(name=f"sie_{i}", cosmology=cosmology) for i in range(len(xs))],
    )
    thx, thy = get_meshgrid(0.1, 10, 10)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor(xs).flatten()
    A = lens.jacobian_lens_equation(thx, thy, z_s, lens.pack(x))
    assert A.shape == (10, 10, 2, 2)


def _test_multiplane_jacobian_autograd_vs_finitediff():
    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=torch.float32)

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = torch.tensor([p for _xs in xs for p in _xs], dtype=torch.float32)

    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[SIE(name=f"sie_{i}", cosmology=cosmology) for i in range(len(xs))],
    )
    thx, thy = get_meshgrid(0.01, 10, 10)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor(xs).flatten()

    # Evaluate Jacobian
    J_autograd = lens.jacobian_lens_equation(thx, thy, z_s, lens.pack(x))
    J_finitediff = lens.jacobian_lens_equation(
        thx, thy, z_s, lens.pack(x), method="finitediff", pixelscale=torch.tensor(0.01)
    )

    assert (
        torch.sum(((J_autograd - J_finitediff) / J_autograd).abs() < 1e-2)
        > 0.5 * J_autograd.numel()
    )


def _test_multiplane_effective_convergence():
    # Setup
    z_s = torch.tensor(1.5, dtype=torch.float32)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=torch.float32)

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = torch.tensor([p for _xs in xs for p in _xs], dtype=torch.float32)

    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[SIE(name=f"sie_{i}", cosmology=cosmology) for i in range(len(xs))],
    )
    thx, thy = get_meshgrid(0.1, 10, 10)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor(xs).flatten()
    C = lens.effective_convergence_div(thx, thy, z_s, lens.pack(x))
    assert C.shape == (10, 10)
    curl = lens.effective_convergence_curl(thx, thy, z_s, lens.pack(x))
    assert curl.shape == (10, 10)


def test():
    """
    Run tests for caustics basic functionallity.
    Run this function to ensure that caustics is working properly.

    Simply call::

        >>> import caustics
        >>> caustics.test()
        all tests passed!

    To run the checks.
    """

    _test_simulator_runs()
    _test_jacobian_autograd_vs_finitediff()
    _test_multiplane_jacobian()
    _test_multiplane_jacobian_autograd_vs_finitediff()
    _test_multiplane_effective_convergence()
    print("all tests passed!")
