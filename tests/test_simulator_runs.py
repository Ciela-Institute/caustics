from math import pi

import torch

from caustics.sims import LensSource, Microlens
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE
from caustics.light import Sersic
from caustics.utils import gaussian
from caustics import build_simulator

from utils import mock_from_file


def test_simulator_runs(sim_source, device, mocker):
    if sim_source == "yaml":
        yaml_str = """\
        cosmology: &cosmology
            name: "cosmo"
            kind: FlatLambdaCDM

        lensmass: &lensmass
            name: lens
            kind: SIE
            params:
                z_l: 1.0
                x0: 0.0
                y0: 0.01
                q: 0.5
                phi: pi / 3.0
                b: 1.0
            init_kwargs:
                cosmology: *cosmology

        source: &source
            name: source
            kind: Sersic
            params:
                x0: 0.01
                y0: -0.03
                q: 0.6
                phi: -pi / 4
                n: 2.0
                Re: 0.5
                Ie: 1.0

        lenslight: &lenslight
            name: lenslight
            kind: Sersic
            params:
                x0: 0.0
                y0: 0.01
                q: 0.7
                phi: pi / 4
                n: 3.0
                Re: 0.7
                Ie: 1.0

        psf: &psf
            func: caustics.utils.gaussian
            kwargs:
                pixelscale: 0.05
                nx: 11
                ny: 12
                sigma: 0.2
                upsample: 2

        simulator:
            name: simulator
            kind: LensSource
            params:
                z_s: 2.0
            init_kwargs:
                # Single lense
                lens: *lensmass
                source: *source
                lens_light: *lenslight
                pixelscale: 0.05
                pixels_x: 50
                psf: *psf
        """
        mock_from_file(mocker, yaml_str)
        sim = build_simulator("/path/to/sim.yaml")  # Path doesn't actually exists
    else:
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

        sim = LensSource(
            name="simulator",
            lens=lensmass,
            source=source,
            pixelscale=0.05,
            pixels_x=50,
            lens_light=lenslight,
            psf=psf,
            z_s=2.0,
        )

    sim.to(device=device)

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

    # Check quadrature integration is accurate
    assert torch.allclose(sim(), sim(quad_level=3), rtol=1e-1)
    assert torch.allclose(sim(quad_level=3), sim(quad_level=5), rtol=1e-2)


def test_microlens_simulator_runs():
    cosmology = FlatLambdaCDM()
    sie = SIE(cosmology=cosmology, name="lens")
    src = Sersic(name="source")

    x = torch.tensor([
    #   z_s  z_l   x0   y0   q    phi     b    x0   y0   q     phi    n    Re   Ie
        1.5, 0.5, -0.2, 0.0, 0.4, 1.5708, 1.7, 0.0, 0.0, 0.5, -0.985, 1.3, 1.0, 5.0
    ])  # fmt: skip
    fov = torch.tensor((-1, -0.5, -0.25, 0.25))
    sim = Microlens(lens=sie, source=src)
    sim(x, fov=fov)
    sim(x, fov=fov, method="grid")
