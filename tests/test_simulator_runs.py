from math import pi

import torch

from caustics.sims import Lens_Source
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
            kind: Lens_Source
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

        sim = Lens_Source(
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
