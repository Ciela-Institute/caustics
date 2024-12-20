from io import StringIO
from math import pi

import torch

from caustics.sims import LensSource, Microlens
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE
from caustics.light import Sersic
from caustics.utils import gaussian
from caustics import build_simulator


def test_simulator_runs(sim_source, device):
    if sim_source == "yaml":
        yaml_str = """\
        cosmology: &cosmology
            name: "cosmo"
            kind: FlatLambdaCDM

        lensmass: &lensmass
            name: lens
            kind: SIE
            init_kwargs:
                z_l: 1.0
                z_s: 2.0
                x0: 0.0
                y0: 0.01
                q: 0.5
                phi: 1.05
                b: 1.0
                cosmology: *cosmology

        source: &source
            name: source
            kind: Sersic
            init_kwargs:
                x0: 0.01
                y0: -0.03
                q: 0.6
                phi: -0.785
                n: 1.5
                Re: 0.5
                Ie: 1.0

        lenslight: &lenslight
            name: lenslight
            kind: Sersic
            init_kwargs:
                x0: 0.0
                y0: 0.01
                q: 0.7
                phi: 0.785
                n: 3.0
                Re: 0.7
                Ie: 1.0

        psf: &psf
            kind: utils.gaussian
            init_kwargs:
                pixelscale: 0.05
                nx: 11
                ny: 11
                sigma: 0.2
                upsample: 2

        simulator:
            name: simulator
            kind: LensSource
            init_kwargs:
                # Single lens
                lens: *lensmass
                source: *source
                lens_light: *lenslight
                pixelscale: 0.05
                pixels_x: 50{quad_level}
        """
        with StringIO(yaml_str.format(quad_level="")) as f:
            sim = build_simulator(f)
        with StringIO(yaml_str.format(quad_level="\n            quad_level: 3")) as f:
            sim_q3 = build_simulator(f)
    else:
        # Model
        cosmology = FlatLambdaCDM(name="cosmo")
        lensmass = SIE(
            name="lens",
            cosmology=cosmology,
            z_l=1.0,
            z_s=2.0,
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
        )

        sim_q3 = LensSource(
            name="simulator",
            lens=lensmass,
            source=source,
            pixelscale=0.05,
            pixels_x=50,
            lens_light=lenslight,
            psf=psf,
            quad_level=3,
        )

    sim.to(device=device)
    sim_q3.to(device=device)

    # Test setters
    sim.pixelscale = 0.05
    sim.pixels_x = 50
    sim.pixels_y = 50
    sim_q3.quad_level = 3
    sim.upsample_factor = 1
    sim.psf_shape = (11, 11)
    sim.psf_mode = "conv2d"

    assert torch.all(torch.isfinite(sim()))
    assert torch.all(
        torch.isfinite(
            sim(
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
                source_light=False,
                lens_light=True,
                lens_source=True,
                psf_convolve=True,
            )
        )
    )

    # Check quadrature integration is accurate
    assert torch.allclose(sim(), sim_q3(), rtol=1e-1)


def test_fft_vs_conv2d():
    # Model
    cosmology = FlatLambdaCDM(name="cosmo")
    lensmass = SIE(
        name="lens",
        cosmology=cosmology,
        z_l=1.0,
        z_s=2.0,
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
    psf[3, 4] = 0.1  # make PSF asymmetric
    psf /= psf.sum()

    sim_fft = LensSource(
        name="simulatorfft",
        lens=lensmass,
        source=source,
        pixelscale=0.05,
        pixels_x=50,
        lens_light=lenslight,
        psf=psf,
        psf_mode="fft",
        quad_level=3,
    )

    sim_conv2d = LensSource(
        name="simulatorconv2d",
        lens=lensmass,
        source=source,
        pixelscale=0.05,
        pixels_x=50,
        lens_light=lenslight,
        psf=psf,
        psf_mode="conv2d",
        quad_level=3,
    )

    print(torch.max(torch.abs((sim_fft() - sim_conv2d()) / sim_fft())))
    assert torch.allclose(sim_fft(), sim_conv2d(), rtol=1e-1)


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
