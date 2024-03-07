from tempfile import NamedTemporaryFile
import sys
import os

import pytest
import torch

import caustics
from utils.models import setup_complex_multiplane_yaml


@pytest.fixture
def sim_yaml():
    return """\
    cosmology: &cosmo
        name: cosmo
        kind: FlatLambdaCDM

    lens: &lens
        name: lens
        kind: SIE
        init_kwargs:
            cosmology: *cosmo

    src: &src
        name: source
        kind: Sersic

    lnslt: &lnslt
        name: lenslight
        kind: Sersic

    simulator:
        name: minisim
        kind: Lens_Source
        init_kwargs:
            # Single lense
            lens: *lens
            source: *src
            lens_light: *lnslt
            pixelscale: 0.05
            pixels_x: 100
    """


@pytest.fixture
def sim_yaml_file(sim_yaml):
    with NamedTemporaryFile("w") as f:
        f.write(sim_yaml)
        f.flush()
        yield f.name


@pytest.fixture
def sim_obj():
    cosmology = caustics.FlatLambdaCDM()
    sie = caustics.SIE(cosmology=cosmology, name="lens")
    src = caustics.Sersic(name="source")
    lnslt = caustics.Sersic(name="lenslight")
    return caustics.Lens_Source(
        lens=sie, source=src, lens_light=lnslt, pixelscale=0.05, pixels_x=100
    )


def test_build_simulator(sim_yaml_file, sim_obj):
    # TODO: Add test for using state_dict
    sim = caustics.build_simulator(sim_yaml_file)
    x = torch.tensor([
    #   z_s  z_l   x0   y0   q    phi     b    x0   y0   q     phi    n    Re
        1.5, 0.5, -0.2, 0.0, 0.4, 1.5708, 1.7, 0.0, 0.0, 0.5, -0.985, 1.3, 1.0,
    #   Ie    x0   y0   q    phi  n   Re   Ie
        5.0, -0.2, 0.0, 0.8, 0.0, 1., 1.0, 10.0
    ])  # fmt: skip

    result = sim(x, quad_level=3)
    expected_result = sim_obj(x, quad_level=3)
    assert sim.get_graph(True, True)
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected_result)


def test_complex_build_simulator():
    yaml_str = setup_complex_multiplane_yaml()
    x = torch.tensor(
        [
            #   z_s   x0   y0   q     phi    n    Re
            1.5,
            0.0,
            0.0,
            0.5,
            -0.985,
            1.3,
            1.0,
            #   Ie    x0   y0   q    phi  n   Re   Ie
            5.0,
            -0.2,
            0.0,
            0.8,
            0.0,
            1.0,
            1.0,
            10.0,
        ]
    )
    delete_file = False if sys.platform.startswith("win") else True
    with NamedTemporaryFile("w", delete=delete_file) as f:  # Don't delete for windows
        f.write(yaml_str)
        f.flush()
        sim = caustics.build_simulator(f.name)
        image = sim(x, quad_level=3)
        assert isinstance(image, torch.Tensor)
        f.close()
        if not delete_file:
            os.unlink(f.name)


def test_build_simulator_w_state():
    # TODO: Add test for using state_dict
    pass