from tempfile import NamedTemporaryFile
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


def _write_temp_yaml(yaml_str: str):
    # Create temp file
    f = NamedTemporaryFile("w", delete=False)
    f.write(yaml_str)
    f.flush()
    f.close()

    return f.name


@pytest.fixture
def sim_yaml_file(sim_yaml):
    temp_file = _write_temp_yaml(sim_yaml)

    yield temp_file

    if os.path.exists(temp_file):
        os.unlink(temp_file)


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
    # Create temp file
    temp_file = _write_temp_yaml(yaml_str)

    # Open the temp file and build the simulator
    sim = caustics.build_simulator(temp_file)
    image = sim(x, quad_level=3)
    assert isinstance(image, torch.Tensor)

    # Remove the temp file
    if os.path.exists(temp_file):
        os.unlink(temp_file)


def test_build_simulator_w_state():
    # TODO: Add test for using state_dict
    pass
