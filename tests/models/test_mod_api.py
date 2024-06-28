from tempfile import NamedTemporaryFile
import os
import yaml

import pytest
import torch

try:
    from pydantic import create_model
except ImportError:
    raise ImportError(
        "The `pydantic` package is required to use this feature. "
        "You can install it using `pip install pydantic==2.7`. This package requires rust. Make sure you have the permissions to install the dependencies.\n "
        "Otherwise, the maintainer can install the package for you, you can then use `pip install --no-index pydantic`"
    )

import caustics
from caustics.models.utils import setup_simulator_models
from caustics.models.base_models import StateConfig, Field
from utils.models import setup_complex_multiplane_yaml
import textwrap


@pytest.fixture
def ConfigModel():
    simulators = setup_simulator_models()
    return create_model(
        "Config", __base__=StateConfig, simulator=(simulators, Field(...))
    )


@pytest.fixture
def x_input():
    return torch.tensor([
    #   z_s  z_l   x0   y0   q    phi     b    x0   y0   q     phi    n    Re
        1.5, 0.5, -0.2, 0.0, 0.4, 1.5708, 1.7, 0.0, 0.0, 0.5, -0.985, 1.3, 1.0,
    #   Ie    x0   y0   q    phi  n   Re   Ie
        5.0, -0.2, 0.0, 0.8, 0.0, 1., 1.0, 10.0
    ])  # fmt: skip


@pytest.fixture
def sim_yaml():
    return textwrap.dedent(
        """\
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
        kind: LensSource
        init_kwargs:
            # Single lense
            lens: *lens
            source: *src
            lens_light: *lnslt
            pixelscale: 0.05
            pixels_x: 100
    """
    )


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
def simple_config_dict(sim_yaml):
    return yaml.safe_load(sim_yaml)


@pytest.fixture
def sim_obj():
    cosmology = caustics.FlatLambdaCDM()
    sie = caustics.SIE(cosmology=cosmology, name="lens")
    src = caustics.Sersic(name="source")
    lnslt = caustics.Sersic(name="lenslight")
    return caustics.LensSource(
        lens=sie, source=src, lens_light=lnslt, pixelscale=0.05, pixels_x=100
    )


def test_build_simulator(sim_yaml_file, sim_obj, x_input):
    sim = caustics.build_simulator(sim_yaml_file)

    result = sim(x_input, quad_level=3)
    expected_result = sim_obj(x_input, quad_level=3)
    assert sim.graph(True, True)
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


def test_build_simulator_w_state(sim_yaml_file, sim_obj, x_input):
    sim = caustics.build_simulator(sim_yaml_file)
    params = dict(zip(sim.x_order, x_input))

    # Set the parameters from x input
    # using set attribute to the module objects
    # this makes the the params to be static
    for k, v in params.items():
        n, p = k.split(".")
        if n == sim.name:
            setattr(sim, p, v)
            continue
        key = sim._module_key_map[n]
        mod = getattr(sim, key)
        setattr(mod, p, v)

    state_dict = sim.state_dict()

    # Save the state
    state_path = None
    with NamedTemporaryFile("wb", suffix=".st", delete=False) as f:
        state_path = f.name
        state_dict.save(state_path)

    # Add the path to state to the sim yaml
    with open(sim_yaml_file, "a") as f:
        f.write(
            textwrap.dedent(
                f"""
        state:
            load:
                path: {state_path}
        """
            )
        )

    # Load the state
    # First remove the original sim
    del sim
    newsim = caustics.build_simulator(sim_yaml_file)
    result = newsim(quad_level=3)
    expected_result = sim_obj(x_input, quad_level=3)
    assert newsim.graph(True, True)
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected_result)


@pytest.mark.parametrize(
    "psf",
    [
        {
            "func": "caustics.utils.gaussian",
            "kwargs": {
                "pixelscale": 0.05,
                "nx": 11,
                "ny": 12,
                "sigma": 0.2,
                "upsample": 2,
            },
        },
        {"function": "caustics.utils.gaussian", "sigma": 0.2},
        [[2.0], [2.0]],
    ],
)
@pytest.mark.parametrize("pixels_y", ["50", 50.3])  # will get casted to int
def test_init_kwargs_validate(ConfigModel, simple_config_dict, psf, pixels_y):
    # Add psf
    test_config_dict = {**simple_config_dict}
    test_config_dict["simulator"]["init_kwargs"]["psf"] = psf
    test_config_dict["simulator"]["init_kwargs"]["pixels_y"] = pixels_y
    if isinstance(psf, dict) and "func" not in psf:
        with pytest.raises(ValueError):
            ConfigModel(**test_config_dict)
    else:
        # Test that the init_kwargs are validated
        config = ConfigModel(**test_config_dict)
        assert config.simulator.model_obj()
