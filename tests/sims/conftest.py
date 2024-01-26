import pytest

from caustics.sims.simulator import Simulator
from caustics.lenses import EPL
from caustics.light import Sersic
from caustics.cosmology import FlatLambdaCDM


@pytest.fixture
def test_epl_values():
    return {
        "z_l": 0.5,
        "phi": 0.0,
        "b": 1.0,
        "t": 1.0,
    }


@pytest.fixture
def test_sersic_values():
    return {
        "q": 0.9,
        "phi": 0.3,
        "n": 1.0,
    }


@pytest.fixture
def simple_common_sim(test_epl_values, test_sersic_values):
    class Sim(Simulator):
        def __init__(self):
            super().__init__()
            self.cosmo = FlatLambdaCDM(h0=None)
            self.epl = EPL(self.cosmo, **test_epl_values)
            self.sersic = Sersic(**test_sersic_values)
            self.add_param("z_s", 1.0)

    sim = Sim()
    yield sim
    del sim
