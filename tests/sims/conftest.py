import pytest

from caustics.sims.simulator import Simulator
from caustics.lenses import EPL
from caustics.light import Sersic
from caustics.cosmology import FlatLambdaCDM


@pytest.fixture
def simple_common_sim():
    class Sim(Simulator):
        def __init__(self):
            super().__init__()
            self.cosmo = FlatLambdaCDM(h0=None)
            self.epl = EPL(self.cosmo)
            self.sersic = Sersic()
            self.add_param("z_s", 1.0)

    sim = Sim()
    yield sim
    del sim
