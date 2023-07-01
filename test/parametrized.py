import torch
from caustic.parametrized import Parametrized
from caustic import Simulator, EPL, Sersic, FlatLambdaCDM
from itertools import chain


def test_module_creation():
    class SomeModule(Parametrized):
        def __init__(self):
            super().__init__()
            self.add_param("some_dynamic_param")


def test_module_params():
    # Test common simulator structure
    class Sim(Simulator):
        def __init__(self, name="test"):
            super().__init__(name)
            self.cosmo = FlatLambdaCDM("cosmo", h0=None)
            self.epl = EPL("lens", self.cosmo)
            self.sersic = Sersic("source")
            self.z_s = torch.tensor(1.0)
            self.thx, self.thy = get_meshgrid(0.04, 20, 20)
    
    sim = Sim()
    assert len(sim.params.keys()) == 2
    assert len(sim.params.dynamic.keys()) == 3
    assert len(sim.params.static.keys()) == 1
    # Test that total number of dynamic params is respected
    assert len(sim.params.dynamic.flatten().keys()) == 15
    
