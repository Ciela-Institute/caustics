import torch
from caustic import Simulator, EPL, Sersic, FlatLambdaCDM
from caustic.utils import get_meshgrid


def test_params():
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
    assert len(sim.module_params.keys()) == 2
    assert len(sim.module_params.dynamic.keys()) == 0
    assert len(sim.module_params.static.keys()) == 0
    assert len(sim.params.keys()) == 2
    assert len(sim.params.dynamic.keys()) == 3
    assert len(sim.params.static.keys()) == 1
    # Test that total number of dynamic params is respected
    assert len(sim.params.dynamic.flatten().keys()) == 15


def test_graph():
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
    sim.get_graph(True, True)
    sim.get_graph()


def test_unpack():
    class Sim(Simulator):
        def __init__(self, name="test"):
            super().__init__(name)
            self.cosmo = FlatLambdaCDM("cosmo", h0=None)
            self.epl = EPL("lens", self.cosmo)
            self.sersic = Sersic("source")
            self.z_s = torch.tensor(1.0)
            self.thx, self.thy = get_meshgrid(0.04, 20, 20)

        def forward(self, params):
            alphax, alphay = self.epl.reduced_deflection_angle(x=self.thx, y=self.thy, z_s=self.z_s, params=params) 
            bx = self.thx - alphax
            by = self.thy - alphay
            return self.sersic.brightness(bx, by, params)

    sim = Sim()

    cosmo_params = [0.7]
    lens_params = [0.5, 0., 0., 0.8, 0.5, 1.2, 1.1]
    source_params = [0.0, 0.0, 0.8, 0.0, 1., 0.2, 10.]
    
    # test list input
    x = [torch.tensor(_x) for _x in cosmo_params + lens_params + source_params]
    sim.forward(sim.pack(x))
    sim(x)
    
    # test tensor input
    x_tensor = torch.stack(x)
    sim.forward(sim.pack(x))
    sim(x)
    
    # Test dictionary input: Only module with dynamic parameters are required
    _map = {"cosmo": cosmo_params, "source": source_params, "lens": lens_params}
    x_dict = {k: [torch.tensor(_x) for _x in _map[k]] for k in sim.params.dynamic.keys()}
    print(x_dict)
    sim.forward(sim.pack(x_dict))
    sim(x_dict)
    
    # Test semantic list (one tensor per module)
    _map = [cosmo_params, lens_params, source_params]
    x_semantic = [torch.stack([torch.tensor(_x) for _x in p]) for p in _map]
    sim.forward(sim.pack(x_semantic))
    sim(x_semantic)
    
