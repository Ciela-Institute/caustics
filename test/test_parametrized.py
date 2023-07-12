import torch
from caustic import Simulator, EPL, Sersic, FlatLambdaCDM
from caustic.utils import get_meshgrid


def test_params():
    # Test common simulator structure
    class Sim(Simulator):
        def __init__(self):
            super().__init__()
            self.cosmo = FlatLambdaCDM(h0=None)
            self.epl = EPL(self.cosmo)
            self.sersic = Sersic()
            self.add_param("z_s", 1.0)
    
    sim = Sim()
    assert len(sim.module_params) == 2 # dynamic and static
    assert len(sim.module_params.dynamic) == 0 # simulator has no dynmaic params
    assert len(sim.module_params.static) == 1 # and 1 static param (z_s)
    assert len(sim.params) == 2 # dynamic and static
    assert len(sim.params.dynamic) == 3 # cosmo, epl and sersic
    assert len(sim.params.static) == 2 # simulator and cosmo have static params
    assert len(sim.params.dynamic.flatten()) == 15 # total number of params


def test_graph():
    # Test common simulator structure
    class Sim(Simulator):
        def __init__(self, name="test_simulator"):
            super().__init__(name)
            self.cosmo = FlatLambdaCDM(h0=None, name="cosmo")
            self.lens = EPL(self.cosmo, name="lens")
            self.source = Sersic(name="source")
            self.z_s = torch.tensor(1.0)
            self.thx, self.thy = get_meshgrid(0.04, 20, 20)

    sim = Sim()
    sim.get_graph(True, True)
    sim.get_graph()


def test_unpack_all_modules_dynamic():
    n_pix = 20
    class Sim(Simulator):
        def __init__(self, name="test"):
            super().__init__(name)
            self.cosmo = FlatLambdaCDM(h0=None, name="cosmo")
            self.epl = EPL(self.cosmo, name="lens")
            self.sersic = Sersic(name="source")
            self.z_s = torch.tensor(1.0)
            self.thx, self.thy = get_meshgrid(0.04, n_pix, n_pix)

        def forward(self, params):
            alphax, alphay = self.epl.reduced_deflection_angle(self.thx, self.thy, self.z_s, params) 
            bx = self.thx - alphax
            by = self.thy - alphay
            return self.sersic.brightness(bx, by, params)

    sim = Sim()

    cosmo_params = [0.7]
    lens_params = [0.5, 0., 0., 0.8, 0.5, 1.2, 1.1]
    source_params = [0.0, 0.0, 0.8, 0.0, 1., 0.2, 10.]
    
    # test list input
    x = [torch.tensor(_x) for _x in cosmo_params + lens_params + source_params]
    assert sim(x).shape == torch.Size([n_pix, n_pix])
    
    # test tensor input
    x_tensor = torch.stack(x)
    assert sim(x_tensor).shape == torch.Size([n_pix, n_pix])
    
    # Test dictionary input: Only module with dynamic parameters are required
    _map = {"cosmo": cosmo_params, "source": source_params, "lens": lens_params}
    x_dict = {k: [torch.tensor(_x) for _x in _map[k]] for k in sim.params.dynamic.keys()}
    print(x_dict)
    assert sim(x_dict).shape == torch.Size([n_pix, n_pix])
    
    # Test semantic list (one tensor per module)
    _map = [cosmo_params, lens_params, source_params]
    x_semantic = [torch.stack([torch.tensor(_x) for _x in p]) for p in _map]
    assert sim(x_semantic).shape == torch.Size([n_pix, n_pix])


def test_unpack_some_modules_static():
    # Repeat previous test, but with one module completely static
    n_pix = 20
    class Sim(Simulator):
        def __init__(self, name="test"):
            super().__init__(name)
            self.cosmo = FlatLambdaCDM(name="cosmo")
            self.epl = EPL(self.cosmo, name="lens")
            self.sersic = Sersic(name="source")
            self.z_s = torch.tensor(1.0)
            self.thx, self.thy = get_meshgrid(0.04, n_pix, n_pix)

        def forward(self, params):
            alphax, alphay = self.epl.reduced_deflection_angle(x=self.thx, y=self.thy, z_s=self.z_s, params=params) 
            bx = self.thx - alphax
            by = self.thy - alphay
            return self.sersic.brightness(bx, by, params)

    sim = Sim()

    lens_params = [0.5, 0., 0., 0.8, 0.5, 1.2, 1.1]
    source_params = [0.0, 0.0, 0.8, 0.0, 1., 0.2, 10.]
    
    # test list input
    x = [torch.tensor(_x) for _x in lens_params + source_params]
    assert sim(x).shape == torch.Size([n_pix, n_pix])
    
    # test tensor input
    x_tensor = torch.stack(x)
    assert sim(x_tensor).shape == torch.Size([n_pix, n_pix])
    
    # Test dictionary input: Only module with dynamic parameters are required
    _map = {"source": source_params, "lens": lens_params}
    x_dict = {k: [torch.tensor(_x) for _x in _map[k]] for k in sim.params.dynamic.keys()}
    print(x_dict)
    assert sim(x_dict).shape == torch.Size([n_pix, n_pix])

    sim(x_dict)
    
    # Test semantic list (one tensor per module)
    _map = [lens_params, source_params]
    x_semantic = [torch.stack([torch.tensor(_x) for _x in p]) for p in _map]
    assert sim(x_semantic).shape == torch.Size([n_pix, n_pix])
    

def test_default_names():
    cosmo = FlatLambdaCDM()
    assert cosmo.name == "FlatLambdaCDM"
    epl = EPL(cosmo)
    assert epl.name == "EPL"
    source = Sersic()
    assert source.name == "Sersic"


def test_parametrized_name_setter():
    class Sim(Simulator):
        def __init__(self):
            super().__init__()
            self.cosmo = FlatLambdaCDM()
            self.lens = EPL(self.cosmo, name="lens")
            self.source = Sersic(name="source")
    
    sim = Sim()
    assert sim.name == "Sim"
    sim.name = "Test"
    assert sim.name == "Test"

    # Check that DAG in SIM is being update updated
    sim.lens.name = "Test Lens"
    assert sim.lens.name == "Test Lens"
    assert "Test Lens" in sim.params.dynamic.keys()
    assert "Test Lens" in sim.cosmo._parents.keys()


def test_parametrized_name_collision():
    # Case 1: Name collision in children of simulator
    class Sim(Simulator):
        def __init__(self):
            super().__init__()
            self.cosmo = FlatLambdaCDM(h0=None)
            # These two module are identical and will create a name collision
            self.lens1 = EPL(self.cosmo)
            self.lens2 = EPL(self.cosmo)
    
    sim = Sim()
    # Current way names are updated. Could be chnaged so that all params in collision
    # Get a number
    assert sim.lens1.name == "EPL"
    assert sim.lens2.name == "EPL_1"

    # Case 2: name collision in parents of a module
    cosmo = FlatLambdaCDM(h0=None)
    lens = EPL(cosmo)
    class Sim(Simulator):
        def __init__(self):
            super().__init__()
            self.lens = lens
    
    sim1 = Sim()
    sim2 = Sim()
    assert sim1.name == "Sim"
    assert sim2.name == "Sim_1"
    assert "Sim_1" in lens._parents.keys()
    assert "Sim" in lens._parents.keys()
