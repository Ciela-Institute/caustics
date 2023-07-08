import torch
from torch import vmap
from caustic import Simulator, EPL, Sersic, FlatLambdaCDM
from utils import setup_image_simulator, setup_simulator
import pytest


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
    sim, _ = setup_simulator()
    sim.get_graph(True, True)
    sim.get_graph()


def test_unpack_all_modules_dynamic():
    sim, (cosmo_params, lens_params, source_params) = setup_simulator()
    n_pix = sim.n_pix
    
    # test list input
    x = cosmo_params + lens_params + source_params
    assert sim(x).shape == torch.Size([n_pix, n_pix])
    
    # test tensor input
    x_tensor = torch.stack(x)
    assert sim(x_tensor).shape == torch.Size([n_pix, n_pix])
    
    # Test dictionary input: Only module with dynamic parameters are required
    x_dict = {"cosmo": cosmo_params, "source": source_params, "lens": lens_params}
    print(x_dict)
    assert sim(x_dict).shape == torch.Size([n_pix, n_pix])
    
    # Test semantic list (one tensor per module)
    x_semantic = [torch.stack(module) for module in [cosmo_params, lens_params, source_params]]
    assert sim(x_semantic).shape == torch.Size([n_pix, n_pix])


def test_unpack_some_modules_static():
    # same test as above but cosmo is completely static so not fed in the forward method
    sim, (_, lens_params, source_params) = setup_simulator(cosmo_static=True)
    n_pix = sim.n_pix

    # test list input
    x = lens_params + source_params
    assert sim(x).shape == torch.Size([n_pix, n_pix])
    
    # test tensor input
    x_tensor = torch.stack(x)
    assert sim(x_tensor).shape == torch.Size([n_pix, n_pix])
    
    # Test dictionary input: Only module with dynamic parameters are required
    x_dict = {"source": source_params, "lens": lens_params}
    print(x_dict)
    assert sim(x_dict).shape == torch.Size([n_pix, n_pix])
    
    # Test semantic list (one tensor per module)
    x_semantic = [torch.stack(module) for module in [lens_params, source_params]]
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


def test_vmapped_simulator():
    sim, (cosmo_params, lens_params, source_params) = setup_simulator(batched_params=True)
    n_pix = sim.n_pix
    print(sim.params)
 
    # test list input
    x = cosmo_params + lens_params + source_params
    print(x[0].shape)
    assert vmap(sim)(x).shape == torch.Size([2, n_pix, n_pix])
    
    # test tensor input
    x_tensor = torch.stack(x, dim=1)
    print(x_tensor.shape)
    assert vmap(sim)(x_tensor).shape == torch.Size([2, n_pix, n_pix])
    
    # Test dictionary input: Only module with dynamic parameters are required
    x_dict = {"cosmo": cosmo_params, "source": source_params, "lens": lens_params}
    print(x_dict)
    assert vmap(sim)(x_dict).shape == torch.Size([2, n_pix, n_pix])
    
    # Test semantic list (one tensor per module)
    x_semantic = [cosmo_params, lens_params, source_params]
    assert vmap(sim)(x_semantic).shape == torch.Size([2, n_pix, n_pix])

def test_vmapped_simulator_with_pixelated_modules():
    sim, (cosmo_params, lens_params, kappa, source) = setup_image_simulator(batched_params=True)
    n_pix = sim.n_pix
    print(sim.params)
 
    # test list input
    x = cosmo_params + lens_params + kappa + source 
    print(x[2].shape)
    assert vmap(sim)(x).shape == torch.Size([2, n_pix, n_pix])
    
    # test tensor input
    x_tensor = torch.stack(x, dim=1)
    print(x_tensor.shape)
    assert vmap(sim)(x_tensor).shape == torch.Size([2, n_pix, n_pix])
    
    # Test dictionary input: Only module with dynamic parameters are required
    x_dict = {"cosmo": cosmo_params, "source": source_params, "lens": lens_params, "kappa": kappa}
    print(x_dict)
    assert vmap(sim)(x_dict).shape == torch.Size([2, n_pix, n_pix])
    
    # Test passing tensor in source and kappa instead of list
    x_dict = {"cosmo": cosmo_params, "source": source[0], "lens": lens_params, "kappa": kappa[0]}
    print(x_dict)
    assert vmap(sim)(x_dict).shape == torch.Size([2, n_pix, n_pix])
    
    # Test semantic list (one tensor per module)
    x_semantic = [cosmo_params, lens_params, kappa, source]
    assert vmap(sim)(x_semantic).shape == torch.Size([2, n_pix, n_pix])


# TODO make the params attribute -> parameters and make it more intuitive
def test_to_method():
    sim, (cosmo_params, lens_params, source_params) = setup_simulator(batched_params=True)
    n_pix = sim.n_pix
    print(sim.params)
   
    # Check that static params have correct type 
    module = Sersic(x0=0.5)
    print(module.params.static)
    assert module.params.static.x0.dtype == torch.float32

    module = Sersic(x0=torch.tensor(0.5))
    assert module.params.static.x0.dtype == torch.float32

    module = Sersic(x0=np.array(0.5))
    assert module.params.static.x0.dtype == torch.float32
  

# def test_static_param_definition():
    # sim, (cosmo_params, lens_params, source_params) = setup_simulator(batched_params=True)


    # # Make a test to catch parameters not in order
    # x_wrong_semantic = [lens_params, cosmo_params, source_params]
    # with pytest.raises(L):
        # vmap(sim)(x_wrong_semantic)


