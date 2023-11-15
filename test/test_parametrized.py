import torch
from torch import vmap
import pytest
import numpy as np
from caustics.sims import Simulator
from caustics.parameter import Parameter
from caustics.lenses import EPL
from caustics.light import Sersic
from caustics.cosmology import FlatLambdaCDM
from utils import setup_simulator


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
    sim, (sim_params, cosmo_params, lens_params, source_params) = setup_simulator()
    n_pix = sim.n_pix
    
    # test list input
    x = sim_params + cosmo_params + lens_params + source_params
    assert sim(x).shape == torch.Size([n_pix, n_pix])
    
    # test tensor input
    x_tensor = torch.stack(x)
    assert sim(x_tensor).shape == torch.Size([n_pix, n_pix])
    
    # Test dictionary input: Only module with dynamic parameters are required
    x_dict = {"simulator": sim_params, "cosmo": cosmo_params, "source": source_params, "lens": lens_params}
    
    assert sim(x_dict).shape == torch.Size([n_pix, n_pix])
    
    # Test semantic list (one tensor per module)
    x_semantic = [torch.stack(module) for module in [sim_params, cosmo_params, lens_params, source_params]]
    assert sim(x_semantic).shape == torch.Size([n_pix, n_pix])


def test_unpack_some_modules_static():
    # same test as above but cosmo is completely static so not fed in the forward method
    sim, (_, _, lens_params, source_params) = setup_simulator(cosmo_static=True, simulator_static=True)
    n_pix = sim.n_pix
    
    # test list input
    x = lens_params + source_params
    assert sim(x).shape == torch.Size([n_pix, n_pix])
    
    # test tensor input
    x_tensor = torch.stack(x)
    assert sim(x_tensor).shape == torch.Size([n_pix, n_pix])
    
    # Test dictionary input: Only module with dynamic parameters are required
    x_dict = {"source": source_params, "lens": lens_params}
    
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
    sim.lens.name = "test_lens"
    assert sim.lens.name == "test_lens"
    assert "test_lens" in sim.params.dynamic.keys()
    assert "test_lens" in sim.cosmo._parents.keys()


def test_parametrized_name_setter_bad_names():
    # Make sure bad names are catched by our added method. Bad names are name which cannot be used as class attributes.
    good_names = ["variable", "_variable", "var_iable2"]
    for name in good_names:
        module = Sersic(name=name)
    bad_names = ["for", "2variable", "variable!", "var-iable", "var iable", "def"]
    for name in bad_names:
        print(name)
        with pytest.raises(NameError):
            module = Sersic(name=name)

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




# TODO make the params attribute -> parameters and make it more intuitive
def test_to_method():
    sim, (sim_params, cosmo_params, lens_params, source_params) = setup_simulator(batched_params=True)
   
    # Check that static params have correct type 
    module = Sersic(x0=0.5)
    assert module.x0.dtype == torch.float32

    module = Sersic(x0=torch.tensor(0.5))
    assert module.x0.dtype == torch.float32

    module = Sersic(x0=np.array(0.5))
    assert module.x0.dtype == torch.float64 # Decided against default type, so gets numpy type here
    
    # Check that all parameters are converted to correct type
    sim.to(dtype=torch.float16)
    assert sim.z_s.dtype is None # dynamic parameter
    assert sim.lens.cosmo.Om0.dtype == torch.float16
    assert sim.cosmo.Om0.dtype == torch.float16
  

def test_parameter_redefinition():
    sim, _ = setup_simulator()
    
    # Make sure the __getattribute__ method works as intended
    assert isinstance(sim.z_s, Parameter)
    # Now test __setattr__ method, we need to catch the redefinition here and keep z_s a parameter
    sim.z_s = 42
    # make sure z_s is still a parameter
    assert sim.z_s.value == torch.tensor(42).float()
    assert sim.z_s.static is True
    sim.z_s = None
    assert sim.z_s.value is None
    assert sim.z_s.dynamic is True
    
