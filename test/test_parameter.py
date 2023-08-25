import torch
from caustic import Simulator, FlatLambdaCDM, PixelatedConvergence
from utils import setup_simulator

import pytest

# For future PR currently this test fails
# def test_static_parameter_init():
    # module = EPL(FlatLambdaCDM(h0=0.7, Om0=0.3))
    # print(module.params)
    # module.to(dtype=torch.float16)
    # assert module.params.static.FlatLambdaCDM.h0.value.dtype == torch.float16

def test_shape_error_messages():
    # with pytest.raises(TypeError):
        # # User cannot enter a list, only a tuple for type checking and consistency with torch
        # module = Pixelated(shape=[8, 8])
    
    # with pytest.raises(ValueError):
        # module = Pixelated(shape=(8,))
    
    fov = 7.8
    n_pix = 20
    cosmo = FlatLambdaCDM()
    with pytest.raises(TypeError):
        # User cannot enter a list, only a tuple (because of type checking and consistency with torch)
        PixelatedConvergence(fov, n_pix, cosmo, shape=[8, 8])
    
    with pytest.raises(ValueError): 
        # wrong number of dimensions
        PixelatedConvergence(fov, n_pix, cosmo, shape=(8,))



def test_static_parameter():
    class Module(Simulator):
        def __init__(self, name="module"):
            super().__init__(name)
            self.add_param("a", None)
            self.add_param("b", self.a)
    
        def forward(self, x):
            a, b = self.unpack(x)
            return a, b
    


def test_symbolic_link():
    class Module(Simulator):
        def __init__(self, name="module"):
            super().__init__(name)
            self.add_param("a", None)
            self.add_param("b", self.a) # make a symbolic link between a and b
    
        def forward(self, x):
            a, b = self.unpack(x)
            return a, b

    module = Module()
    x = torch.tensor([42]).float()
    assert module.a.dynamic
    assert module.b.symbolic
    assert module.b.static # make sure symbolic parameter is treated as static

    a, b = module(x)
    assert a == torch.tensor(42).float()
    assert a == b
