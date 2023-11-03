import torch
from caustic.lenses import EPL, PixelatedConvergence
from caustic.sims import Simulator
from caustic.cosmology import FlatLambdaCDM
from caustic.light import Pixelated
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


def test_repr():
    cosmo = FlatLambdaCDM()
    print(cosmo.h0)
    assert cosmo.h0.__repr__() == f"Param(value={cosmo.h0.value}, dtype={str(cosmo.h0.dtype)})"
    cosmo = FlatLambdaCDM(h0=None)
    assert cosmo.h0.__repr__() == f"Param(shape={cosmo.h0.shape})"
