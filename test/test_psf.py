import torch
from caustic import GaussianPSF, Airy, Moffat
from caustic.utils import get_meshgrid
import numpy as np

#TODO compare with reference implementation of these profiles

def test_gaussian_psf():
    psf = GaussianPSF()
    width = torch.tensor([0.05, 0.1]).view(-1, 1)
    x, y = get_meshgrid(0.04, 21, 21)
    _func = lambda params: psf.kernel(x, y, psf.pack(params))

    kernel = torch.vmap(_func)(width) 
    assert kernel.sum() == 2

    psf = GaussianPSF(q=None, phi=None, x0=None, y0=None)
    width = torch.tensor([0.05, 0.1]).view(-1, 1)
    q = torch.tensor([0.9, 0.6]).view(-1, 1)
    phi = torch.tensor([0.1, 1.57]).view(-1, 1)
    x0 = torch.tensor([0.03, 0.09]).view(-1, 1)
    y0 = torch.tensor([-0.03, 0.0]).view(-1, 1)
    x, y = get_meshgrid(0.04, 21, 21)
    params = [width, q, phi, x0, y0]

    kernel = torch.vmap(_func)(params)
    assert np.isclose(kernel.sum(), 2, rtol=1e-5)
    

def test_airy():
    psf = Airy()
    scale = torch.tensor([10, 50]).view(-1, 1)
    x, y = get_meshgrid(0.04, 100, 100)
    _func = lambda params: psf.kernel(x, y, psf.pack(params))

    kernel = torch.vmap(_func)(scale)
    assert np.isclose(kernel.sum(), 2, atol=1e-5)

    psf = Airy(x0=None, y0=None, I=None)
    scale = torch.tensor([10, 100]).view(-1, 1)
    x0 = torch.tensor([0.03, 0.08]).view(-1, 1)
    y0 = torch.tensor([-0.03, 0.00]).view(-1, 1)
    I = torch.tensor([1, 2.]).view(-1, 1)
    x, y = get_meshgrid(0.04, 21, 21)
    params = [scale, x0, y0, I]

    kernel = torch.vmap(_func)(params)

    assert np.isclose(kernel.sum(), 3, atol=1e-5)
    

def test_moffat():
    psf = Moffat()
    n = torch.tensor([1, 1]).view(-1, 1)
    rd = torch.tensor([0.04, 0.2]).view(-1, 1)
    params = [n, rd]
    x, y = get_meshgrid(0.04, 100, 100)
    _func = lambda params: psf.kernel(x, y, psf.pack(params))

    kernel = torch.vmap(_func)(params)
    
    assert np.isclose(kernel.sum(), 2, atol=1e-5)
    
    psf = Moffat(q=None, phi=None, x0=None, y0=None, I=None)
    n = torch.tensor([1, 2]).view(-1, 1)
    rd = torch.tensor([0.04, 0.08]).view(-1, 1)
    q = torch.tensor([0.3, 0.8]).view(-1, 1)
    phi = torch.tensor([0.5, 1.57]).view(-1, 1)
    x0 = torch.tensor([0.03, 0.08]).view(-1, 1)
    y0 = torch.tensor([-0.03, 0.00]).view(-1, 1)
    I = torch.tensor([1, 2.]).view(-1, 1)
    x, y = get_meshgrid(0.04, 21, 21)
    params = [n, rd, q, phi, x0, y0, I]

    kernel = torch.vmap(_func)(params)
    
    assert np.isclose(kernel.sum(), 3, atol=1e-5)
