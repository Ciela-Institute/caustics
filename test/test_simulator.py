from math import pi

import torch

from caustic.sims import Lens_Source
from caustic.cosmology import FlatLambdaCDM
from caustic.lenses import SIE
from caustic.sources import Sersic
from caustic.utils import get_meshgrid, gaussian

def test_simulator_runs():

    # Model
    cosmology = FlatLambdaCDM(name="cosmo")
    lensmass = SIE(name="lens", cosmology=cosmology, z_l = 1., x0 = 0., y0 = 0.01, q = 0.5, phi = pi/3., b = 1.)

    source = Sersic(name="source", x0 = 0.01, y0 = -0.03, q = 0.6, phi = -pi/4, n = 2., Re = 0.5, Ie = 1.)
    lenslight = Sersic(name="lenslight", x0 = 0.0, y0 = 0.01, q = 0.7, phi = pi/4, n = 3., Re = 0.7, Ie = 1.)

    psf = gaussian(11, 11, 0.05, 0.2, upsample = 2)
    
    sim = Lens_Source(lens_mass = lensmass, source_light = source, lens_light = lenslight, psf = psf, z_s = 2.)

    assert torch.all(torch.isfinite(sim()))
    assert torch.all(torch.isfinite(sim({}, sourcelight=True, lenslight=True, lenssource=True, psfconvolve=False)))
    assert torch.all(torch.isfinite(sim({}, sourcelight=True, lenslight=True, lenssource=False, psfconvolve=True)))
    assert torch.all(torch.isfinite(sim({}, sourcelight=True, lenslight=False, lenssource=True, psfconvolve=True)))
    assert torch.all(torch.isfinite(sim({}, sourcelight=False, lenslight=True, lenssource=True, psfconvolve=True)))
