from ..simulator import Simulator
from torch.nn.functional import avg_pool2d
from typing import Tuple
import torch

__all__ = ("Lens_Source",)

class Lens_Source(Simulator):
    """Lens image of a source.
    
    Striaghtforward simulator to sample a lensed image of a source
    object. Constructs a sampling grid internally based on the
    pixelscale and gridding parameters. It can automatically upscale
    and fine sample an image. This is the most straightforward
    simulator to view the image if you aready have a lens and source
    chosen.

    Example usage::

       import matplotlib.pyplot as plt
       import caustic

       cosmo = caustic.FlatLambdaCDM()
       lens = caustic.lenses.SIS(cosmology = cosmo, x0 = 0., y0 = 0., th_ein = 1.)
       source = caustic.sources.Sersic(x0 = 0., y0 = 0., q = 0.5, phi = 0.4, n = 2., Re = 1., Ie = 1.)
       sim = caustic.sims.Lens_Source(lens, source, pixelscale = 0.05, gridx = 100, gridy = 100, upsample_factor = 2, z_s = 1.)
       
       img = sim()
       plt.imshow(img, origin = "lower")
       plt.show()

    """
    def __init__(
        self,
        lens,
        src,
        pixelscale: float = 0.05,
        gridx: int = 100,
        gridy: int = 100,
        upsample_factor: int = 1,
        z_s = None,
        name: str = "sim",
    ):
        super().__init__(name)
        
        
        self.lens = lens
        self.src = src
        
        self.add_param("z_s", z_s)

        self.upsample_factor = upsample_factor
        tx = torch.linspace(-0.5 * (pixelscale * gridx), 0.5 * (pixelscale * gridx), gridx*upsample_factor)
        ty = torch.linspace(-0.5 * (pixelscale * gridy), 0.5 * (pixelscale * gridy), gridy*upsample_factor)
        self.grid = torch.meshgrid(tx, ty, indexing = "xy")

        
    def forward(self, params):
        z_s, = self.unpack(params)
        
        bx, by = self.lens.raytrace(*self.grid, z_s, params)
        mu_fine = self.src.brightness(bx, by, params)
        
        return avg_pool2d(mu_fine[None, None], self.upsample_factor, divisor_override = 1).squeeze()
