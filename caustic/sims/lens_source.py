from ..simulator import Simulator
from torch.nn.functional import avg_pool2d

__all__ = ("Lens_Source",)

class Lens_Source(Simulator):
    """Lens image of a source.
    
    Striaghtforward simulator to sample a lensed image of a source object.

    Example usage::

       import matplotlib.pyplot as plt
       import caustic, torch
       
       lens = caustic.demo.SIS()
       source = caustic.demo.Sersic()
       sim = caustic.sims.Lens_Source(lens, source)
       
       thx, thy = caustic.demo.grid()
       img = sim([torch.tensor(1.)], thx, thy, 2)
       plt.imshow(img, origin = "lower")
       plt.show()
    """
    def __init__(
        self,
        lens,
        src,
        z_s = None,
        name: str = "sim",
    ):
        super().__init__(name)
        
        
        self.lens = lens
        self.src = src
        
        self.add_param("z_s", z_s)
        
    def forward(self, params, thx, thy, upsample_factor = 1):
        z_s, = self.unpack(params)
        
        bx, by = self.lens.raytrace(thx, thy, z_s, params)
        mu_fine = self.src.brightness(bx, by, params)
        
        return avg_pool2d(mu_fine.squeeze()[None, None], upsample_factor)[0, 0]
