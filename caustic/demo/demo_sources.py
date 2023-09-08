from ..sources import Sersic as cSersic
import torch

__all__ = ("Sersic", )

def Sersic(cosmology = None):
    return cSersic(
        x0 = torch.tensor(0.),
        y0 = torch.tensor(0.),
        q = torch.tensor(0.5),
        phi = 3 * torch.pi / 4.,
        n = torch.tensor(2.),
        Re = torch.tensor(1.),
        Ie = torch.tensor(1.),
        name = "demo sersic",
    )
