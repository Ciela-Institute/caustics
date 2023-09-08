from ..lenses import SIS as cSIS, SIE as cSIE
from ..cosmology import FlatLambdaCDM
import torch

__all__ = ("SIS", "SIE")

def SIS(cosmology = None):
    if cosmology is None:
        cosmology = FlatLambdaCDM(name = "cosmo")
        cosmology.to(dtype=torch.float32)

    return cSIS(
        cosmology = cosmology,
        z_l = torch.tensor(0.5),
        x0 = torch.tensor(0.),
        y0 = torch.tensor(0.),
        th_ein = torch.tensor(1.),
        name = "demo sis",
    )

def SIE(cosmology = None):
    if cosmology is None:
        cosmology = FlatLambdaCDM(name = "cosmo")
        cosmology.to(dtype=torch.float32)
    return cSIE(
        cosmology = cosmology,
        z_l = torch.tensor(0.5),
        x0 = torch.tensor(0.),
        y0 = torch.tensor(0.),
        q = torch.tensor(0.5),
        phi = torch.pi / 3.,
        b = torch.tensor(1.),
        name = "demo sie",
    )
