import torch

__all__ = ("grid", )

def grid(upsample_factor = 2):

    t = torch.linspace(-2.5, 2.5, 100*upsample_factor)
    thx, thy = torch.meshgrid(t, t, indexing = "xy")

    return thx, thy
