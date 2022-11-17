import torch

from ..utils import transform_scalar_fn, transform_vector_fn
from .base import AbstractLens


class Point(AbstractLens):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__(device)

    def alpha(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein):
        @transform_vector_fn(thx0, thy0)
        def helper(thx, thy):
            th = (thx**2 + thy**2).sqrt()
            return th_ein**2 / th, th_ein**2 / th

        return helper(thx, thy)

    def Psi(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein):
        @transform_scalar_fn(thx0, thy0)
        def helper(thx, thy):
            th = (thx**2 + thy**2).sqrt()
            return th_ein**2 * th.log()

        return helper(thx, thy)

    def kappa(self, thx, thy, z_l, z_s, cosmology, thx0, thy0, th_ein):
        @transform_scalar_fn(thx0, thy0)
        def helper(thx, thy):
            return torch.where((thx == 0) & (thy == 0), torch.inf, 0.0)

        return helper(thx, thy)
