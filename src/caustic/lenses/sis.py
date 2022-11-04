from ..utils import transform_scalar_fn, transform_vector_fn
from .base import AbstractLens


class SIS(AbstractLens):
    def __init__(self, thx0, thy0, th_ein, z_l, cosmology=None, device=None):
        super().__init__(z_l, cosmology, device)
        self.thx0 = thx0
        self.thy0 = thy0
        self.th_ein = th_ein

    @transform_vector_fn
    def alpha(self, thx, thy, z_s):
        th = (thx**2 + thy**2).sqrt()
        return self.th_ein * thx / th, self.th_ein * thy / th

    @transform_scalar_fn
    def Psi(self, thx, thy, z_s):
        th = (thx**2 + thy**2).sqrt()
        return self.th_ein * th

    @transform_scalar_fn
    def kappa(self, thx, thy, z_s):
        th = (thx**2 + thy**2).sqrt()
        return self.th_ein / th
