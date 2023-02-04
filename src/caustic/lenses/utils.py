import functorch
from torch import Tensor

from ..utils import vmap_n


def _pix_magnification(
    raytrace, thx, thy, z_l, z_s, cosmology, *args, **kwargs
) -> Tensor:
    jac = functorch.jacfwd(raytrace, (0, 1))(
        thx, thy, z_l, z_s, cosmology, *args, **kwargs
    )
    return 1 / (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]).abs()


def get_magnification(
    raytrace, thx, thy, z_l, z_s, cosmology, *args, **kwargs
) -> Tensor:
    n_args = 3 + len(args) + len(kwargs)
    in_dims = (None,) + 2 * (-1,) + n_args * (None,)
    return vmap_n(_pix_magnification, 2, in_dims)(
        raytrace, thx, thy, z_l, z_s, cosmology, *args, **kwargs
    )
