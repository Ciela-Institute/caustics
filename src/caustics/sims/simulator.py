from typing import Optional

from ..packed import Packed
from ..parametrized import Parametrized, unpack

__all__ = ("Simulator",)


class Simulator(Parametrized):
    """A caustics simulator using Parametrized framework.

    Defines a simulator class which is a callable function that
    operates on the Parametrized framework. Users define the `forward`
    method which takes as its first argument an object which can be
    packed, all other args and kwargs are simply passed to the forward
    method.

    See `Parametrized` for details on how to add/access parameters.

    """

    @unpack
    def __call__(self, *args, params: Optional[Packed] = None, **kwargs):
        return self.forward(*args, params=params, **kwargs)
