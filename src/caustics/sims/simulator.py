from ..parametrized import Parametrized

__all__ = ("Simulator",)


class Simulator(Parametrized):
    """A caustic simulator using Parametrized framework.

    Defines a simulator class which is a callable function that
    operates on the Parametrized framework. Users define the `forward`
    method which takes as its first argument an object which can be
    packed, all other args and kwargs are simply passed to the forward
    method.

    See `Parametrized` for details on how to add/access parameters.

    """

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            packed_args = self.pack(args[0])
            rest_args = args[1:]
        else:
            packed_args = self.pack()
            rest_args = tuple()

        return self.forward(packed_args, *rest_args, **kwargs)
