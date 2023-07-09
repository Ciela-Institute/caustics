from . import Parametrized

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
        packed_args = self.pack(args[0])
        return self.forward(packed_args, *args[1:], **kwargs)
        # packed_args, batched = self.pack(args[0])
        # out = self.forward(packed_args, *args[1:], **kwargs)
        # if batched:
            # return out
        # else:
            # return out[0] # removes the singleton batch dimension used internally
