"""
The backend the mesh bookkeeping runs on, and conversions to and from the user's.

The bookkeeping is gathers, sorts and searches on arrays whose sizes depend
on the data. Torch runs them as they come; eager jax compiles each anew for
nearly every call, so under jax the bookkeeping runs on torch on the CPU.
Lens calls and root finding stay in the user's backend, and the public
functions convert at their boundary (:func:`to_mesh`, :func:`to_user`).
Under torch ``mesh_backend`` is ``backend`` itself and nothing converts.
"""

import importlib

import numpy as np

from ....backend_obj import Backend, backend


class TorchMeshBackend(Backend):
    """
    ``Backend``'s torch methods, without switching caskade's global backend.

    ``as_array`` also takes jax arrays, and sequences holding them, through a
    numpy copy.
    """

    def __init__(self):
        self.module = importlib.import_module("torch")
        self.setup_torch()
        self._backend = "torch"

    def _readable(self, array):
        """``array``, or a numpy copy of it where torch cannot read it."""
        if isinstance(array, (list, tuple)) or type(array).__module__.startswith("jax"):
            return np.array(array)
        return array

    def _as_array_torch(self, array, dtype=None, device=None):
        return super()._as_array_torch(self._readable(array), dtype, device)


mesh_backend = backend if backend.backend == "torch" else TorchMeshBackend()


def map_arrays(value, fn):
    """
    ``fn`` on every array of ``value``, recursively through tuples and NamedTuples.

    Anything without a ``shape`` passes through. ``value`` itself comes back
    when ``fn`` returns every array unchanged.
    """
    if isinstance(value, tuple):
        items = [map_arrays(v, fn) for v in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if hasattr(value, "shape"):
        return fn(value)
    return value


if mesh_backend is backend:

    def to_mesh(value):
        """``value`` with its arrays as ``mesh_backend`` arrays: under torch, ``value``."""
        return value

    def to_user(value, device=None):
        """``value`` with its arrays as ``backend`` arrays on ``device``."""
        return map_arrays(value, lambda a: backend.to(a, device=device))

else:
    _Tensor = mesh_backend.module.Tensor

    def to_mesh(value):
        """``value`` with its arrays as ``mesh_backend`` arrays, copied."""
        # A copy: the bookkeeping writes in place, and a jax buffer is immutable.
        return map_arrays(
            value,
            lambda a: (
                a
                if isinstance(a, _Tensor)
                else mesh_backend.module.as_tensor(np.array(a))
            ),
        )

    def to_user(value, device=None):
        """``value`` with its arrays as ``backend`` arrays on ``device``, copied."""
        # Copied on the host, so no later in-place write reaches it, then put:
        # ``jnp.array`` and ``jnp.asarray`` compile a program for every new shape.
        return map_arrays(
            value,
            lambda a: (
                backend.to(a.numpy().copy(), device=device)
                if isinstance(a, _Tensor)
                else a
            ),
        )
