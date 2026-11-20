"""Lens methods compiled once per lens, with their parameter values as inputs."""

import weakref
from types import MethodType
from typing import Callable, Optional

from caskade import ActiveContext, Module, Param

from .backend_obj import backend

__all__ = ("jit_forward", "JitForward")

# Per module, the `(pure, on_points)` pair for each (method name, fixed keyword
# arguments). Keyed weakly and never an attribute of the module, which deepcopy
# and cloudpickle would carry. The pairs hold the module only through
# `weakref.ref`: a value that references its own key keeps that key alive
# forever in a `WeakKeyDictionary`, and jax's own caches, keyed on `pure`,
# would do the same.
_ENTRIES: "weakref.WeakKeyDictionary[Module, dict]" = weakref.WeakKeyDictionary()


def _param_nodes(module):
    # Topological order ignores static versus dynamic. caskade's `all_params`
    # groups by status, so a status flip would reorder same-shaped inputs and
    # a cached program would read them in the wrong slots.
    return tuple(n for n in module.topological_ordering() if isinstance(n, Param))


def _make(ref, name, kwargs):
    def pure(*args):
        *args, values = args
        module = ref()
        nodes = _param_nodes(module)
        saved = [p._value for p in nodes]
        # Entering a context on an online module raises; it is already active then.
        context = None if module.online else ActiveContext(module)
        if context is not None:
            context.__enter__()
        try:
            given = iter(values)
            for p in nodes:
                # A pointer recomputes from the traced values; a value it
                # cached before must not be read.
                p._value = None if p.pointer else next(given)
            return getattr(module, name)(*args, **kwargs)
        finally:
            # Restore before leaving the context, so no tracer stays in the
            # lens's state and an outer context finds its own values again.
            for p, value in zip(nodes, saved):
                p._value = value
            if context is not None:
                context.__exit__(None, None, None)

    def on_points(xy, values):
        # Traced inside the caller's own jitted program, so `pure` is not jitted here.
        out = pure(*(xy[..., i] for i in range(xy.shape[-1])), values)
        return backend.stack(out, dim=-1)

    return backend.jit(pure), on_points


class JitForward:
    """A lens method compiled with every parameter value as an input.

    Made by :func:`jit_forward`; see there for its behaviour and limits.

    Attributes
    ----------
    jitted: bool
        True under a compiling backend (jax), unless the lens's class sets
        ``_jit_forward_eager``.
    pure: Callable or None
        The jitted ``(*args, values) -> output``, with ``values`` from
        :meth:`values` as the last positional argument. None when not
        ``jitted``.
    on_points: Callable or None
        ``(xy, values) -> backend.stack(output, dim=-1)``, with ``xy[..., i]``
        as the method's ``i``-th argument, for a caller that jits it within
        its own program. Not jitted itself. None when not ``jitted``.
    """

    def __init__(self, method: MethodType, kwargs: dict):
        self.method = method
        self.kwargs = kwargs
        self.module: Module = method.__self__
        self.jitted = backend.compiles and not getattr(
            type(self.module), "_jit_forward_eager", False
        )
        self.pure: Optional[Callable] = None
        self.on_points: Optional[Callable] = None
        if self.jitted:
            entries = _ENTRIES.setdefault(self.module, {})
            key = (method.__name__, tuple(sorted(kwargs.items())))
            if key not in entries:
                entries[key] = _make(weakref.ref(self.module), method.__name__, kwargs)
            self.pure, self.on_points = entries[key]

    def values(self, params=None) -> tuple:
        """The value of every non-pointer parameter, in topological order.

        Parameters
        ----------
        params: ArrayLike, Sequence or Mapping, optional
            Dynamic parameter values, in any form caskade's ``@forward``
            accepts. Ignored while the lens is online, where the values in
            force are used.

        Returns
        -------
        tuple
            One value per non-pointer ``Param`` below the lens.
        """
        nodes = [p for p in _param_nodes(self.module) if not p.pointer]
        if self.module.online or params is None:
            return tuple(p.value for p in nodes)
        with ActiveContext(self.module):
            self.module.fill_params(params)
            return tuple(p.value for p in nodes)

    def __call__(self, *args, params=None):
        if self.jitted:
            return self.pure(*args, self.values(params))
        if params is None:
            return self.method(*args, **self.kwargs)
        return self.method(*args, params=params, **self.kwargs)


def jit_forward(method: MethodType, **kwargs) -> JitForward:
    """Compile a lens's ``@forward`` method once, with its parameters as inputs.

    Under jax, the method runs as a jitted pure function of its arguments and
    the value of every non-pointer parameter, static and dynamic alike, so new
    parameter values reuse the compiled program. Pointer parameters are
    recomputed from those values inside it. The program is cached per (lens
    object, method, fixed keyword arguments) and freed with the lens. Under
    torch, and for a ``ThickLens``, the method runs eagerly.

    The wrapper works both inside and outside a ``@forward`` call of the lens,
    and leaves the lens's parameter state as it found it.

    Parameters
    ----------
    method: MethodType
        A bound ``@forward`` method of a lens, e.g. ``lens.raytrace``.
    **kwargs
        Fixed keyword arguments for ``method``, e.g. ``method="autograd"``.
        They must be hashable.

    Returns
    -------
    JitForward
        Called as ``(*args, params=None)``, like ``method``. While the lens is
        online, as inside one of its ``@forward`` methods, the values in force
        are used; otherwise ``params`` fills the dynamic parameters as
        ``@forward`` would.

    Notes
    -----
    - A new input shape compiles a new program, so callers pad inputs to
      stable lengths.
    - A compiled program holds the lens as it was when traced. After a call,
      changing the parameter graph or a plain attribute, or calling
      ``.to(...)``, needs a new lens object.
    - A ``ThickLens`` runs eagerly: ``Multiplane`` sorts its planes by
      redshift in Python, which traced values cannot do.

    Examples
    --------
    ::

        magnification = caustics.jit_forward(lens.magnification)
        for params in samples:
            mu = magnification(x, y, params=params)  # compiles on the first call only
    """
    return JitForward(method, kwargs)
