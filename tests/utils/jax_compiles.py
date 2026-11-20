"""Counts XLA backend compiles so tests can assert nothing (re)compiles.

Torch runs never need this, so ``jax`` is imported lazily on first use
rather than at module import time.
"""

_count = 0
_registered = False


def compile_count() -> int:
    """XLA backend compiles seen so far in this process."""
    global _registered
    if not _registered:
        import jax

        def _count_compiles(event, duration_secs, **kwargs):
            global _count
            if event == "/jax/core/compile/backend_compile_duration":
                _count += 1

        jax.monitoring.register_event_duration_secs_listener(_count_compiles)
        _registered = True
    return _count
