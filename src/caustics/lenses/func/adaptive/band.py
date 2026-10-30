"""
The critical band: the ``max_level`` leaves ``det A`` changes sign across.

:func:`build_band` builds it when the lens mesh is frozen, from the samples
of the finest-level pass; :mod:`~caustics.lenses.func.adaptive.curves`
traces it.
"""

from typing import NamedTuple

from ....backend_obj import ArrayLike
from .mesh_backend import mesh_backend
from .lattice import lattice_ij_from_key, lattice_xy


class CriticalBand(NamedTuple):
    """
    The ``max_level`` leaves ``det A`` changes sign across, with ``det A`` at their samples.

    A leaf is in the band when :func:`in_band` holds for the determinants at
    its six samples -- the three vertices and the three edge midpoints, the
    corners of its four red-split children.

    Samples are deduplicated by lattice key: a sample shared by several band
    leaves is one row, with one ``det``, so every leaf sharing it agrees on
    its class. That consistency is what lets
    :func:`~caustics.lenses.func.adaptive.curves.critical_curves_and_caustics`
    chain the crossings of neighbouring leaves.

    Parameters
    ----------
    leaves: ArrayLike
        Shape ``(F,)`` int64 index into ``LensMesh.origin_leaves``.
    samples: ArrayLike
        Shape ``(F, 6)`` int64 index into ``lens``, ``source``, ``det`` and
        ``sigma_min``: each leaf's ``theta_1, theta_2, theta_3, m_1, m_2,
        m_3``, the vertices in the leaf's own order and ``m_i`` opposite
        ``theta_i``. Samples are numbered in ascending lattice-key order.
    lens: ArrayLike
        Shape ``(S, 2)`` float64 lens-plane position of each sample.

        *Unit: arcsec*
    source: ArrayLike
        Shape ``(S, 2)`` float64 source-plane image of each sample.

        *Unit: arcsec*
    det: ArrayLike
        Shape ``(S,)`` float64 ``det A`` at each sample.
    sigma_min: ArrayLike
        Shape ``(S,)`` float64 smallest singular value of ``A`` at each
        sample, which an extension's criterion reads.
    """

    leaves: ArrayLike
    samples: ArrayLike
    lens: ArrayLike
    source: ArrayLike
    det: ArrayLike
    sigma_min: ArrayLike


def in_band(det6):
    """
    True where a leaf's six ``det A`` are finite and not all of one class.

    A sample's class is ``det >= 0``: an exact zero counts as positive, which
    keeps a critical curve through lattice points from vanishing.
    """
    positive = det6 >= 0
    return (
        mesh_backend.all(mesh_backend.isfinite(det6), dim=1)
        & mesh_backend.any(positive, dim=1)
        & mesh_backend.any(~positive, dim=1)
    )


def build_band(lat, leaves, keys6, table_keys, table_values):
    """
    The :class:`CriticalBand` of ``leaves``, its sample values read from a table.

    Parameters
    ----------
    lat: Lattice
    leaves: ArrayLike
        ``(F,)`` int64 index of each band leaf into ``origin_leaves``.
    keys6: ArrayLike
        ``(F, 6)`` int64 lattice keys of each leaf's ``theta_1, theta_2,
        theta_3, m_1, m_2, m_3``.
    table_keys: ArrayLike
        ``(T,)`` int64 ascending keys, holding every key of ``keys6``.
    table_values: ArrayLike
        ``(T, 4)`` float64 ``(bx, by, det A, sigma_min A)`` at ``table_keys``.

    Returns
    -------
    CriticalBand
        With samples numbered in ascending key order.
    """
    keys, samples = mesh_backend.unique(keys6.reshape(-1), return_inverse=True)
    values = table_values[mesh_backend.searchsorted(table_keys, keys)]
    return CriticalBand(
        leaves=leaves,
        samples=samples.reshape(-1, 6),
        lens=lattice_xy(lat, lattice_ij_from_key(lat, keys)),
        source=values[:, :2],
        det=values[:, 2],
        sigma_min=values[:, 3],
    )
