"""
The critical band: the ``max_level`` leaves the mesh's ``det A`` changes sign across.

``A`` is the mesh's Jacobian, not the lens's: the fitted Jacobians its
vertices carry (:func:`jacobian_det6`), and where they vary too much across
a leaf to trust (:func:`resolved`), a ``det A`` measured from the raytrace
at the leaf's samples. :func:`build_band` builds the band when the lens
mesh is frozen; :mod:`~caustics.lenses.func.adaptive.curves` traces it.
"""

from typing import NamedTuple

from ....backend_obj import ArrayLike
from .mesh_backend import mesh_backend
from .lattice import lattice_ij_from_key, lattice_xy

RESOLVED_SPREAD = 4.0
"""Largest Frobenius norm of a difference of two vertex Jacobians that
:func:`resolved` still trusts, one value for every lens."""

MEASURE_OFFSET = 1.0 / 1024
"""Offset of a measured ``det A``'s central differences, as a fraction of
the mesh's ``min_img_sep``."""


class CriticalBand(NamedTuple):
    """
    The ``max_level`` leaves the mesh's ``det A`` changes sign across, with ``det A`` at their samples.

    A leaf is in the band when :func:`in_band` holds for the determinants at
    its six samples -- the three vertices and the three edge midpoints, the
    corners of its four red-split children. Each is :func:`jacobian_det6`
    of the leaf's three vertex Jacobians, except at a measured sample, one
    of the six of a finest-level leaf that is not :func:`resolved` and has
    no non-finite image: there it is ``det`` of the central differences of
    the raytrace at four points :data:`MEASURE_OFFSET` ``* min_img_sep``
    away along the axes, NaN where one of their images is non-finite, for
    every leaf that holds the sample.

    Samples are deduplicated by lattice key: a sample shared by several band
    leaves is one row, with one ``det``, so every leaf sharing it agrees on
    its class. That consistency is what lets
    :func:`~caustics.lenses.func.adaptive.curves.critical_curves_and_caustics`
    chain the crossings of neighbouring leaves.

    Parameters
    ----------
    leaves: ArrayLike
        Shape ``(F,)`` int64 index into ``LensMesh.origin_leaves``, ascending.
    samples: ArrayLike
        Shape ``(F, 6)`` int64 index into ``lens``, ``source`` and ``det``:
        each leaf's ``theta_1, theta_2, theta_3, m_1, m_2, m_3``, the
        vertices in the leaf's own order and ``m_i`` opposite ``theta_i``.
        Samples are numbered in ascending lattice-key order.
    lens: ArrayLike
        Shape ``(S, 2)`` float64 lens-plane position of each sample.

        *Unit: arcsec*
    source: ArrayLike
        Shape ``(S, 2)`` float64 source-plane image of each sample.

        *Unit: arcsec*
    det: ArrayLike
        Shape ``(S,)`` float64 ``det`` of the mesh's Jacobian at each
        sample, fitted or measured as above. The traced critical curves are
        its zero set.
    measured_ij: ArrayLike
        Shape ``(K, 2)`` int64 lattice coordinates of every measured sample
        of the mesh, band or not, in ascending lattice-key order. An
        extension reads them rather than measure them again.
    measured_det: ArrayLike
        Shape ``(K,)`` float64 the measured ``det A`` at each.
    """

    leaves: ArrayLike
    samples: ArrayLike
    lens: ArrayLike
    source: ArrayLike
    det: ArrayLike
    measured_ij: ArrayLike
    measured_det: ArrayLike


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


def _det2(jac):
    """``det`` of each 2x2 matrix in ``jac`` (``(..., 2, 2)``), by the explicit formula."""
    return jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]


def jacobian_det6(jacobian3):
    """
    ``det A`` at a triangle's six samples, from its three vertex Jacobians.

    Columns ``0..2`` are ``det J_i`` at the vertices; column ``3 + i`` is
    ``det((J_j + J_k) / 2)`` at the midpoint of the edge opposite vertex
    ``i``. The midpoint Jacobian is the entrywise sum halved, and the
    determinant the explicit 2x2 formula, so two triangles sharing an edge
    get the same bits from either order of its two ends.

    Parameters
    ----------
    jacobian3: ArrayLike
        Shape ``(n, 3, 2, 2)`` float64 Jacobian at each vertex.

    Returns
    -------
    ArrayLike
        Shape ``(n, 6)`` float64, ordered ``theta_1, theta_2, theta_3, m_1,
        m_2, m_3``.
    """
    mid = [
        (jacobian3[:, (i + 1) % 3] + jacobian3[:, (i + 2) % 3]) * 0.5 for i in range(3)
    ]
    return mesh_backend.stack(
        [_det2(jacobian3[:, i]) for i in range(3)] + [_det2(m) for m in mid],
        dim=1,
    )


def resolved(jacobian3):
    """
    True where the Jacobian varies little enough across a triangle to trust its six ``det A``.

    The largest Frobenius norm of ``J_i - J_j`` over the three vertex pairs
    must be below ``RESOLVED_SPREAD``, read at call time. A NaN anywhere
    gives False.

    Parameters
    ----------
    jacobian3: ArrayLike
        Shape ``(n, 3, 2, 2)`` float64 Jacobian at each vertex.

    Returns
    -------
    ArrayLike
        Shape ``(n,)`` bool.
    """
    norms = [
        mesh_backend.sqrt(
            ((jacobian3[:, i] - jacobian3[:, (i + 1) % 3]) ** 2).sum(-1).sum(-1)
        )
        for i in range(3)
    ]
    spread = mesh_backend.max(mesh_backend.stack(norms, dim=1), dim=1)
    return spread < RESOLVED_SPREAD


def build_band(lat, leaves, keys6, source6, det6, measured_ij, measured_det):
    """
    The :class:`CriticalBand` of ``leaves``, from each leaf's values at its six samples.

    A sample several leaves hold carries identical values in each of them,
    so filling its row from all of them is deterministic.

    Parameters
    ----------
    lat: Lattice
    leaves: ArrayLike
        ``(F,)`` int64 index of each band leaf into ``origin_leaves``.
    keys6: ArrayLike
        ``(F, 6)`` int64 lattice keys of each leaf's ``theta_1, theta_2,
        theta_3, m_1, m_2, m_3``.
    source6: ArrayLike
        ``(F, 6, 2)`` float64 image at each of them.

        *Unit: arcsec*
    det6: ArrayLike
        ``(F, 6)`` float64 ``det`` of the mesh's Jacobian at each of them.
    measured_ij, measured_det: ArrayLike
        ``(K, 2)`` int64 and ``(K,)`` float64, the mesh's measured samples,
        stored as they are.

    Returns
    -------
    CriticalBand
        With samples numbered in ascending key order.
    """
    keys, samples = mesh_backend.unique(keys6.reshape(-1), return_inverse=True)
    n, device = keys.shape[0], mesh_backend.device(keys6)
    f64 = mesh_backend.float64
    # Repeated indices carry identical values.
    source = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n, 2), dtype=f64, device=device),
        samples,
        source6.reshape(-1, 2),
    )
    det = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n,), dtype=f64, device=device), samples, det6.reshape(-1)
    )
    return CriticalBand(
        leaves=leaves,
        samples=samples.reshape(-1, 6),
        lens=lattice_xy(lat, lattice_ij_from_key(lat, keys)),
        source=source,
        det=det,
        measured_ij=measured_ij,
        measured_det=measured_det,
    )
