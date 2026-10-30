"""
The lens mesh's refinement criterion, and the ``LEAF_*`` status flags it records.

:func:`lens_status` decides, vectorized over triangles, whether a triangle's
affine model of the lens map is accurate to ``min_img_sep`` and whether a
critical curve runs between its samples.
"""

from .mesh_backend import mesh_backend
from .geometry import (
    CHILD_VERTEX_INDICES,
    COMPOSE,
    PINV0,
    sigma_min_2x2,
)

# Why a leaf's affine model is not trusted, as a bitmask with one bit per
# failed test; `LEAF_CONVERGED` (zero) is the only status that is trusted.
# Test a flag with `(status & FLAG) != 0`. Only `max_level` leaves store a
# nonzero status: below it a failure means a split, and a closure triangle
# reads its origin's status.
#
# - `LEAF_CONVERGENCE_FAILED`: a mapped midpoint strays from its affine
#   prediction by more than `min_img_sep` allows, carried to the source plane
#   by the children's affine maps or by the Jacobian at the six samples.
# - `LEAF_APPROX_PARITY_UNRESOLVED`: the four red-split children disagree on
#   their orientation in the source plane.
# - `LEAF_JACOBIAN_PARITY_UNRESOLVED`: `det A` changes sign between the six
#   samples, or is exactly zero at one: a zero has no sign, and puts a
#   critical point on the triangle. (The band instead counts a zero as
#   positive, to trace the curve.)
# - `LEAF_RAYTRACE_NONFINITE`: some image is non-finite; then it is the only
#   flag.
# - `LEAF_JACOBIAN_NONFINITE`: some `det A` is non-finite.
LEAF_CONVERGED = 0
LEAF_CONVERGENCE_FAILED = 1 << 0
LEAF_APPROX_PARITY_UNRESOLVED = 1 << 1
LEAF_JACOBIAN_PARITY_UNRESOLVED = 1 << 2
LEAF_RAYTRACE_NONFINITE = 1 << 3
LEAF_JACOBIAN_NONFINITE = 1 << 4


def midpoint_deviation(beta_v, beta_m):
    """
    Distance between each mapped edge midpoint and its affine prediction.

    With ``m_i`` opposite ``theta_i``, the predicted image of ``m_i`` under an
    affine map is the mean of the two ``beta`` values at the endpoints of the edge
    it bisects, ``(beta_j + beta_k) / 2`` for cyclic ``(i, j, k)``.

    Parameters
    ----------
    beta_v: ArrayLike
        Source-plane vertices, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    beta_m: ArrayLike
        Source-plane edge midpoints ``m1, m2, m3``, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        Shape ``(n, 3)``, a source-plane length.

        *Unit: arcsec*
    """
    predicted = 0.5 * (beta_v[:, [1, 2, 0], :] + beta_v[:, [2, 0, 1], :])
    return mesh_backend.norm(predicted - beta_m, dim=-1)


def child_shape_matrices(beta_v, beta_m):
    """
    Source-plane edge matrices ``Q_k`` of the four red-split children.

    Parameters
    ----------
    beta_v: ArrayLike
        Source-plane vertices, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    beta_m: ArrayLike
        Source-plane midpoints ``m1, m2, m3``, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        Shape ``(n, 4, 2, 2)``, child index in :data:`CHILD_VERTEX_INDICES`
        order.
    """
    stacked = mesh_backend.concatenate([beta_v, beta_m], dim=1)  # (n, 6, 2)
    idx0, idx1, idx2 = zip(*CHILD_VERTEX_INDICES)  # each length 4
    q1 = stacked[:, list(idx0), :]
    q2 = stacked[:, list(idx1), :]
    q3 = stacked[:, list(idx2), :]
    return mesh_backend.stack((q2 - q1, q3 - q1), dim=-1)  # (n, 4, 2, 2)


def parity_from_children(Q):
    """
    True where ``sign(det Q_k)`` is constant over the four children.

    The children's affine maps share a sign of determinant exactly when
    their ``Q_k`` do, since every ``det M_k == +1``. A ``NaN`` fails; an exact
    zero fails unless every child's is zero.

    Parameters
    ----------
    Q: ArrayLike
        Child edge matrices from :func:`child_shape_matrices`, shape
        ``(n, 4, 2, 2)``.

    Returns
    -------
    ArrayLike
        Shape ``(n,)`` bool.
    """
    det_q = Q[..., 0, 0] * Q[..., 1, 1] - Q[..., 0, 1] * Q[..., 1, 0]
    sign_q = mesh_backend.sign(det_q)
    return mesh_backend.all(sign_q == sign_q[:, :1], dim=1)


def children_sigma_min(Q, cls, level, h0):
    """
    Smallest singular value over the four red-split children's affine maps, ``(n,)``.

    A child's affine map is its ``Q_k`` times the inverse of its lens-plane
    edge matrix: ``PINV0[COMPOSE[cls]]`` at unit size, scaled by
    ``2**(level + 1) / h0``.

    Parameters
    ----------
    Q: ArrayLike
        Child edge matrices from :func:`child_shape_matrices`, shape
        ``(n, 4, 2, 2)``.
    cls: ArrayLike
        ``(n,)`` int64 orientation classes.
    level: ArrayLike
        ``(n,)`` int64 refinement levels.
    h0: float
        Level-0 cell size.

        *Unit: arcsec*
    """
    return (
        mesh_backend.min(sigma_min_2x2(Q @ PINV0[COMPOSE[cls]]), dim=1)
        * mesh_backend.to(2 ** (level + 1), dtype=mesh_backend.float64)
        / h0
    )


def lens_status(beta6, det6, sigma6, cls, level, h0, min_img_sep):
    """
    Why each triangle's affine model of the lens map is not trusted, and the deviation and smallest singular value its error bound reads.

    The status is a ``LEAF_*`` bitmask, ``LEAF_CONVERGED`` -- zero -- only
    where every one of these holds:

    - every image is finite; otherwise ``LEAF_RAYTRACE_NONFINITE`` is the
      row's only flag;
    - the four red-split children's source-plane edge matrices share a
      sign of determinant (``LEAF_APPROX_PARITY_UNRESOLVED``);
    - the largest mapped-midpoint deviation from its affine prediction is
      below ``sigma_min * min_img_sep`` (``LEAF_CONVERGENCE_FAILED``), with
      ``sigma_min`` the smaller of the children's affine maps' smallest
      singular value and the smallest ``sigma_min(A)`` at the six samples:
      the lens-plane tolerance carried to the source plane. Written as
      ``deviation < threshold``, so that a NaN fails, and so does an
      exactly singular triangle, ``0 < 0``;
    - ``det A`` is finite at all six samples (``LEAF_JACOBIAN_NONFINITE``)
      and of one strict sign, an exact zero having none
      (``LEAF_JACOBIAN_PARITY_UNRESOLVED``).

    Parameters
    ----------
    beta6: ArrayLike
        ``(n, 6, 2)`` images at ``theta_1, theta_2, theta_3, m_1, m_2, m_3``,
        ``m_i`` opposite ``theta_i``.

        *Unit: arcsec*
    det6: ArrayLike
        ``(n, 6)`` ``det A`` at the same samples.
    sigma6: ArrayLike
        ``(n, 6)`` smallest singular value of ``A`` at the same samples.
    cls: ArrayLike
        ``(n,)`` int64 orientation classes.
    level: ArrayLike
        ``(n,)`` int64 refinement levels.
    h0: float
        Level-0 cell size.

        *Unit: arcsec*
    min_img_sep: float
        Lens-plane tolerance.

        *Unit: arcsec*

    Returns
    -------
    status: ArrayLike
        ``(n,)`` int64.
    deviation: ArrayLike
        ``(n,)``, the largest of :func:`midpoint_deviation`.

        *Unit: arcsec*
    sigma_min: ArrayLike
        ``(n,)``, the smaller of :func:`children_sigma_min` and the smallest
        of ``sigma6``.
    """
    beta_v, beta_m = beta6[:, :3], beta6[:, 3:]
    Q = child_shape_matrices(beta_v, beta_m)
    deviation = mesh_backend.max(midpoint_deviation(beta_v, beta_m), dim=1)
    sigma_min = mesh_backend.minimum(
        children_sigma_min(Q, cls, level, h0), mesh_backend.min(sigma6, dim=1)
    )
    det_ok = mesh_backend.all(mesh_backend.isfinite(det6), dim=1)
    one_sign = mesh_backend.all(det6 > 0, dim=1) | mesh_backend.all(det6 < 0, dim=1)
    status = (
        mesh_backend.long(~parity_from_children(Q)) * LEAF_APPROX_PARITY_UNRESOLVED
        | mesh_backend.long(~(deviation < sigma_min * min_img_sep))
        * LEAF_CONVERGENCE_FAILED
        | mesh_backend.long(~det_ok) * LEAF_JACOBIAN_NONFINITE
        | mesh_backend.long(det_ok & ~one_sign) * LEAF_JACOBIAN_PARITY_UNRESOLVED
    )
    finite = mesh_backend.all(mesh_backend.isfinite(beta6), dim=(1, 2))
    status = mesh_backend.where(finite, status, LEAF_RAYTRACE_NONFINITE)
    return status, deviation, sigma_min


def affine_error(deviation, sigma_min):
    """
    ``deviation / sigma_min``: how far a seed of an affine model can lie from the image it approximates.

    A source-plane error ``deviation`` pulls back to at most
    ``deviation / sigma_min`` in the lens plane. ``+inf`` where ``sigma_min``
    is zero, no bound. Below ``min_img_sep`` exactly where both deviation
    checks of :func:`lens_status` pass, up to rounding.

    Parameters
    ----------
    deviation: ArrayLike
        ``(n,)`` from :func:`lens_status`.

        *Unit: arcsec*
    sigma_min: ArrayLike
        ``(n,)`` from :func:`lens_status`.

    Returns
    -------
    ArrayLike
        ``(n,)``.

        *Unit: arcsec*
    """
    zero = sigma_min == 0
    return mesh_backend.where(
        zero, float("inf"), deviation / mesh_backend.where(zero, 1.0, sigma_min)
    )
