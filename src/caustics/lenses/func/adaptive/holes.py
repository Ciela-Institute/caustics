"""
Holes cut around lens centers, and the images of their boundary circles.

The lens map can jump at a lens center, so no curve of ``det A = 0``
describes what happens to images there. A hole of radius ``min_img_sep`` is
left out of the lens plane instead, and the image of its boundary circle, the
hole curve, is traced (:func:`sample_holes`).
:mod:`~caustics.lenses.func.adaptive.curves` re-joins critical curves along
it.
"""

import math
from typing import NamedTuple
from warnings import warn

from ....backend_obj import ArrayLike
from .mesh_backend import mesh_backend
from .geometry import (
    build_device,
    component_labels,
    csr_offsets,
    expand,
    ring_next,
    to_device,
)
from .refine import sample_points


class CenterHoles(NamedTuple):
    """
    Small disks cut around lens centers, and the images of their boundaries.

    The lens map can jump at a lens center -- an isothermal profile's
    deflection depends only on the direction from its center -- so no curve
    of ``det A = 0`` describes what happens to images there. A hole of radius
    ``radius`` around each center is left out of the lens plane instead, and
    the image of its boundary circle, the *hole curve*, is kept. At an
    isothermal center the hole curve is the pseudo-caustic to within about
    ``radius``; at a point mass, or a cusp steeper than isothermal, it is a
    huge loop; at a regular point, or a profile whose deflection vanishes at
    its center, it is a speck. ``growth`` tells these apart: the hole curve's
    size scales as ``radius**growth`` there, so ``growth`` is 0 for a
    pseudo-caustic, negative for a loop that runs off to infinity as the hole
    shrinks, and positive for a speck -- exactly ``1 - t`` for a power law of
    slope ``t``.

    ``growth`` is measured, so never exactly 0: ``growth_err`` bounds its
    error, and ``pseudo_caustic`` marks the holes whose ``growth`` is 0
    within it.

    Samples are stored CSR per hole: hole ``h`` is rows
    ``offsets[h]:offsets[h + 1]`` of ``angle``, ``lens`` and ``source``, in
    strictly ascending ``angle``.

    Parameters
    ----------
    centers: ArrayLike
        ``(H, 2)`` float64 hole centers after merging.

        *Unit: arcsec*
    radius: ArrayLike
        ``(H,)`` float64 hole radii: the mesh's ``min_img_sep``, plus the
        spread of the centers a hole merged.

        *Unit: arcsec*
    offsets: ArrayLike
        ``(H + 1,)`` int64 CSR offsets, ``offsets[0] == 0``.
    angle: ArrayLike
        ``(P,)`` float64 sample angles in ``[0, 2 pi)``.

        *Unit: radians*
    lens: ArrayLike
        ``(P, 2)`` float64 circle points ``center + radius * (cos, sin)(angle)``.

        *Unit: arcsec*
    source: ArrayLike
        ``(P, 2)`` float64 images of ``lens``: the hole curve.

        *Unit: arcsec*
    growth: ArrayLike
        ``(H,)`` float64 log-slope of the hole curve's size against the hole
        radius as the hole shrinks to nothing, extrapolated from circles
        down to ``radius / 64`` (:func:`sample_holes`). NaN when the circles
        map to a single point.
    growth_err: ArrayLike
        ``(H,)`` float64 bound on the error of ``growth``; NaN where
        ``growth`` is.
    pseudo_caustic: ArrayLike
        ``(H,)`` bool, True where ``|growth| <= growth_err``: where the hole
        curve is the pseudo-caustic, to within about ``radius``. False where
        ``growth`` is NaN.
    """

    centers: ArrayLike
    radius: ArrayLike
    offsets: ArrayLike
    angle: ArrayLike
    lens: ArrayLike
    source: ArrayLike
    growth: ArrayLike
    growth_err: ArrayLike
    pseudo_caustic: ArrayLike


def empty_holes(device=None):
    """A :class:`CenterHoles` with no hole."""
    return CenterHoles(
        centers=mesh_backend.zeros((0, 2), dtype=mesh_backend.float64, device=device),
        radius=mesh_backend.zeros((0,), dtype=mesh_backend.float64, device=device),
        offsets=mesh_backend.zeros((1,), dtype=mesh_backend.int64, device=device),
        angle=mesh_backend.zeros((0,), dtype=mesh_backend.float64, device=device),
        lens=mesh_backend.zeros((0, 2), dtype=mesh_backend.float64, device=device),
        source=mesh_backend.zeros((0, 2), dtype=mesh_backend.float64, device=device),
        growth=mesh_backend.zeros((0,), dtype=mesh_backend.float64, device=device),
        growth_err=mesh_backend.zeros((0,), dtype=mesh_backend.float64, device=device),
        pseudo_caustic=mesh_backend.zeros((0,), dtype=mesh_backend.bool, device=device),
    )


def merge_centers(centers, min_img_sep):
    """
    Merge lens centers into holes whose disks do not overlap.

    Centers closer than ``2 * min_img_sep`` share a hole. A hole sits at its
    members' mean, with radius ``min_img_sep`` plus the members' largest
    distance from that mean, so a lone center gets exactly ``min_img_sep``.
    A merged hole is larger than ``min_img_sep``, so it can reach another
    hole's disk; merging repeats until no two disks overlap. The centers are
    put in lexicographic order first and the holes are returned in it, so
    the result -- sums included -- depends on the set of centers, not on the
    order they were given in.

    Parameters
    ----------
    centers: Optional[ArrayLike]
        ``(S, 2)`` lens-plane positions, any array-like. ``None`` or an empty
        array gives no hole.

        *Unit: arcsec*
    min_img_sep: float
        The mesh's own tolerance, already halved.

        *Unit: arcsec*

    Returns
    -------
    centers: ArrayLike
        ``(H, 2)`` float64 hole centers, in lexicographic ``(x, y)`` order.

        *Unit: arcsec*
    radius: ArrayLike
        ``(H,)`` float64 hole radii.

        *Unit: arcsec*
    """
    f64 = mesh_backend.float64
    if centers is None:
        return mesh_backend.zeros((0, 2), dtype=f64), mesh_backend.zeros(
            (0,), dtype=f64
        )
    c = to_device(mesh_backend.as_array(centers, dtype=f64), build_device())
    if c.reshape(-1).shape[0] == 0:
        return mesh_backend.zeros((0, 2), dtype=f64), mesh_backend.zeros(
            (0,), dtype=f64
        )
    c = c[mesh_backend.lexsort([c[:, 1], c[:, 0]])]
    # Every center links itself, at distance 0: the self-loops component_labels reads.
    link = (
        mesh_backend.norm(
            mesh_backend.unsqueeze(c, 1) - mesh_backend.unsqueeze(c, 0), dim=-1
        )
        < 2.0 * min_img_sep
    )
    while True:
        _, member = mesh_backend.unique(component_labels(link), return_inverse=True)
        k = int(mesh_backend.to_numpy(mesh_backend.max(member))) + 1
        onehot = mesh_backend.unsqueeze(
            mesh_backend.arange(k, dtype=mesh_backend.int64), 1
        ) == mesh_backend.unsqueeze(member, 0)
        weight = mesh_backend.to(onehot, dtype=f64)
        mean = (weight @ c) / mesh_backend.unsqueeze(mesh_backend.sum(weight, dim=1), 1)
        spread = mesh_backend.norm(
            mesh_backend.unsqueeze(c, 0) - mesh_backend.unsqueeze(mean, 1), dim=-1
        )
        radius = min_img_sep + mesh_backend.max(
            mesh_backend.where(onehot, spread, 0.0), dim=1
        )
        gap = mesh_backend.norm(
            mesh_backend.unsqueeze(mean, 1) - mesh_backend.unsqueeze(mean, 0), dim=-1
        )
        overlap = (
            gap < mesh_backend.unsqueeze(radius, 1) + mesh_backend.unsqueeze(radius, 0)
        ) & ~mesh_backend.eye(k, dtype=mesh_backend.bool)
        if not bool(mesh_backend.any(overlap)):
            break
        link = link | overlap[member][:, member]
    order = mesh_backend.lexsort([mean[:, 1], mean[:, 0]])
    return mean[order], radius[order]


# Hole sampling: the first even pass, the refinement cap, and the even
# samples on each of the circles the growth exponent is measured on.
HOLE_INITIAL_SAMPLES = 256
HOLE_MAX_SAMPLES = 1 << 16
HOLE_GROWTH_SAMPLES = 1024


def hole_circle(centers, radius, hole, angle):
    """
    Points ``centers[hole] + radius[hole] * (cos, sin)(angle)``, shape ``(K, 2)``.

    *Unit: arcsec*
    """
    c, r = centers[hole], radius[hole]
    return mesh_backend.stack(
        (c[:, 0] + r * mesh_backend.cos(angle), c[:, 1] + r * mesh_backend.sin(angle)),
        dim=-1,
    )


def _trace_hole_points(trace, centers, radius, hole, angle, batch_size, scale=1.0):
    """
    Images of points on the hole circles scaled by ``scale``, shape ``(K, 2)``.

    Raises
    ------
    ValueError
        If any image is non-finite, naming the first such hole's center and
        the radius of the circle traced.
    """
    source = sample_points(
        hole_circle(centers, radius * scale, hole, angle), trace, batch_size
    )
    bad = mesh_backend.flatnonzero(
        ~mesh_backend.all(mesh_backend.isfinite(source), dim=1)
    )
    if bad.shape[0]:
        h = int(mesh_backend.to_numpy(hole[bad[0]]))
        x, y = mesh_backend.to_numpy(centers[h]).tolist()
        r = float(mesh_backend.to_numpy(radius[h])) * scale
        raise ValueError(
            f"the lens is not finite on the hole circle of radius {r:g} around "
            f"({x:g}, {y:g}); a hole's boundary must lie where the lens is finite"
        )
    return source


def _extent(points):
    """Diagonal of each row's bounding box, ``(H, K, 2) -> (H,)``."""
    return mesh_backend.norm(
        mesh_backend.max(points, dim=1) - mesh_backend.min(points, dim=1), dim=-1
    )


def _even_angles(n_holes, n):
    """``n`` even angles in ``[0, 2 pi)`` on each of ``n_holes`` holes: every sample's hole and angle, hole by hole."""
    hole, k = expand(
        mesh_backend.zeros((n_holes,), dtype=mesh_backend.int64) + n, n_holes * n
    )
    return hole, mesh_backend.to(k, dtype=mesh_backend.float64) * (2.0 * math.pi / n)


def sample_holes(trace, centers, radius, min_img_sep, batch_size):
    """
    Trace every hole's boundary circle and its image, the hole curve.

    Each circle starts with :data:`HOLE_INITIAL_SAMPLES` evenly spaced angles.
    Every angular interval whose source-plane chord exceeds ``min_img_sep``
    -- the wrap-around one included -- is then bisected, round after round,
    one ``trace`` call per round for all holes together, until no chord
    exceeds ``min_img_sep`` or a hole would pass :data:`HOLE_MAX_SAMPLES`
    samples. A hole stopped by the cap stays exact at its samples but coarser
    between them, and is warned about; only enormous hole curves, such as a
    point mass's, reach it.

    ``growth`` is read off the hole curve's size -- its bounding box's
    diagonal -- on :data:`HOLE_GROWTH_SAMPLES` even samples at ``radius``,
    ``radius / 4``, ``/ 16`` and ``/ 64``. The slope between neighbouring
    circles, ``g(r) = log(size(r) / size(r / 4)) / log(4)``, carries a bias
    linear in ``r`` from the smooth part of the lens map; Richardson
    extrapolation, ``R(r) = (4 g(r / 4) - g(r)) / 3``, cancels it, and
    ``growth = R(r / 4)``. What is left falls as ``r**p``, so
    ``growth_err = |R(r) - R(r / 4)|`` bounds it for any ``p >= 1/2``.

    Parameters
    ----------
    trace: Callable[[ArrayLike], ArrayLike]
        ``(N, 2)`` lens-plane positions to ``(N, 2)`` images, as
        :func:`~caustics.lenses.func.adaptive.lens_mesh.make_sampler` gives
        without a Jacobian.
    centers: ArrayLike
        ``(H, 2)`` float64 hole centers, from :func:`merge_centers`.

        *Unit: arcsec*
    radius: ArrayLike
        ``(H,)`` float64 hole radii, from :func:`merge_centers`.

        *Unit: arcsec*
    min_img_sep: float
        Largest source-plane chord left between neighbouring samples.

        *Unit: arcsec*
    batch_size: Optional[int]
        Most points per ``trace`` call.

    Returns
    -------
    CenterHoles
        On the backend's default device.

    Raises
    ------
    ValueError
        If the lens is not finite somewhere on a circle at ``radius``,
        ``radius / 4``, ``/ 16`` or ``/ 64``.

    Warns
    -----
    UserWarning
        For each hole whose refinement stopped at :data:`HOLE_MAX_SAMPLES`.
    """
    n_holes = centers.shape[0]
    if n_holes == 0:
        return empty_holes()
    int64 = mesh_backend.int64
    two_pi = 2.0 * math.pi
    hole, angle = _even_angles(n_holes, HOLE_INITIAL_SAMPLES)
    source = _trace_hole_points(trace, centers, radius, hole, angle, batch_size)
    capped = mesh_backend.zeros((n_holes,), dtype=mesh_backend.bool)
    while True:
        counts = mesh_backend.bincount(hole, minlength=n_holes)
        i = mesh_backend.arange(hole.shape[0], dtype=int64)
        _, nxt = ring_next(csr_offsets(counts), hole.shape[0])
        wrap = nxt <= i  # a hole's last sample wraps round to its first
        chord = mesh_backend.norm(source[nxt] - source, dim=-1)
        bad = mesh_backend.flatnonzero(chord > min_img_sep)
        wanted = mesh_backend.bincount(hole[bad], minlength=n_holes)
        capped = capped | ((wanted > 0) & (counts + wanted > HOLE_MAX_SAMPLES))
        rows = bad[mesh_backend.flatnonzero(~capped[hole[bad]])]
        if rows.shape[0] == 0:
            break
        upper = mesh_backend.where(
            wrap[rows], angle[nxt[rows]] + two_pi, angle[nxt[rows]]
        )
        mid = 0.5 * (angle[rows] + upper)
        mid = mesh_backend.where(mid >= two_pi, mid - two_pi, mid)
        new_hole = hole[rows]
        new_source = _trace_hole_points(
            trace, centers, radius, new_hole, mid, batch_size
        )
        hole = mesh_backend.concatenate((hole, new_hole), dim=0)
        angle = mesh_backend.concatenate((angle, mid), dim=0)
        source = mesh_backend.concatenate((source, new_source), dim=0)
        order = mesh_backend.lexsort([angle, hole])
        hole, angle, source = hole[order], angle[order], source[order]
    for h in mesh_backend.to_numpy(mesh_backend.flatnonzero(capped)).tolist():
        x, y = mesh_backend.to_numpy(centers[h]).tolist()
        n = int(
            mesh_backend.to_numpy(mesh_backend.bincount(hole, minlength=n_holes)[h])
        )
        warn(
            f"the hole curve around ({x:g}, {y:g}) stopped at {n} samples "
            f"(cap {HOLE_MAX_SAMPLES}) before every chord fell below "
            f"min_img_sep={min_img_sep:g}; it is exact at its samples but "
            "coarser than min_img_sep between them"
        )

    g_hole, g_angle = _even_angles(n_holes, HOLE_GROWTH_SAMPLES)
    shape = (n_holes, HOLE_GROWTH_SAMPLES, 2)
    size = [
        _extent(
            _trace_hole_points(
                trace, centers, radius, g_hole, g_angle, batch_size, 0.25**k
            ).reshape(shape)
        )
        for k in range(4)
    ]
    # `step[k] / log(4)` is the slope between circles k and k + 1. Scale, never
    # divide: jax turns a division by a constant into a multiply by its
    # reciprocal for some shapes only, so a quotient would depend on H.
    step = [mesh_backend.log(size[k] / size[k + 1]) for k in range(3)]
    scale = 1.0 / (3.0 * math.log(4.0))
    growth = (4.0 * step[2] - step[1]) * scale
    growth_err = mesh_backend.abs(5.0 * step[1] - step[0] - 4.0 * step[2]) * scale

    counts = mesh_backend.bincount(hole, minlength=n_holes)
    offsets = csr_offsets(counts)
    return CenterHoles(
        centers=centers,
        radius=radius,
        offsets=offsets,
        angle=angle,
        lens=hole_circle(centers, radius, hole, angle),
        source=source,
        growth=growth,
        growth_err=growth_err,
        pseudo_caustic=mesh_backend.abs(growth) <= growth_err,
    )
