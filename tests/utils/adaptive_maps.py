"""Analytic lens maps and helpers shared by the adaptive-mesh tests."""

from types import SimpleNamespace

import numpy as np

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh


def to_np(x):
    """Any array, from either backend, as numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def f64(x):
    return mesh_backend.as_array(
        np.asarray(x, dtype=np.float64), dtype=mesh_backend.float64
    )


def i64(x):
    return mesh_backend.as_array(
        np.asarray(x, dtype=np.int64), dtype=mesh_backend.int64
    )


def stack_2x2(a, b, c, d):
    return backend.stack(
        (backend.stack((a, b), dim=-1), backend.stack((c, d), dim=-1)), dim=-2
    )


def lens(raytrace, jacobian):
    """A stand-in for a caustics lens."""
    return SimpleNamespace(raytrace=raytrace, jacobian_lens_equation=jacobian)


def requested(xy):
    """
    The rows of a lens call the mesh asked for, ``(N, 2)``.

    Under jax the sampler pads a batch by repeating its last row
    (``backend.padded_size``); those copies are dropped. The bookkeeping is
    the same torch code under both backends, so the torch run still sees
    every row.
    """
    if backend.backend != "jax":
        return xy
    n = len(xy)
    while n > 1 and (xy[n - 1] == xy[n - 2]).all():
        n -= 1
    return xy[:n]


def numpy_lens(fn, jac):
    """
    A lens over numpy maps, recording every point each method is called on.

    ``fn`` maps ``(N, 2) -> (N, 2)`` and ``jac`` gives its ``(N, 2, 2)``
    Jacobian. Returns the lens and ``calls``, whose ``raytrace`` and
    ``jacobian`` lists hold the numpy ``(N, 2)`` points of each call.
    """
    calls = SimpleNamespace(raytrace=[], jacobian=[])

    def raytrace(x, y):
        xy = np.stack((to_np(x), to_np(y)), axis=-1)
        calls.raytrace.append(requested(xy))
        out = backend.as_array(fn(xy), dtype=x.dtype, device=backend.device(x))
        return out[:, 0], out[:, 1]

    def jacobian(x, y):
        xy = np.stack((to_np(x), to_np(y)), axis=-1)
        calls.jacobian.append(requested(xy))
        return backend.as_array(jac(xy), dtype=x.dtype, device=backend.device(x))

    return lens(raytrace, jacobian), calls


def build(fn, jac, fov=4.0, init_res=4, min_img_sep=0.25, **kw):
    """``build_lens_mesh`` of numpy maps, and the recorded calls."""
    from caustics.lenses.func.adaptive.lens_mesh import build_lens_mesh

    lens_, calls = numpy_lens(fn, jac)
    mesh = build_lens_mesh(
        lens_.raytrace, lens_.jacobian_lens_equation, fov, init_res, min_img_sep, **kw
    )
    return to_mesh(mesh), calls


def assert_same(a, b, path="mesh"):
    """
    ``a`` and ``b`` equal: NamedTuples field by field, arrays bit for bit.

    Two lattices are equal when they place every point alike: an extended
    lattice keeps its ``lo`` and shifts ``origin`` where a fresh one starts
    at its own corner.
    """
    a, b = to_mesh(a), to_mesh(b)
    from caustics.lenses.func.adaptive.lattice import Lattice, lattice_xy

    if isinstance(a, Lattice):
        assert (a.level, a.n, a.scale) == (b.level, b.n, b.scale), path
        corners = i64([[0, 0], [a.n, a.n], [a.n // 3, 2 * a.n // 3]])
        np.testing.assert_array_equal(
            to_np(lattice_xy(a, corners)),
            to_np(lattice_xy(b, corners)),
            err_msg=path,
        )
        return
    if isinstance(a, tuple) and hasattr(a, "_fields"):
        assert type(a) is type(b), path
        for name in a._fields:
            assert_same(getattr(a, name), getattr(b, name), f"{path}.{name}")
    elif hasattr(a, "shape"):
        assert a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape), path
        np.testing.assert_array_equal(to_np(a), to_np(b), err_msg=path)
    else:
        assert a == b, path


def localised_fold(p):
    """Affine outside ``|y| < 0.5``; inside, ``beta_y = 0.6 y + y**2 - 0.25`` folds at ``y = -0.3``."""
    y = p[:, 1]
    bend = np.where(np.abs(y) < 0.5, y**2 - 0.25, 0.0)
    return np.stack([p[:, 0], 0.6 * y + bend], axis=-1)


def localised_fold_jacobian(p):
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = np.where(np.abs(p[:, 1]) < 0.5, 0.6 + 2.0 * p[:, 1], 0.6)
    return J


def row_fold(p):
    """``(x, y - y**2)``: ``det A = 1 - 2y`` is exactly zero on the lattice row ``y = 0.5``."""
    return np.stack([p[:, 0], p[:, 1] - p[:, 1] ** 2], axis=-1)


def row_fold_jacobian(p):
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 1.0 - 2.0 * p[:, 1]
    return J


def sie_like(p):
    """A cored isothermal sphere: tangential curve at radius ~1.18, radial at ~0.32."""
    r = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2 + 0.05)
    return p - 1.2 * p / r[:, None]


def sie_like_jacobian(p):
    x, y = p[:, 0], p[:, 1]
    r = np.sqrt(x * x + y * y + 0.05)
    k = 1.2 / r**3
    J = np.empty((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0 - 1.2 / r + k * x * x
    J[:, 0, 1] = k * x * y
    J[:, 1, 0] = k * x * y
    J[:, 1, 1] = 1.0 - 1.2 / r + k * y * y
    return J


AFFINE = np.array([[0.7, 0.1], [-0.2, 0.9]])


def affine(p):
    return p @ AFFINE.T


def affine_jacobian(p):
    return np.tile(AFFINE, (p.shape[0], 1, 1))


def collapse(p):
    """A ``kappa == 1`` sheet: the whole lens plane maps to one point."""
    return np.zeros_like(p)


def collapse_jacobian(p):
    return np.zeros((p.shape[0], 2, 2))


def broken_where(fn, jac, where, value=np.nan):
    """``fn`` and ``jac`` with ``value`` wherever ``where(p)`` holds."""

    def broken(p):
        out = fn(p)
        out[where(p)] = value
        return out

    def broken_jacobian(p):
        J = jac(p)
        J[where(p)] = value
        return J

    return broken, broken_jacobian


def sis_raytrace(p, b=1.0):
    """SIS deflection ``beta = theta (1 - b/|theta|)``, non-finite at ``theta = 0``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.linalg.norm(p, axis=-1, keepdims=True)
        return p * (1.0 - b / r)


def sis_jacobian(p, b=1.0):
    """``(1 - b/r) I + b theta theta^T / r**3``, non-finite at the origin too."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.linalg.norm(p, axis=-1)[:, None, None]
        return (1.0 - b / r) * np.eye(2) + b * p[:, :, None] * p[:, None, :] / r**3


def point_raytrace(p, b=1.0):
    """Point mass ``beta = theta - b**2 theta / |theta|**2``, non-finite at ``theta = 0``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return p - b * b * p / np.sum(p * p, axis=-1, keepdims=True)


def point_jacobian(p, b=1.0):
    """``I - b**2 (|theta|**2 I - 2 theta theta^T) / |theta|**4``, non-finite at the origin too."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.sum(p * p, axis=-1)[:, None, None]
        return (
            np.eye(2)
            - b * b * (r2 * np.eye(2) - 2.0 * p[:, :, None] * p[:, None, :]) / r2**2
        )


def brute_hits(vertices, triangles, rows, beta, grow=None):
    """
    ``index_hits``'s answer, found by testing every one of ``rows`` against every point.

    Returns numpy ``qidx``, ``tri`` and ``w``, ordered by point and then by
    triangle. It uses the geometry functions ``index_hits`` uses on the same
    operands, so the weights agree bit for bit. ``grow`` is ``(T,)``, aligned
    with ``triangles``.
    """
    from caustics.lenses.func.adaptive.geometry import (
        contains,
        edge_nearest,
        triangle_weights,
    )

    rows_np = to_np(rows)
    tri = vertices[triangles[rows]]
    reach = None if grow is None else to_np(grow)[rows_np]
    qidx, hits, weights = [], [], []
    for b in range(beta.shape[0]):
        point = mesh_backend.repeat(beta[b : b + 1], rows_np.size, axis=0)
        w = triangle_weights(tri, point)
        hit = to_np(contains(w))
        if reach is not None:
            dist, _ = edge_nearest(tri, point)
            hit = hit | ((reach > 0) & (to_np(dist) <= reach))
        sel = np.flatnonzero(hit)
        qidx.append(np.full(sel.size, b, dtype=np.int64))
        hits.append(rows_np[sel])
        weights.append(to_np(w)[sel])
    return (
        np.concatenate(qidx),
        np.concatenate(hits).astype(np.int64),
        np.concatenate(weights).reshape(-1, 3),
    )


def assert_hits_equal(got, want):
    """``index_hits`` output ``got`` equals numpy ``want`` from :func:`brute_hits`, bit for bit."""
    for name, a, b in zip(("qidx", "tri", "w"), got, want):
        np.testing.assert_array_equal(to_np(a), b, err_msg=name)


def finite_rows(mesh):
    """The leaves every index of ``mesh`` holds: those whose origin has a finite raytrace and Jacobian."""
    from caustics.lenses.func.adaptive.criterion import (
        LEAF_JACOBIAN_NONFINITE,
        LEAF_RAYTRACE_NONFINITE,
    )

    status = to_np(mesh.origin_status)[to_np(mesh.leaf_origin)]
    return np.flatnonzero(
        (status & (LEAF_RAYTRACE_NONFINITE | LEAF_JACOBIAN_NONFINITE)) == 0
    )


def index_point_cells(index, level, points):
    """
    The cell of each of ``points`` ``(N, 2)`` at the matching entry of ``level`` ``(N,)``, numpy ``(N, 2)``.

    From the documented rule rather than ``index.py``'s code: the floor of
    the point's offset from ``lo`` over the level's cell side
    ``fine * 2**level``, clamped to the level's ``2**(KEY_BITS - level)``
    cells per axis.
    """
    from caustics.lenses.func.adaptive.index import KEY_BITS

    level = np.asarray(level)
    lo, fine = to_np(index.lo), float(to_np(index.fine))
    side = (fine * 2.0**level)[:, None]
    top = (2 ** (KEY_BITS - level) - 1)[:, None]
    return np.clip(np.floor((points - lo) / side), 0, top).astype(np.int64)


def index_cell_ranges(index, tri, grow=None):
    """
    Each triangle's level in ``index``, and the cells of its lower and upper box corners there, numpy.

    From the documented rule rather than ``index.py``'s code: the level is
    the number of powers ``2**0 .. 2**(KEY_BITS - 1)`` below the longer side
    of the triangle's box, grown by ``grow`` ``(n,)``, over
    ``CELLS_PER_BOX * fine``. Returns ``level`` ``(n,)``, ``i0`` and ``i1``
    ``(n, 2)``.
    """
    from caustics.lenses.func.adaptive.index import CELLS_PER_BOX, KEY_BITS

    fine = float(to_np(index.fine))
    box_lo, box_hi = tri.min(axis=1), tri.max(axis=1)
    if grow is not None:
        box_lo, box_hi = box_lo - grow[:, None], box_hi + grow[:, None]
    size = (box_hi - box_lo).max(axis=1) / (CELLS_PER_BOX * fine)
    level = np.searchsorted(2.0 ** np.arange(KEY_BITS), size)
    return (
        level,
        index_point_cells(index, level, box_lo),
        index_point_cells(index, level, box_hi),
    )


def index_cell(index, level, ix, iy):
    """The rows ``index`` lists in cell ``(ix, iy)`` of ``level``, numpy; none where it stores no such cell."""
    from caustics.lenses.func.adaptive.index import KEY_BITS

    key = (int(level) << (2 * KEY_BITS)) | (int(ix) << KEY_BITS) | int(iy)
    keys, offsets = to_np(index.keys), to_np(index.offsets)
    u = int(np.searchsorted(keys, key))
    if u == keys.size or keys[u] != key:
        return np.zeros(0, dtype=np.int64)
    return to_np(index.leaves)[offsets[u] : offsets[u + 1]]
