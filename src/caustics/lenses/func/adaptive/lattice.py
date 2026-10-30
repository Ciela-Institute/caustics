"""
The dyadic integer lattice every mesh vertex lives on, its sizes, and triangles on it.

Keys and lens-plane positions, growth by whole level-0 cells, the depth
floor and the int64 key limit, the level-0 triangles of a grid or of the ring
an extension adds, and exact edge midpoints.
"""

import math
from typing import NamedTuple
from warnings import warn

from ....backend_obj import ArrayLike
from .mesh_backend import mesh_backend
from .geometry import ROOT_SHAPES

MAX_KEY = 2**63 - 1


def depth_floor(h0, tol):
    """Level at which the longest leaf edge, ``sqrt(2) * h0 / 2**level``, first falls to ``tol``."""
    l_max0 = math.sqrt(2.0) * h0
    if l_max0 <= tol:
        return 0
    return int(math.ceil(math.log2(l_max0 / tol)))


def warn_depth_limited(what, request, h0, tol, max_level):
    """
    Warn when ``max_depth`` stopped refinement before the longest leaf edge fell to ``tol``.

    ``what`` names the mesh and ``request`` the tolerance as the caller gave it.
    """
    d_floor = depth_floor(h0, tol)
    if d_floor <= max_level:
        return
    l_max = math.sqrt(2.0) * h0 / 2**max_level
    warn(
        f"{what} is depth-limited: max_depth={max_level} is below d_floor={d_floor}, "
        f"the depth required to reach {request}. Refinement stops at level "
        f"{max_level}, where the longest leaf edge is {l_max:.3g} arcsec. Set "
        f"max_depth >= {d_floor}, or raise init_res or the tolerance."
    )


def max_key_level(init_res):
    """The deepest ``max_level`` whose lattice, one level finer, int64 keys hold; ``-1`` if not even level 0's."""
    level = -1
    while ((int(init_res) << (level + 2)) + 1) ** 2 < MAX_KEY:
        level += 1
    return level


def check_lattice_keys(init_res, max_level, remedy):
    """
    Raise if the lattice for ``init_res`` cells at ``max_level`` overflows int64 keys.

    The lattice is built one level finer than ``max_level`` (see
    :class:`Lattice`), so that is the one sized here. ``remedy`` ends the
    message: what the caller can change.
    """
    if max_level > max_key_level(init_res):
        n = init_res * (1 << (max_level + 1))
        raise ValueError(
            f"lattice too fine to key in int64: init_res={init_res} at level "
            f"{max_level} needs a lattice of {n + 1} points per axis, one level "
            f"finer than max_level so that max_level edge midpoints are lattice "
            f"points. {remedy}"
        )


class Lattice(NamedTuple):
    """
    Dyadic integer lattice over the square domain, one level finer than ``max_level``.

    Every mesh vertex is an integer pair and its position a function of the
    pair alone, so midpoints are exact integer averages and triangles sharing
    a vertex see bit-identical coordinates. At level ``d <= max_level`` a
    triangle's edges are multiples of ``1 << (max_level + 1 - d) >= 2``
    lattice steps, so :func:`midpoint_ij` is exact at every level. Vertices,
    and midpoints below ``max_level``, have even coordinates; a ``max_level``
    midpoint has an odd one, so it is never a vertex.

    Parameters
    ----------
    level: int
        Level the lattice is built at, ``max_level + 1``.
    n: int
        Lattice steps per axis, ``init_res << level``.
    scale: float
        Lattice step.

        *Unit: arcsec*
    lo: ArrayLike
        ``(2,)`` float64 position of lattice index ``(origin, origin)``.

        *Unit: arcsec*
    origin: int
        Lattice index of ``lo`` on both axes; :func:`extend_lattice` moves it
        with every index, so growth leaves every position bit-identical.
    """

    level: int
    n: int
    scale: float
    lo: ArrayLike
    origin: int = 0

    @property
    def h0(self):
        """Level-0 cell size, ``fov / init_res``.

        *Unit: arcsec*
        """
        return self.scale * (1 << self.level)

    @property
    def fov(self):
        """Side of the square the lattice covers.

        *Unit: arcsec*
        """
        return self.n * self.scale

    @property
    def init_res(self):
        """Level-0 cells per axis."""
        return self.n >> self.level

    @property
    def corner(self):
        """Position of lattice point ``(0, 0)``, ``(2,)``, in the backend and on the device of ``lo``.

        :func:`lattice_xy` of ``(0, 0)``, operation for operation.

        *Unit: arcsec*
        """
        return self.lo + float(-self.origin) * self.scale


def make_lattice(fov, x0, y0, init_res, lattice_level):
    """
    Build a :class:`Lattice` covering ``fov``, centered at ``(x0, y0)``.

    Parameters
    ----------
    fov: float
        Field of view.

        *Unit: arcsec*

    x0: float
        Domain center, x.

        *Unit: arcsec*

    y0: float
        Domain center, y.

        *Unit: arcsec*

    init_res: int
        Level-0 grid resolution.
    lattice_level: int
        Level at which the lattice itself is built -- one finer than
        ``max_level``, see :class:`Lattice`.

    Returns
    -------
    Lattice
        With origin 0: lo is the lattice's own corner.
    """
    level = int(lattice_level)
    n = int(init_res) * (1 << level)
    scale = float(fov) / n
    lo = mesh_backend.as_array(
        [x0 - fov / 2.0, y0 - fov / 2.0], dtype=mesh_backend.float64
    )
    return Lattice(level=level, n=n, scale=scale, lo=lo, origin=0)


def extend_lattice(lat, k):
    """
    ``lat`` grown by ``k`` level-0 cells on every side, every old point in place.

    A point at ``ij`` on ``lat`` is at ``ij + (k << lat.level)`` on the
    result and ``origin`` moves by as much, so :func:`lattice_xy` is
    unchanged bit for bit; ``lo`` and ``scale`` are never recomputed, which
    could round differently. Keys change, but keep their order.

    Parameters
    ----------
    lat: Lattice
    k: int
        Level-0 cells added on each side.

    Returns
    -------
    Lattice
    """
    pad = int(k) << lat.level
    n = lat.n + 2 * pad
    return lat._replace(n=n, origin=lat.origin + pad)


def lattice_key(lat, ij):
    """Lattice key of integer coordinates, shape ``(..., 2) -> (...)``."""
    return ij[..., 0] * (lat.n + 1) + ij[..., 1]


def lattice_ij_from_key(lat, key):
    """Inverse of :func:`lattice_key`, shape ``(...) -> (..., 2)``."""
    stride = lat.n + 1
    return mesh_backend.stack((key // stride, key % stride), dim=-1)


def lattice_xy(lat, ij):
    """Lens-plane position ``lo + (ij - origin) * scale``, shape ``(..., 2) -> (..., 2)``.

    *Unit: arcsec*
    """
    return (
        lat.lo
        + mesh_backend.to(ij - lat.origin, dtype=mesh_backend.float64) * lat.scale
    )


def level0_cell(ij, lattice_level):
    """
    The level-0 cell of each triangle: the lower corner of its lattice bounding box, in level-0 cells.

    Parameters
    ----------
    ij: ArrayLike
        ``(n, 3, 2)`` int64 lattice coordinates.
    lattice_level: int
        The level the lattice is built at.

    Returns
    -------
    ArrayLike
        ``(n, 2)`` int64.
    """
    return mesh_backend.min(ij, dim=1) // (1 << int(lattice_level))


def initial_triangles(init_res, lattice_level, root_class):
    """
    Level-0 triangles: two per cell, split on the ``(0,0)-(1,1)`` diagonal.

    Both are positively oriented; cells come in row-major ``(i, j)`` order.

    Parameters
    ----------
    init_res: int
        Level-0 grid resolution.
    lattice_level: int
        Level the lattice is built at -- one finer than ``max_level``, see
        :class:`Lattice`.
    root_class: ArrayLike
        ``ROOT_CLASS`` from :func:`child_matrix_tables`, shape ``(2,)``.

    Returns
    -------
    ij: ArrayLike
        ``(2 * init_res**2, 3, 2)`` int64 lattice coordinates.
    cls: ArrayLike
        ``(2 * init_res**2,)`` int64 orientation classes.
    """
    step = 1 << int(lattice_level)
    axis = mesh_backend.arange(init_res, dtype=mesh_backend.int64)
    i, j = mesh_backend.meshgrid(axis, axis, indexing="ij")
    base = (
        mesh_backend.stack((i.reshape(-1), j.reshape(-1)), dim=-1) * step
    )  # (r**2, 2)
    blocks, classes = [], []
    for s, shape in enumerate(ROOT_SHAPES):
        offs = mesh_backend.as_array(shape, dtype=mesh_backend.int64) * step  # (3, 2)
        blocks.append(base[:, None, :] + offs[None, :, :])
        classes.append(
            mesh_backend.zeros((base.shape[0],), dtype=mesh_backend.int64)
            + root_class[s]
        )
    return mesh_backend.concatenate(blocks, dim=0), mesh_backend.concatenate(
        classes, dim=0
    )


def ring_triangles(init_res, k, lattice_level, root_class):
    """
    Level-0 triangles of the cells within ``k`` cells of the domain's edge.

    :func:`initial_triangles` of the whole ``init_res x init_res`` grid, less
    the central ``(init_res - 2k) x (init_res - 2k)`` block of cells: the
    ring an extension by ``k`` adds around an existing mesh. A row subset of
    :func:`initial_triangles`, in its order.

    Parameters
    ----------
    init_res: int
        Level-0 cells per axis of the grown grid.
    k: int
        Width of the ring, in cells.
    lattice_level: int
    root_class: ArrayLike
        ``ROOT_CLASS`` from :func:`child_matrix_tables`.

    Returns
    -------
    ij: ArrayLike
        ``(T, 3, 2)`` int64 lattice coordinates.
    cls: ArrayLike
        ``(T,)`` int64 orientation classes.
    """
    ij, cls = initial_triangles(init_res, lattice_level, root_class)
    cell = level0_cell(ij, lattice_level)
    inner = (cell >= k) & (cell < init_res - k)
    keep = mesh_backend.flatnonzero(~(inner[:, 0] & inner[:, 1]))
    return ij[keep], cls[keep]


def midpoint_ij(ij):
    """
    Edge midpoints ``m1, m2, m3``, with ``m_i`` opposite ``theta_i``.

    Exact integer averaging via integer ``//``: the lattice is one level finer
    than ``max_level`` (see :class:`Lattice`), so the coordinate sums are even
    at every level up to and including ``max_level``.

    Parameters
    ----------
    ij: ArrayLike
        Triangle vertex coordinates, shape ``(n, 3, 2)`` int64.

    Returns
    -------
    ArrayLike
        Shape ``(n, 3, 2)`` int64.
    """
    return mesh_backend.stack(
        (
            (ij[:, 1] + ij[:, 2]) // 2,
            (ij[:, 2] + ij[:, 0]) // 2,
            (ij[:, 0] + ij[:, 1]) // 2,
        ),
        dim=1,
    )
