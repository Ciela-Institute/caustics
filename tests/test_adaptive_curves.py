"""Critical curves and caustics traced through the lens mesh's band, and the center holes they join at."""

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, SinglePlane
from caustics.lenses.func.adaptive import (
    CenterHoles,
    CriticalBand,
    CriticalCurvesAndCaustics,
    LensMesh,
    build_closed_lens_mesh,
    build_lens_mesh,
    caustic_area,
    critical_curves_and_caustics,
)
from caustics.lenses.func.adaptive.curves import (
    chain_order,
    chain_segments,
    child_segments,
    edge_zeros,
    join_at_holes,
    pseudo_caustics,
    trace_band,
    triangle_segments,
)
from caustics.lenses.func.adaptive.geometry import CHILD_VERTEX_INDICES
from caustics.lenses.func.adaptive.holes import (
    HOLE_GROWTH_SAMPLES,
    HOLE_MAX_SAMPLES,
    empty_holes,
    merge_centers,
    sample_holes,
)
from caustics.lenses.func.adaptive.lens_mesh import make_sampler
from adaptive_maps import (
    affine,
    affine_jacobian,
    numpy_lens,
    requested,
    sie_like,
    sie_like_jacobian,
    sis_jacobian as sis_map_jacobian,
    sis_raytrace as sis_map,
    to_np,
)


def _arr(x):
    return mesh_backend.as_array(
        np.asarray(x, dtype=np.float64), dtype=mesh_backend.float64
    )


def _band(samples, lens, det, source=None):
    """A hand-built `CriticalBand`; ``source`` defaults to ``lens``."""
    samples = np.asarray(samples, dtype=np.int64)
    return CriticalBand(
        leaves=mesh_backend.as_array(
            np.arange(samples.shape[0]), dtype=mesh_backend.int64
        ),
        samples=mesh_backend.as_array(samples, dtype=mesh_backend.int64),
        lens=_arr(lens),
        source=_arr(lens if source is None else source),
        det=_arr(det),
        sigma_min=_arr(np.ones(np.shape(det))),
    )


# The positively oriented unit leaf theta_1 theta_2 theta_3 = (0,0), (1,0),
# (0,1), followed by its midpoints m_1 m_2 m_3, m_i opposite theta_i.
UNIT_LENS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]]
)


def _unit_band(det):
    return _band([[0, 1, 2, 3, 4, 5]], UNIT_LENS, det)


# ---------------------------------------------------------------------------
# chain_order: list ranking by pointer jumping
# ---------------------------------------------------------------------------


def _random_chains(rng, sizes, cyclic):
    """A successor array over shuffled node ids, one chain per size.

    A chain of one node stays a path even when drawn cyclic: a one-node
    cycle would be a self-loop, which two distinct crossing edges never form.
    """
    k = int(sum(sizes))
    ids = rng.permutation(k)
    succ = -np.ones(k, dtype=np.int64)
    at = 0
    for n, cyc in zip(sizes, cyclic):
        chain = ids[at : at + n]
        succ[chain[:-1]] = chain[1:]
        if cyc and n > 1:
            succ[chain[-1]] = chain[0]
        at += n
    return succ


def _walk(succ):
    """Reference order: each path from its start, each cycle from its smallest node."""
    k = succ.size
    has_pred = np.zeros(k, dtype=bool)
    has_pred[succ[succ >= 0]] = True
    seen = np.zeros(k, dtype=bool)
    curves = []
    for s in range(k):
        if has_pred[s]:
            continue
        chain = [s]
        while succ[chain[-1]] >= 0:
            chain.append(int(succ[chain[-1]]))
        seen[chain] = True
        curves.append((s, chain, False))
    for s in range(k):
        if seen[s]:
            continue
        chain = [s]
        seen[s] = True
        while succ[chain[-1]] != s:
            chain.append(int(succ[chain[-1]]))
            seen[chain[-1]] = True
        curves.append((s, chain, True))
    return sorted(curves, key=lambda curve: curve[0])


@pytest.mark.parametrize("seed", range(6))
def test_chain_order_matches_a_python_walk(seed):
    """Pointer jumping gives the plain walk's curves, order and ``closed``."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 40, size=12)
    cyclic = rng.random(12) < 0.5
    succ = _random_chains(rng, sizes, cyclic)
    order, offsets, closed = (
        to_np(x)
        for x in chain_order(mesh_backend.as_array(succ, dtype=mesh_backend.int64))
    )
    want = _walk(succ)
    assert offsets.tolist() == np.cumsum([0] + [len(c) for _, c, _ in want]).tolist()
    assert closed.tolist() == [cl for _, _, cl in want]
    assert order.tolist() == [n for _, c, _ in want for n in c]


@pytest.mark.parametrize(
    "succ, order, offsets, closed",
    [
        ([], [], [0], []),
        ([-1], [0], [0, 1], [False]),
        ([1, 0], [0, 1], [0, 2], [True]),
        ([2, -1, 1], [0, 2, 1], [0, 3], [False]),
    ],
    ids=["empty", "lone node", "two-cycle", "path"],
)
def test_chain_order_small_cases(succ, order, offsets, closed):
    got = chain_order(mesh_backend.as_array(np.asarray(succ, dtype=np.int64)))
    assert to_np(got[0]).tolist() == order
    assert to_np(got[1]).tolist() == offsets
    assert to_np(got[2]).tolist() == closed


# ---------------------------------------------------------------------------
# Tracer kernels on hand-built bands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("signs", list(itertools.product((1.0, -1.0), repeat=6)))
def test_child_segments_keep_positive_det_on_the_left(signs):
    """Every segment, for all 64 sign patterns of a leaf, has ``det > 0`` on its left.

    Between them the 64 patterns give each of the four children all six of
    its mixed sign patterns. For each segment, the child it crosses is the
    three distinct samples of its two edges; every positive corner of that
    child must lie strictly left of the segment and every negative one
    strictly right. Magnitudes are varied so the crossings are not all
    midpoints of their edges.
    """
    det = np.asarray(signs) * np.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.25])
    band = _unit_band(det)
    start, end = child_segments(band.samples, band.det)
    start, end = to_np(start), to_np(end)
    # Every mixed child -- not all three corners of one class -- returns
    # exactly one segment, and no other child returns any.
    children = np.asarray(CHILD_VERTEX_INDICES)
    classes = det[children] >= 0
    mixed = classes.any(axis=1) & ~classes.all(axis=1)
    assert start.shape[0] == int(mixed.sum())
    p, _ = edge_zeros(mesh_backend.as_array(start), band.det, (band.lens, band.source))
    q, _ = edge_zeros(mesh_backend.as_array(end), band.det, (band.lens, band.source))
    p, q = to_np(p), to_np(q)
    for a, b, s, e in zip(p, q, start, end):
        corners = set(s.tolist()) | set(e.tolist())
        assert len(corners) == 3
        for c in corners:
            d = b - a
            r = UNIT_LENS[c] - a
            side = d[0] * r[1] - d[1] * r[0]
            assert side > 0 if det[c] > 0 else side < 0


def test_child_segments_skip_a_leaf_of_one_class():
    for det in (np.ones(6), -np.ones(6)):
        band = _unit_band(det)
        start, end = child_segments(band.samples, band.det)
        assert start.shape[0] == 0 and end.shape[0] == 0


def test_an_exact_zero_counts_as_positive_and_puts_the_crossing_on_it():
    """``det == 0`` at ``theta_1`` with every other sample negative.

    Zero counts positive, so ``theta_1`` is the lone positive corner of its
    child and every crossing of that child sits on ``theta_1`` itself.
    """
    det = np.array([0.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    band = _unit_band(det)
    curves = trace_band(band)
    lens = to_np(curves.lens)
    assert lens.shape[0] > 0
    assert np.array_equal(lens, np.zeros_like(lens))


def test_two_leaves_sharing_an_edge_chain_through_one_crossing():
    """``det = x - 1.5`` across the square ``[0, 2]**2`` split on its diagonal.

    The line crosses the lower-right leaf from ``y = 0`` to the diagonal and
    the upper-left one from the diagonal to ``y = 2``; the two share the
    crossing at ``(1.5, 1.5)``, which must appear once, with the chain running
    through it. ``det`` is affine, so the linear interpolation is exact, and
    ``det > 0`` on the right of the line puts it on the left of downward
    travel.
    """
    lens = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [0.0, 1.0],
        ]
    )
    samples = [[0, 1, 2, 4, 5, 6], [0, 2, 3, 7, 8, 5]]
    curves = trace_band(_band(samples, lens, lens[:, 0] - 1.5))
    assert to_np(curves.offsets).tolist() == [0, 5]
    assert to_np(curves.closed).tolist() == [False]
    assert to_np(curves.lens).tolist() == [
        [1.5, 2.0],
        [1.5, 1.5],
        [1.5, 1.0],
        [1.5, 0.5],
        [1.5, 0.0],
    ]
    assert np.array_equal(to_np(curves.source), to_np(curves.lens))


@pytest.mark.parametrize(
    "band",
    [_band(np.zeros((0, 6)), np.zeros((0, 2)), np.zeros(0)), _unit_band(np.ones(6))],
    ids=["empty band", "one class"],
)
def test_a_band_without_crossings_has_no_curves(band):
    curves = trace_band(band)
    assert to_np(curves.offsets).tolist() == [0]
    assert tuple(curves.lens.shape) == (0, 2)
    assert tuple(curves.source.shape) == (0, 2)
    assert tuple(curves.closed.shape) == (0,)
    assert tuple(curves.hole.shape) == (0,)


# ---------------------------------------------------------------------------
# join_at_holes: cutting at hole circles and re-joining clockwise
# ---------------------------------------------------------------------------


def _one_hole(radius=0.1, n=64, shift=(5.0, 0.0)):
    """A hand-built hole at the origin: ``n`` even samples; source = lens + shift."""
    angle = 2.0 * np.pi * np.arange(n) / n
    lens = radius * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    return CenterHoles(
        centers=_arr([[0.0, 0.0]]),
        radius=_arr([radius]),
        offsets=mesh_backend.as_array(np.array([0, n]), dtype=mesh_backend.int64),
        angle=_arr(angle),
        lens=_arr(lens),
        source=_arr(lens + np.array(shift)),
        growth=_arr([0.0]),
        growth_err=_arr([0.0]),
        pseudo_caustic=mesh_backend.as_array(np.array([True]), dtype=mesh_backend.bool),
    )


def _traced(*curves):
    """Hand-built ``CriticalCurvesAndCaustics`` from ``(points, closed)`` pairs.

    ``source`` is ``2 * lens``.
    """
    pts = np.concatenate([np.asarray(p, dtype=np.float64) for p, _ in curves])
    offsets = np.cumsum([0] + [len(p) for p, _ in curves])
    return CriticalCurvesAndCaustics(
        lens=_arr(pts),
        source=_arr(2.0 * pts),
        offsets=mesh_backend.as_array(offsets, dtype=mesh_backend.int64),
        closed=mesh_backend.as_array(
            np.array([c for _, c in curves]), dtype=mesh_backend.bool
        ),
        hole=mesh_backend.as_array(np.full(len(pts), -1), dtype=mesh_backend.int64),
        pseudo_caustic=mesh_backend.as_array(
            np.zeros(len(curves), dtype=bool), dtype=mesh_backend.bool
        ),
    )


def _parts(curves):
    """``[(lens, source, hole, closed), ...]`` per curve, as numpy."""
    off = to_np(curves.offsets)
    lens, source, hole = to_np(curves.lens), to_np(curves.source), to_np(curves.hole)
    closed = to_np(curves.closed)
    return [
        (lens[a:b], source[a:b], hole[a:b], bool(c))
        for a, b, c in zip(off[:-1], off[1:], closed)
    ]


def _sample_index(points, n=64):
    """Which of ``_one_hole``'s ``n`` even samples each circle point is."""
    angle = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    return np.rint(angle / (2 * np.pi / n)).astype(int) % n


# Arms into the origin from the west and east, out to the south and north.
# ``det A > 0`` is on the left of travel, so the ends around a small circle
# alternate: east arrives (0), north departs (pi/2), west arrives (pi), south
# departs (3 pi / 2). Points within 0.1 of the origin lie inside the hole.
WEST_IN = [(-1.0, 0.0), (-0.5, 0.0), (-0.2, 0.0), (-0.05, 0.0)]
EAST_IN = [(1.0, 0.0), (0.5, 0.0), (0.2, 0.0), (0.05, 0.0)]
SOUTH_OUT = [(0.0, -0.05), (0.0, -0.2), (0.0, -0.5), (0.0, -1.0)]
NORTH_OUT = [(0.0, 0.05), (0.0, 0.2), (0.0, 0.5), (0.0, 1.0)]
# One closed loop through the origin twice: west -> south, then east -> north.
FIGURE_EIGHT = (
    WEST_IN
    + [(0.0, 0.0)]
    + SOUTH_OUT
    + [(0.7, -0.7)]
    + EAST_IN
    + [(0.0, 0.0)]
    + NORTH_OUT
    + [(-0.7, 0.7)]
)


def test_a_loop_through_a_center_twice_splits_into_two_loops_joined_clockwise():
    holes = _one_hole()
    parts = _parts(join_at_holes(_traced((FIGURE_EIGHT, True)), holes))
    assert len(parts) == 2 and all(closed for *_, closed in parts)
    arcs = []
    for lens, source, hole, _ in parts:
        on = hole == 0
        k = _sample_index(lens[on])
        assert np.array_equal(lens[on], to_np(holes.lens)[k])
        assert np.array_equal(source[on], to_np(holes.source)[k])
        assert (np.diff(k) == -1).all()
        assert (np.hypot(*lens[~on].T) >= 0.1).all()
        assert np.array_equal(source[~on], 2.0 * lens[~on])
        arcs.append(sorted(k.tolist()))
    # west (pi) joins north (pi/2); east (0) joins south (3 pi / 2)
    assert sorted(arcs) == [list(range(17, 32)), list(range(49, 64))]


def test_joined_arms_keep_their_direction_of_travel():
    """Each arm's points outside the hole lie in one curve, consecutive and in order."""
    parts = _parts(join_at_holes(_traced((FIGURE_EIGHT, True)), _one_hole()))
    traced = [lens[hole == -1].tolist() for lens, _, hole, _ in parts]
    for arm in (WEST_IN, EAST_IN, SOUTH_OUT, NORTH_OUT):
        kept = [list(p) for p in arm if np.hypot(*p) >= 0.1]
        holding = [t for t in traced if any(p in t for p in kept)]
        assert len(holding) == 1 and all(p in holding[0] for p in kept)
        at = [holding[0].index(p) for p in kept]
        assert at == list(range(at[0], at[0] + len(kept)))


def test_a_curve_whose_ends_stop_inside_a_hole_is_closed_along_it():
    """Both ends inside the hole, as at a band gap: out north, back from the west."""
    broken = NORTH_OUT + [(-0.7, 0.7)] + WEST_IN
    parts = _parts(join_at_holes(_traced((broken, False)), _one_hole()))
    assert len(parts) == 1
    lens, _, hole, closed = parts[0]
    assert closed
    assert sorted(_sample_index(lens[hole == 0]).tolist()) == list(range(17, 32))


def test_ends_that_do_not_alternate_stay_open():
    """Two arms that both arrive: nothing departs to pair them with."""
    parts = _parts(
        join_at_holes(_traced((WEST_IN, False), (EAST_IN, False)), _one_hole())
    )
    assert len(parts) == 2 and not any(closed for *_, closed in parts)
    assert all((hole == -1).all() for _, _, hole, _ in parts)
    assert sorted(len(lens) for lens, *_ in parts) == [3, 3]


def test_a_curve_wholly_inside_a_hole_is_dropped():
    tiny = [(0.01, 0.0), (0.0, 0.01), (-0.01, 0.0), (0.0, -0.01)]
    far = [(2.0, 0.0), (2.0, 1.0), (3.0, 1.0)]
    parts = _parts(join_at_holes(_traced((tiny, True), (far, True)), _one_hole()))
    assert len(parts) == 1 and np.array_equal(parts[0][0], np.array(far))


# A single traced point just outside a hole's circle, within a leaf edge of
# it, sandwiched between two points inside the same hole: the tracer's zigzag
# can leave one, and it must count as inside the hole too, or its two ends --
# arriving and departing at the same angle -- break the alternation below.
_POKE_WEST = [(-1.0, -0.3), (-0.3, -0.15), (-0.03, -0.09)]
_POKE_EAST = [(0.03, -0.09), (0.3, -0.15), (1.0, -0.3)]
_POKE_POINT = (0.0, -0.105)


@pytest.mark.parametrize("reverse", [False, True], ids=["ccw", "cw"])
def test_a_lone_point_just_outside_a_hole_counts_as_inside_it(reverse):
    without = _POKE_WEST + _POKE_EAST
    poked = _POKE_WEST + [_POKE_POINT] + _POKE_EAST
    if reverse:
        without, poked = without[::-1], poked[::-1]
    holes = _one_hole()
    got = join_at_holes(_traced((poked, False)), holes)
    want = join_at_holes(_traced((without, False)), holes)
    for field in ("lens", "source", "offsets", "closed", "hole"):
        assert np.array_equal(
            to_np(getattr(got, field)), to_np(getattr(want, field))
        ), field
    parts = _parts(got)
    assert len(parts) == 1
    assert (parts[0][2] >= 0).any()


def test_a_closed_curve_with_one_point_outside_a_hole_is_dropped():
    poked = [(0.02, 0.0), (0.0, 0.02), (-0.02, 0.0), (0.0, -0.105)]
    far = [(2.0, 0.0), (2.0, 1.0), (3.0, 1.0)]
    parts = _parts(join_at_holes(_traced((poked, True), (far, True)), _one_hole()))
    assert len(parts) == 1 and np.array_equal(parts[0][0], np.array(far))


def test_curves_that_never_enter_a_hole_are_left_bit_for_bit():
    before = _traced(
        ([(2.0, 0.0), (2.0, 1.0), (3.0, 1.0)], True),
        ([(-2.0, 0.0), (-2.0, 1.0), (-3.0, 1.0), (-3.0, 0.5)], False),
    )
    after = join_at_holes(before, _one_hole())
    for field in ("lens", "source", "offsets", "closed", "hole"):
        assert np.array_equal(
            to_np(getattr(after, field)), to_np(getattr(before, field))
        ), field


def test_joining_at_no_hole_changes_nothing():
    before = _traced((FIGURE_EIGHT, True))
    after = join_at_holes(before, empty_holes())
    for field in ("lens", "source", "offsets", "closed", "hole"):
        assert np.array_equal(
            to_np(getattr(after, field)), to_np(getattr(before, field))
        ), field


# Two holes of radius 0.1 on the x axis with a 0.005 gap between their disks:
# A, hole 0, to the west and B, hole 1, to the east. One traced step can cross
# the gap, from a point inside A straight to a point inside B.
_A, _B = (-0.1025, 0.0), (0.1025, 0.0)


def _hole_pair(n=64):
    """Hand-built holes at ``_A`` and ``_B``: ``n`` even samples each.

    Their sources are ``lens + (5, 0)`` and ``lens + (0, 5)``, so a sample of
    one hole cannot pass for a sample of the other.
    """
    angle = 2.0 * np.pi * np.arange(n) / n
    u = 0.1 * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    lens = np.concatenate([np.asarray(_A) + u, np.asarray(_B) + u])
    shift = np.repeat([[5.0, 0.0], [0.0, 5.0]], n, axis=0)
    return CenterHoles(
        centers=_arr([_A, _B]),
        radius=_arr([0.1, 0.1]),
        offsets=mesh_backend.as_array(
            np.array([0, n, 2 * n]), dtype=mesh_backend.int64
        ),
        angle=_arr(np.concatenate([angle, angle])),
        lens=_arr(lens),
        source=_arr(lens + shift),
        growth=_arr([0.0, 0.0]),
        growth_err=_arr([0.0, 0.0]),
        pseudo_caustic=mesh_backend.as_array(
            np.array([True, True]), dtype=mesh_backend.bool
        ),
    )


def _arm(center, degrees, radii):
    """Points at distances ``radii`` from ``center``, in the direction ``degrees``."""
    t = math.radians(degrees)
    return [(center[0] + r * math.cos(t), center[1] + r * math.sin(t)) for r in radii]


# Steps across the gap: east above the axis, and west below it. A step's
# departure from its first hole takes the angle of its second point, and its
# arrival at the second hole the angle of its first: 0.0907 rad about A and
# pi - 0.0907 about B above the axis, pi + 0.0907 about B and 2 pi - 0.0907
# about A below it. No end angle here lies within 0.0075 rad of a sample.
_STEP_EAST = [(-0.0075, 0.01), (0.0075, 0.01)]
_STEP_WEST = [(0.0075, -0.01), (-0.0075, -0.01)]
# In from the north-west into A at 130 degrees, across to B, out to the
# north-east at 50 degrees, and back over the top: one loop, travelling
# counter-clockwise, which bridges the two holes once.
BRIDGED_ONCE = (
    _arm(_A, 130, [0.6, 0.3, 0.15, 0.05])
    + _STEP_EAST
    + _arm(_B, 50, [0.05, 0.15, 0.3, 0.6])
    + [(0.6, 0.9), (0.0, 1.0), (-0.6, 0.9)]
)
# One clockwise loop that crosses the gap twice: round the west from A's
# south-west (230 degrees) to its north-west (130), across to B above the
# axis, round the east from B's north-east (50) to its south-east (310), and
# back across to A below the axis.
_WEST = (
    _arm(_A, 230, [0.15, 0.3, 0.6])
    + [(-0.9, -0.3), (-0.9, 0.3)]
    + _arm(_A, 130, [0.6, 0.3, 0.15])
)
_EAST = (
    _arm(_B, 50, [0.15, 0.3, 0.6])
    + [(0.9, 0.3), (0.9, -0.3)]
    + _arm(_B, -50, [0.6, 0.3, 0.15])
)
BRIDGED_TWICE = (
    _WEST
    + _arm(_A, 130, [0.05])
    + _STEP_EAST
    + _arm(_B, 50, [0.05])
    + _EAST
    + _arm(_B, -50, [0.05])
    + _STEP_WEST
    + _arm(_A, 230, [0.05])
)


def test_a_bridge_from_one_hole_to_another_is_joined_along_both_circles():
    """Without the bridge A sees only an arrival and B only a departure.

    Neither alternates, and the loop would stay open. With it, the arm into
    A follows A clockwise, over the top, to the step, and the step follows
    B clockwise, over the top, to the arm out of B.
    """
    holes = _hole_pair()
    parts = _parts(join_at_holes(_traced((BRIDGED_ONCE, True)), holes))
    assert len(parts) == 1
    lens, source, hole, closed = parts[0]
    assert closed
    traced = hole == -1
    for c in (_A, _B):
        assert (np.hypot(*(lens[traced] - np.asarray(c)).T) >= 0.1).all()
    assert np.array_equal(source[traced], 2.0 * lens[traced])
    stored_lens, stored_source = to_np(holes.lens), to_np(holes.source)
    arcs = []
    for h, c in enumerate((_A, _B)):
        on = hole == h
        k = _sample_index(lens[on] - np.asarray(c))
        assert np.array_equal(lens[on], stored_lens[64 * h + k])
        assert np.array_equal(source[on], stored_source[64 * h + k])
        arcs.append(k.tolist())
    # A from 130 degrees down to 0.0907 rad; B from pi - 0.0907 down to 50.
    assert arcs == [list(range(23, 0, -1)), list(range(31, 8, -1))]
    assert hole.tolist() == [-1] * 9 + [0] * 23 + [1] * 23


def test_two_bridges_between_two_holes_join_into_one_loop_through_the_neck():
    """The canonical output, derived by hand from the clockwise rule.

    Segments are numbered by their first traced point, and bridges after
    them: 0 is the west arc, 1 the east arc, 2 the step east and 3 the step
    west. About A the ends lie at 0.0907 (2 departs), 130 degrees (0
    arrives), 230 degrees (0 departs) and 2 pi - 0.0907 (3 arrives), so 0
    joins 2 over A's top and 3 joins 0 under its bottom. About B, at 50
    degrees (1 departs), pi - 0.0907 (2 arrives), pi + 0.0907 (3 departs)
    and 310 degrees (1 arrives), so 2 joins 1 over B's top and 1 joins 3
    under its bottom. That is one cycle, 0 -> 2 -> 1 -> 3, which starts at
    0. Without the bridges each hole would join its own arc to itself the
    long way round, through the gap: two loops, each wrapping 260 degrees
    round one circle.
    """
    holes = _hole_pair()
    got = join_at_holes(_traced((BRIDGED_TWICE, True)), holes)
    stored_lens, stored_source = to_np(holes.lens), to_np(holes.source)

    def traced(points):
        p = np.asarray(points)
        return p, 2.0 * p, np.full(len(p), -1)

    def arc(h, first, last):
        rows = 64 * h + np.arange(first, last - 1, -1)
        return stored_lens[rows], stored_source[rows], np.full(len(rows), h)

    want = [
        traced(_WEST),
        arc(0, 23, 1),  # A, over the top: 130 degrees down to the step east
        arc(1, 31, 9),  # B, over the top: the step east down to 50 degrees
        traced(_EAST),
        arc(1, 55, 33),  # B, underneath: 310 degrees down to the step west
        arc(0, 63, 41),  # A, underneath: the step west down to 230 degrees
    ]
    want_lens, want_source, want_hole = (
        np.concatenate([piece[i] for piece in want]) for i in range(3)
    )
    assert to_np(got.offsets).tolist() == [0, len(want_lens)]
    assert to_np(got.closed).tolist() == [True]
    assert np.array_equal(to_np(got.lens), want_lens)
    assert np.array_equal(to_np(got.source), want_source)
    assert np.array_equal(to_np(got.hole), want_hole)


def test_a_bridge_left_unjoined_at_both_holes_adds_no_empty_curve():
    """Two arms end inside A, and one of them first crosses into B.

    That leaves three ends about A and one about B, so neither hole
    alternates and nothing is joined. The bridge, with no point of its own,
    would be a curve with no point at all.
    """
    across = _arm(_A, 130, [0.6, 0.3, 0.15, 0.05]) + _STEP_EAST
    into = _arm(_A, 230, [0.6, 0.3, 0.15, 0.05])
    parts = _parts(join_at_holes(_traced((across, False), (into, False)), _hole_pair()))
    assert [len(lens) for lens, *_ in parts] == [3, 3]
    assert not any(closed for *_, closed in parts)
    assert all((hole == -1).all() for _, _, hole, _ in parts)


# ---------------------------------------------------------------------------
# pseudo_caustics: each pseudo-caustic hole's curve as a closed curve
# ---------------------------------------------------------------------------


def test_pseudo_caustics_are_the_flagged_holes_curves_in_stored_order():
    """Only B, hole 1, is a pseudo-caustic: its 64 samples, as stored, make one loop."""
    holes = _hole_pair()._replace(
        pseudo_caustic=mesh_backend.as_array(
            np.array([False, True]), dtype=mesh_backend.bool
        )
    )
    got = pseudo_caustics(holes)
    assert to_np(got.offsets).tolist() == [0, 64]
    assert to_np(got.closed).tolist() == [True]
    assert to_np(got.pseudo_caustic).tolist() == [True]
    assert to_np(got.hole).tolist() == [1] * 64
    assert np.array_equal(to_np(got.lens), to_np(holes.lens)[64:])
    assert np.array_equal(to_np(got.source), to_np(holes.source)[64:])


@pytest.mark.parametrize(
    "holes",
    [
        _hole_pair()._replace(
            pseudo_caustic=mesh_backend.as_array(
                np.array([False, False]), dtype=mesh_backend.bool
            )
        ),
        empty_holes(),
    ],
    ids=["no pseudo-caustic", "no hole"],
)
def test_holes_without_a_pseudo_caustic_give_no_curve(holes):
    got = pseudo_caustics(holes)
    assert to_np(got.offsets).tolist() == [0]
    assert tuple(got.lens.shape) == (0, 2)
    assert tuple(got.source.shape) == (0, 2)
    assert tuple(got.hole.shape) == (0,)
    assert tuple(got.closed.shape) == (0,)
    assert tuple(got.pseudo_caustic.shape) == (0,)


# ---------------------------------------------------------------------------
# End to end, against known answers
# ---------------------------------------------------------------------------


def _stack_2x2(a, b, c, d):
    """``[[a, b], [c, d]]`` at every point, shape ``(N, 2, 2)``."""
    return backend.stack(
        (backend.stack((a, b), dim=-1), backend.stack((c, d), dim=-1)), dim=-2
    )


def _cored_isothermal(x, y):
    """``alpha = 1.2 theta / sqrt(theta**2 + 0.05)``: two circular critical curves."""
    r = (x * x + y * y + 0.05) ** 0.5
    return x - 1.2 * x / r, y - 1.2 * y / r


def _cored_isothermal_jacobian(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    k = 1.2 / r**3
    return _stack_2x2(
        1.0 - 1.2 / r + k * x * x, k * x * y, k * x * y, 1.0 - 1.2 / r + k * y * y
    )


def _row_fold(x, y):
    """``det A = 1 - 2y``: zero on ``y = 0.5``, a lattice row for ``fov=4``."""
    return x * 1.0, y - y * y


def _row_fold_jacobian(x, y):
    one, zero = backend.ones_like(x), backend.zeros_like(x)
    return _stack_2x2(one, zero, zero, 1.0 - 2.0 * y)


def _broken_fold(x, y):
    """``det A = 0.6 + 2y``, zero on ``y = -0.3``, and NaN wherever ``x > 1``."""
    nan = backend.where(
        x > 1.0, backend.zeros_like(x) + float("nan"), backend.zeros_like(x)
    )
    return x + nan, 0.6 * y + y * y + nan


def _broken_fold_jacobian(x, y):
    nan = backend.where(
        x > 1.0, backend.zeros_like(x) + float("nan"), backend.zeros_like(x)
    )
    one, zero = backend.ones_like(x), backend.zeros_like(x)
    return _stack_2x2(one + nan, zero, zero, 0.6 + 2.0 * y + nan)


CORED = SimpleNamespace(
    raytrace=_cored_isothermal, jacobian_lens_equation=_cored_isothermal_jacobian
)
ROW_FOLD = SimpleNamespace(
    raytrace=_row_fold, jacobian_lens_equation=_row_fold_jacobian
)
BROKEN_FOLD = SimpleNamespace(
    raytrace=_broken_fold, jacobian_lens_equation=_broken_fold_jacobian
)

# Tangential: 1 - 1.2 / r = 0 at r = sqrt(theta**2 + 0.05) = 1.2. Radial:
# 1 - 1.2 * 0.05 / r**3 = 0. The tangential caustic is the origin; the radial
# one is a circle of radius theta * |1 - 1.2 / r| there.
TANGENTIAL = math.sqrt(1.2**2 - 0.05)
_R_RADIAL = (1.2 * 0.05) ** (1.0 / 3.0)
RADIAL = math.sqrt(_R_RADIAL**2 - 0.05)
RADIAL_CAUSTIC = RADIAL * abs(1.0 - 1.2 / _R_RADIAL)


def _curves(mesh):
    """``[(lens, source, closed), ...]`` per curve, as numpy."""
    curves = critical_curves_and_caustics(mesh)
    off = to_np(curves.offsets)
    lens, source, closed = (
        to_np(curves.lens),
        to_np(curves.source),
        to_np(curves.closed),
    )
    return [
        (lens[a:b], source[a:b], bool(c)) for a, b, c in zip(off[:-1], off[1:], closed)
    ]


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def test_cored_isothermal_curves_match_the_analytic_answers():
    """Two loops at the analytic radii, oriented, with the analytic caustics.

    The lens-plane tolerance is the guaranteed one, a quarter of the
    ``min_img_sep`` passed. The caustic has no guaranteed bound; measured on
    this exact fixture the tangential caustic lies within 3.5e-7 of the origin
    and the radial caustic radius within 1.5e-6, so 1e-4 is headroom, not
    slack. ``det A > 0`` outside the tangential curve and inside the radial
    one, so keeping it on the left runs the first clockwise and the second
    counter-clockwise.
    """
    min_img_sep = 1e-2
    mesh = build_lens_mesh(
        CORED.raytrace,
        CORED.jacobian_lens_equation,
        fov=4.0,
        init_res=16,
        min_img_sep=min_img_sep,
    )
    curves = _curves(mesh)
    assert len(curves) == 2 and all(closed for _, _, closed in curves)
    (t_lens, t_src, _), (r_lens, r_src, _) = sorted(
        curves, key=lambda c: -np.hypot(*c[0].T).mean()
    )
    assert np.abs(np.hypot(*t_lens.T) - TANGENTIAL).max() < min_img_sep / 4
    assert np.abs(np.hypot(*r_lens.T) - RADIAL).max() < min_img_sep / 4
    assert _signed_area(t_lens) < 0 < _signed_area(r_lens)
    assert np.hypot(*t_src.T).max() < 1e-4
    assert np.abs(np.hypot(*r_src.T) - RADIAL_CAUSTIC).max() < 1e-4


def test_without_holes_the_curves_are_trace_band_s_and_follow_no_hole():
    mesh = to_mesh(
        build_lens_mesh(
            CORED.raytrace,
            CORED.jacobian_lens_equation,
            fov=4.0,
            init_res=16,
            min_img_sep=1e-2,
        )
    )
    got = critical_curves_and_caustics(mesh)
    raw = trace_band(mesh.critical_band)
    for field in ("lens", "source", "offsets", "closed"):
        assert np.array_equal(
            to_np(getattr(got, field)), to_np(getattr(raw, field))
        ), field
    hole = to_np(got.hole)
    assert hole.dtype == np.int64 and hole.shape == (got.lens.shape[0],)
    assert (hole == -1).all()
    assert to_np(got.pseudo_caustic).tolist() == [False] * got.closed.shape[0]
    assert mesh.holes.centers.shape[0] == 0
    assert to_np(mesh.holes.offsets).tolist() == [0]


def test_the_fov_cuts_the_tangential_circle_into_four_open_arcs():
    """The square ``[-1, 1]**2`` meets the tangential circle only near its corners.

    ``TANGENTIAL`` lies between 1 and ``sqrt(2)``, so four arcs of the circle
    are inside the fov, each ending on it, while the radial loop is whole.
    Every end sits on a boundary edge, whose two samples share the boundary
    coordinate exactly, so the end is on the boundary exactly.
    """
    mesh = build_lens_mesh(
        CORED.raytrace,
        CORED.jacobian_lens_equation,
        fov=2.0,
        init_res=8,
        min_img_sep=1e-2,
    )
    curves = _curves(mesh)
    arcs = [lens for lens, _, closed in curves if not closed]
    loops = [lens for lens, _, closed in curves if closed]
    assert len(arcs) == 4 and len(loops) == 1
    for lens in arcs:
        for end in (lens[0], lens[-1]):
            assert np.abs(end).max() == 1.0


def test_a_curve_through_lattice_points_is_traced_exactly_along_the_row():
    """Every sample on ``y = 0.5`` has ``det A`` exactly zero.

    The band's zero-is-positive rule traces one open curve along the row,
    exactly. ``det A > 0`` below the row puts it on the left of travel in
    ``-x``, so the curve runs from ``x = 2`` to ``x = -2``.
    This also pins the spec's rule that coincident consecutive points --
    which tracing a curve through lattice points produces -- are kept, not
    removed.
    """
    mesh = build_lens_mesh(
        ROW_FOLD.raytrace,
        ROW_FOLD.jacobian_lens_equation,
        fov=4.0,
        init_res=8,
        min_img_sep=2e-2,
    )
    curves = _curves(mesh)
    assert len(curves) == 1
    lens, _, closed = curves[0]
    assert not closed
    assert np.abs(lens[:, 1] - 0.5).max() < 1e-12
    assert lens[0, 0] == 2.0 and lens[-1, 0] == -2.0
    assert (np.diff(lens, axis=0) == 0).all(axis=1).any()


def test_a_curve_ends_where_the_lens_turns_nonfinite():
    """The fold ``y = -0.3`` runs into a region where the lens is NaN.

    No band leaf has a non-finite sample, so the curve is open: one end on the
    fov boundary at ``x = -2``, the other where the band stops, within a leaf
    edge of ``x = 1``.
    """
    min_img_sep = 0.05
    mesh = build_lens_mesh(
        BROKEN_FOLD.raytrace,
        BROKEN_FOLD.jacobian_lens_equation,
        fov=4.0,
        init_res=4,
        min_img_sep=min_img_sep,
    )
    curves = _curves(mesh)
    assert len(curves) == 1
    lens, _, closed = curves[0]
    assert not closed
    assert np.abs(lens[:, 1] + 0.3).max() < min_img_sep / 4
    left, right = sorted([lens[0, 0], lens[-1, 0]])
    assert left == -2.0
    assert 1.0 - min_img_sep < right <= 1.0


# ---------------------------------------------------------------------------
# End to end: curves through singular centers, repaired at their holes
# ---------------------------------------------------------------------------

# Two singular isothermal spheres of Einstein radius 1, 0.6" apart: each sits
# in the other's field with kappa = shear = 1/1.2 > 1/2, so four branches of
# det A = 0 end at each center. The first is on the lattice origin, where the
# lens is NaN, so the band has a gap there and trace_band's curves are open.
TWO_SIS = [(0.0, 0.0), (0.6, 0.0)]
TWO_SIS_BUILD = dict(fov=5.0, init_res=10, min_img_sep=2e-2)


def _einstein_radii(centers, b):
    """``b`` for every center: one radius for all, or one per center."""
    return [b] * len(centers) if np.isscalar(b) else list(b)


def _sis_pair(centers, b=1.0):
    """A lens-like object: singular isothermal spheres of Einstein radius ``b``.

    ``b`` is one radius for every center, or a sequence of one per center.
    """
    radii = _einstein_radii(centers, b)

    def raytrace(x, y):
        bx, by = x * 1.0, y * 1.0
        for (cx, cy), rein in zip(centers, radii):
            dx, dy = x - cx, y - cy
            r = (dx * dx + dy * dy) ** 0.5
            bx, by = bx - rein * dx / r, by - rein * dy / r
        return bx, by

    def jacobian(x, y):
        a00, a01, a11 = x * 0.0 + 1.0, x * 0.0, x * 0.0 + 1.0
        for (cx, cy), rein in zip(centers, radii):
            dx, dy = x - cx, y - cy
            r = (dx * dx + dy * dy) ** 0.5
            k = rein / r**3
            a00 = a00 - rein / r + k * dx * dx
            a01 = a01 + k * dx * dy
            a11 = a11 - rein / r + k * dy * dy
        return _stack_2x2(a00, a01, a01, a11)

    return SimpleNamespace(raytrace=raytrace, jacobian_lens_equation=jacobian)


def _sis_pair_map(p, centers, b=1.0):
    """``_sis_pair`` on numpy points ``(N, 2)``."""
    out = p.copy()
    for c, rein in zip(centers, _einstein_radii(centers, b)):
        d = p - np.asarray(c)
        out -= rein * d / np.hypot(d[:, 0], d[:, 1])[:, None]
    return out


def _with_holes(curves, holes):
    """``curves`` with the holes they were joined at alongside, for the count helpers."""
    return SimpleNamespace(**curves._asdict(), holes=holes)


def _grid_of(curves, dx=0.02, margin=0.2):
    """A pixel grid ``(x0, y0, dx, nx, ny)`` over every caustic and hole curve."""
    pts = np.concatenate([to_np(curves.source), to_np(curves.holes.source)])
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    nx, ny = np.ceil((hi - lo) / dx).astype(int)
    return float(lo[0]), float(lo[1]), dx, int(nx), int(ny)


def _pixels(grid):
    x0, y0, dx, nx, ny = grid
    X, Y = np.meshgrid(x0 + (np.arange(nx) + 0.5) * dx, y0 + (np.arange(ny) + 0.5) * dx)
    return np.stack([X.ravel(), Y.ravel()], axis=-1)


def _winding(poly, grid):
    """Winding number of a closed polyline round every pixel center, by signed crossings of a rightward ray."""
    x0, y0, dx, nx, ny = grid
    a, b = poly, np.roll(poly, -1, axis=0)
    lo, hi = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    sgn = np.where(b[:, 1] > a[:, 1], 1, -1)
    r_lo = np.maximum(np.ceil((lo - y0) / dx - 0.5).astype(np.int64), 0)
    r_hi = np.minimum(np.ceil((hi - y0) / dx - 0.5).astype(np.int64) - 1, ny - 1)
    nr = np.maximum(r_hi - r_lo + 1, 0)
    e = np.repeat(np.arange(len(a)), nr)
    r = r_lo[e] + (np.arange(nr.sum()) - np.repeat(np.cumsum(nr) - nr, nr))
    py = y0 + (r + 0.5) * dx
    xcr = a[e, 0] + (py - a[e, 1]) / (b[e, 1] - a[e, 1]) * (b[e, 0] - a[e, 0])
    k = np.clip(np.ceil((xcr - x0) / dx - 0.5).astype(np.int64), 0, nx)
    diff = np.zeros((ny, nx + 1), dtype=np.int64)
    np.add.at(diff, (r, np.zeros_like(r)), sgn[e])
    np.add.at(diff, (r, k), -sgn[e])
    return np.cumsum(diff, axis=1)[:, :nx]


def _lens_winding(loop, s):
    """Winding number of a closed lens-plane polyline round the point ``s``."""
    v = loop - np.asarray(s)
    a, b = v, np.roll(v, -1, axis=0)
    turns = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(axis=1))
    return int(np.rint(turns.sum() / (2 * np.pi)))


def _count(curves, grid):
    """``N = 1 + 2 sum_c w(K_c) + sum_s sigma_s w(h_s)``, ``sigma_s = -1 - 2 sum_c w(loop_c, s)``."""
    x0, y0, dx, nx, ny = grid
    # The pseudo-caustic loops are the hole curves, which the second sum counts.
    pseudo = to_np(curves.pseudo_caustic)
    parts = [part for part, p in zip(_parts(curves), pseudo) if not p]
    N = np.ones((ny, nx), dtype=np.int64)
    for _, source, _, closed in parts:
        assert closed
        N += 2 * _winding(source, grid)
    offsets = to_np(curves.holes.offsets)
    centers, hole_source = to_np(curves.holes.centers), to_np(curves.holes.source)
    for h, s in enumerate(centers):
        sigma = -1 - 2 * sum(_lens_winding(lens, s) for lens, *_ in parts)
        N += sigma * _winding(hole_source[offsets[h] : offsets[h + 1]], grid)
    return N


def _curve_mask(curves, grid, width):
    """Pixels within ``width`` of a caustic or hole curve, where either count is at its resolution."""
    x0, y0, dx, nx, ny = grid
    pts = np.concatenate([to_np(curves.source), to_np(curves.holes.source)])
    return (cKDTree(pts).query(_pixels(grid))[0] < width).reshape(ny, nx)


def _brute_counts(fn, grid, lo, hi, h, singular, excl):
    """Images per pixel center: lens-plane triangles whose piecewise-linear image covers it."""
    x0, y0, dx, nx, ny = grid
    g = np.arange(lo, hi + h / 2, h) + 3.3e-5
    X, Y = np.meshgrid(g, g)
    B = fn(np.stack([X.ravel(), Y.ravel()], axis=-1)).reshape(*X.shape, 2)
    cx = 0.5 * (g[:-1] + g[1:])
    CX, CY = np.meshgrid(cx, cx)
    keep = np.ones(CX.shape, dtype=bool)
    for s in singular:
        keep &= np.hypot(CX - s[0], CY - s[1]) > excl
    p00, p10, p01, p11 = B[:-1, :-1], B[:-1, 1:], B[1:, :-1], B[1:, 1:]
    counts = np.zeros(ny * nx, dtype=np.int64)
    for A_, B_, C_ in ((p00, p10, p11), (p00, p11, p01)):
        a, b, c = A_[keep], B_[keep], C_[keep]
        xs = np.stack([a[:, 0], b[:, 0], c[:, 0]])
        ys = np.stack([a[:, 1], b[:, 1], c[:, 1]])
        i0 = np.maximum(np.ceil((xs.min(0) - x0) / dx - 0.5), 0).astype(np.int64)
        i1 = np.minimum(np.floor((xs.max(0) - x0) / dx - 0.5), nx - 1).astype(np.int64)
        j0 = np.maximum(np.ceil((ys.min(0) - y0) / dx - 0.5), 0).astype(np.int64)
        j1 = np.minimum(np.floor((ys.max(0) - y0) / dx - 0.5), ny - 1).astype(np.int64)
        ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
        tot = ni * nj
        sel = np.flatnonzero(tot > 0)
        tt = np.repeat(sel, tot[sel])
        w = np.arange(tt.size) - np.repeat(np.cumsum(tot[sel]) - tot[sel], tot[sel])
        ii, jj = i0[tt] + w % ni[tt], j0[tt] + w // ni[tt]
        px, py = x0 + (ii + 0.5) * dx, y0 + (jj + 0.5) * dx
        ax, ay = a[tt, 0] - px, a[tt, 1] - py
        bx, by = b[tt, 0] - px, b[tt, 1] - py
        qx, qy = c[tt, 0] - px, c[tt, 1] - py
        w1 = bx * qy - by * qx
        w2 = qx * ay - qy * ax
        w3 = ax * by - ay * bx
        ins = ((w1 >= 0) & (w2 >= 0) & (w3 >= 0)) | ((w1 <= 0) & (w2 <= 0) & (w3 <= 0))
        np.add.at(counts, (jj * nx + ii)[ins], 1)
    return counts.reshape(ny, nx)


@pytest.fixture(scope="module")
def two_sis():
    lens = _sis_pair(TWO_SIS)
    mesh = to_mesh(
        build_lens_mesh(
            lens.raytrace, lens.jacobian_lens_equation, **TWO_SIS_BUILD, centers=TWO_SIS
        )
    )
    return mesh, _with_holes(critical_curves_and_caustics(mesh), mesh.holes)


def test_curves_through_singular_centers_come_out_closed_and_chord_free(two_sis):
    """Without holes a chord crosses a cut of radius ~1"; traced steps are below 0.01"."""
    mesh, curves = two_sis
    assert not to_np(trace_band(mesh.critical_band).closed).all()
    pseudo = to_np(curves.pseudo_caustic)
    parts = [part for part, p in zip(_parts(curves), pseudo) if not p]
    assert parts and all(closed for *_, closed in parts)
    centers, radius = to_np(mesh.holes.centers), to_np(mesh.holes.radius)
    for lens, source, hole, _ in parts:
        traced = (hole == -1) & (np.roll(hole, -1) == -1)
        step = np.hypot(*(np.roll(source, -1, axis=0) - source).T)
        assert step[traced].max() < 0.05
        for h in range(centers.shape[0]):
            d = np.hypot(*(lens - centers[h]).T)
            assert np.allclose(d[hole == h], radius[h], rtol=0, atol=1e-12)
            assert (d[hole == -1] >= radius[h]).all()


def test_the_count_from_the_repaired_curves_matches_a_brute_force_count(two_sis):
    _, curves = two_sis
    grid = _grid_of(curves)
    truth = _brute_counts(
        lambda p: _sis_pair_map(p, TWO_SIS),
        grid,
        lo=-4.0,
        hi=4.6,
        h=0.005,
        singular=TWO_SIS,
        excl=0.02,
    )
    band = _curve_mask(curves, grid, 0.06)
    assert (~band).mean() > 0.75
    assert np.array_equal(_count(curves, grid)[~band], truth[~band])


def test_moving_a_center_off_the_lattice_leaves_the_count_unchanged(two_sis):
    _, curves = two_sis
    shifted = [(0.0003, 0.0002), TWO_SIS[1]]
    lens = _sis_pair(shifted)
    mesh = to_mesh(
        build_lens_mesh(
            lens.raytrace, lens.jacobian_lens_equation, **TWO_SIS_BUILD, centers=shifted
        )
    )
    moved = _with_holes(critical_curves_and_caustics(mesh), mesh.holes)
    grid = _grid_of(curves)
    band = _curve_mask(curves, grid, 0.06) | _curve_mask(moved, grid, 0.06)
    assert np.array_equal(_count(moved, grid)[~band], _count(curves, grid)[~band])


def _assert_ends_with_whole_pseudo_caustics(curves, holes):
    """The last curves are one closed loop per pseudo-caustic hole: its stored circle and hole curve."""
    pseudo = np.flatnonzero(to_np(holes.pseudo_caustic))
    flags = to_np(curves.pseudo_caustic)
    n = flags.shape[0] - pseudo.shape[0]
    assert flags.tolist() == [False] * n + [True] * pseudo.shape[0]
    offsets = to_np(holes.offsets)
    parts = _parts(curves)[n:]
    for h, (lens, source, hole, closed) in zip(pseudo, parts):
        rows = slice(offsets[h], offsets[h + 1])
        assert closed
        assert hole.tolist() == [h] * (offsets[h + 1] - offsets[h])
        assert np.array_equal(lens, to_np(holes.lens)[rows])
        assert np.array_equal(source, to_np(holes.source)[rows])


def test_an_sis_pseudo_caustic_no_critical_curve_reaches_is_among_the_curves():
    """The critical circle ``|theta| = 1`` never nears the hole, so nothing is joined there.

    The pseudo-caustic still comes out, as the hole's whole curve after the
    critical circle.
    """
    lens = _sis_pair([(0.0, 0.0)])
    mesh = to_mesh(
        build_lens_mesh(
            lens.raytrace,
            lens.jacobian_lens_equation,
            fov=4.0,
            init_res=8,
            min_img_sep=1e-2,
            centers=[(0.0, 0.0)],
        )
    )
    curves = critical_curves_and_caustics(mesh)
    assert to_np(mesh.holes.pseudo_caustic).tolist() == [True]
    assert to_np(curves.closed).tolist() == [True, True]
    critical, _ = _parts(curves)
    assert (critical[2] == -1).all()
    assert np.abs(np.hypot(*critical[0].T) - 1.0).max() < 1e-2
    _assert_ends_with_whole_pseudo_caustics(curves, mesh.holes)


def test_holes_critical_curves_reach_still_add_their_whole_pseudo_caustics(two_sis):
    """Arcs joined into the critical curves leave each pseudo-caustic whole as well."""
    mesh, curves = two_sis
    joined = join_at_holes(trace_band(mesh.critical_band), mesh.holes)
    assert (to_np(joined.hole) >= 0).any()
    n = joined.closed.shape[0]
    assert np.array_equal(to_np(curves.offsets)[: n + 1], to_np(joined.offsets))
    m = to_np(joined.offsets)[-1]
    for field in ("lens", "source", "hole"):
        assert np.array_equal(
            to_np(getattr(curves, field))[:m], to_np(getattr(joined, field))
        ), field
    assert np.array_equal(to_np(curves.closed)[:n], to_np(joined.closed))
    _assert_ends_with_whole_pseudo_caustics(curves, mesh.holes)


# ---------------------------------------------------------------------------
# End to end: a traced step from one hole's disk straight into another's
# ---------------------------------------------------------------------------

# Disks only need not overlap, and at the size floor a traced step can be as
# long as half the stored min_img_sep, so a step can cross the gap between two
# holes. Here an SIS of Einstein radius 0.02 sits beside TWO_SIS's center on
# the lattice origin, 0.0203" from it at 40 degrees: the two holes, of radius
# 0.01, are 0.0003" apart.
_COMPANION = (
    0.0203 * math.cos(math.radians(40.0)),
    0.0203 * math.sin(math.radians(40.0)),
)
WITH_COMPANION = TWO_SIS + [_COMPANION]
WITH_COMPANION_B = [1.0, 1.0, 0.02]


def _hole_to_hole_steps(mesh):
    """How many traced steps of ``mesh``'s band go from one hole's disk straight into another's."""
    raw = trace_band(to_mesh(mesh).critical_band)
    lens, off = to_np(raw.lens), to_np(raw.offsets)
    centers, radius = to_np(mesh.holes.centers), to_np(mesh.holes.radius)
    inside = np.hypot(*(lens[:, None, :] - centers[None]).transpose(2, 0, 1)) < radius
    tag = np.where(inside.any(axis=1), inside.argmax(axis=1), -1)
    steps = 0
    for a, b in zip(off[:-1], off[1:]):
        t = tag[a:b]
        steps += int(((t[:-1] >= 0) & (t[1:] >= 0) & (t[:-1] != t[1:])).sum())
    return steps


def _traced_steps(curves):
    """Every step between two consecutive traced points, a loop's closing step included.

    Returns the steps' lens-plane ends, ``(K, 2)`` each, and their
    source-plane lengths, ``(K,)``.
    """
    p, q, length = [], [], []
    for lens, source, hole, closed in _parts(curves):
        i = np.arange(len(lens))
        j = (i + 1) % len(lens)
        step = (hole[i] == -1) & (hole[j] == -1) & (closed | (j > 0))
        p.append(lens[i[step]])
        q.append(lens[j[step]])
        length.append(np.hypot(*(source[j[step]] - source[i[step]]).T))
    return np.concatenate(p), np.concatenate(q), np.concatenate(length)


def _steps_entering_holes(mesh, curves):
    """How many traced lens-plane steps come closer to a hole's center than its radius."""
    p, q, _ = _traced_steps(curves)
    d = q - p
    entering = 0
    for c, r in zip(to_np(mesh.holes.centers), to_np(mesh.holes.radius)):
        t = ((c - p) * d).sum(axis=1) / np.maximum((d * d).sum(axis=1), 1e-300)
        closest = p + np.clip(t, 0.0, 1.0)[:, None] * d
        entering += int((np.hypot(*(closest - c).T) < r).sum())
    return entering


def _assert_closed_off_the_holes(mesh, curves):
    """Every curve closed, its hole points on their circles, and no traced point in a hole."""
    parts = _parts(curves)
    assert parts and all(closed for *_, closed in parts)
    centers, radius = to_np(mesh.holes.centers), to_np(mesh.holes.radius)
    for lens, _, hole, _ in parts:
        for h in range(centers.shape[0]):
            d = np.hypot(*(lens - centers[h]).T)
            assert np.allclose(d[hole == h], radius[h], rtol=0, atol=1e-12)
            assert (d[hole == -1] >= radius[h]).all()


def _assert_closed_and_chord_free(mesh, curves):
    """``_assert_closed_off_the_holes``, and no traced step above 0.05"."""
    _assert_closed_off_the_holes(mesh, curves)
    assert (_traced_steps(curves)[2] < 0.05).all()


@pytest.fixture(scope="module")
def bridged_sis():
    lens = _sis_pair(WITH_COMPANION, b=WITH_COMPANION_B)
    mesh = build_lens_mesh(
        lens.raytrace,
        lens.jacobian_lens_equation,
        **TWO_SIS_BUILD,
        centers=WITH_COMPANION,
    )
    return mesh, _with_holes(critical_curves_and_caustics(mesh), mesh.holes)


def test_curves_bridged_to_a_singular_companion_come_out_closed_and_chord_free(
    bridged_sis,
):
    mesh, curves = bridged_sis
    assert _hole_to_hole_steps(mesh) > 0
    _assert_closed_off_the_holes(mesh, curves)
    # No 0.05" bound on the caustic's steps here: the companion's curve runs
    # where the origin's SIS stretches the caustic 37-94 times, so resolved
    # steps there reach about 0.12". A chord is an interpolation across a
    # hole, and that is what this checks directly.
    assert _steps_entering_holes(mesh, curves) == 0


def test_the_count_through_a_bridged_hole_matches_a_brute_force_count(bridged_sis):
    _, curves = bridged_sis
    grid = _grid_of(curves)
    truth = _brute_counts(
        lambda p: _sis_pair_map(p, WITH_COMPANION, b=WITH_COMPANION_B),
        grid,
        lo=-4.0,
        hi=4.6,
        h=0.005,
        singular=WITH_COMPANION,
        excl=0.02,
    )
    band = _curve_mask(curves, grid, 0.06)
    assert (~band).mean() > 0.75
    assert np.array_equal(_count(curves, grid)[~band], truth[~band])


def test_curves_bridged_to_a_regular_center_come_out_closed_and_chord_free():
    """A third center where the lens is regular, 0.02095" from the origin.

    Its hole lies 0.00095" from the origin's, and one traced step of 0.00123"
    goes from it straight into the origin's.
    """
    centers = TWO_SIS + [(-0.01241, 0.01688)]
    lens = _sis_pair(TWO_SIS)
    mesh = to_mesh(
        build_lens_mesh(
            lens.raytrace, lens.jacobian_lens_equation, **TWO_SIS_BUILD, centers=centers
        )
    )
    assert _hole_to_hole_steps(mesh) > 0
    _assert_closed_and_chord_free(mesh, critical_curves_and_caustics(mesh))


def test_triangle_segments_keep_the_positive_side_on_the_left():
    # (0,0), (1,0), (0,1), positively oriented; only vertex 0 is positive.
    tri = mesh_backend.as_array([[0, 1, 2]], dtype=mesh_backend.int64)
    positive = mesh_backend.as_array([True, False, False])
    start, end = triangle_segments(tri, positive)
    # From edge (0, 1) to edge (2, 0): travelling from (0.5, 0) towards
    # (0, 0.5) keeps vertex 0 on the left.
    assert to_np(start).tolist() == [[0, 1]]
    assert to_np(end).tolist() == [[0, 2]]


def test_triangle_segments_skip_triangles_of_one_class():
    tri = mesh_backend.as_array([[0, 1, 2]], dtype=mesh_backend.int64)
    for value in (True, False):
        start, _ = triangle_segments(tri, mesh_backend.as_array([value] * 3))
        assert start.shape[0] == 0


def test_chain_segments_close_a_fan_into_one_counter_clockwise_loop():
    # A unit square fanned about its center, vertex 4, the only positive one.
    tri = mesh_backend.as_array(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=mesh_backend.int64
    )
    positive = mesh_backend.as_array([False, False, False, False, True])
    start, end = triangle_segments(tri, positive)
    edges, offsets, closed = chain_segments(start, end, 5)
    assert to_np(offsets).tolist() == [0, 4]
    assert to_np(closed).tolist() == [True]
    walk = [tuple(e) for e in to_np(edges).tolist()]
    i = walk.index((0, 4))
    assert walk[i:] + walk[:i] == [(0, 4), (1, 4), (2, 4), (3, 4)]


def test_chain_segments_of_nothing_is_no_curve():
    empty = mesh_backend.zeros((0, 2), dtype=mesh_backend.int64)
    edges, offsets, closed = chain_segments(empty, empty, 3)
    assert edges.shape == (0, 2)
    assert to_np(offsets).tolist() == [0]
    assert closed.shape[0] == 0


def test_edge_zeros_interpolate_the_zero_in_every_plane():
    edges = mesh_backend.as_array([[0, 1]], dtype=mesh_backend.int64)
    field = _arr([-1.0, 3.0])
    a = _arr([[0.0, 0.0], [4.0, 0.0]])
    b = _arr([[0.0, 1.0], [0.0, 5.0]])
    za, zb = edge_zeros(edges, field, (a, b))
    assert to_np(za).tolist() == [[1.0, 0.0]]
    assert to_np(zb).tolist() == [[0.0, 2.0]]


# ---------------------------------------------------------------------------
# merge_centers
# ---------------------------------------------------------------------------


def test_empty_holes_has_no_hole():
    holes = empty_holes()
    assert tuple(holes.centers.shape) == (0, 2)
    assert tuple(holes.lens.shape) == (0, 2) and tuple(holes.source.shape) == (0, 2)
    for field in ("radius", "angle", "growth", "growth_err", "pseudo_caustic"):
        assert tuple(getattr(holes, field).shape) == (0,), field
    assert holes.pseudo_caustic.dtype == mesh_backend.bool
    assert to_np(holes.offsets).tolist() == [0]


@pytest.mark.parametrize(
    "centers", [None, [], np.zeros((0, 2))], ids=["None", "empty list", "(0, 2)"]
)
def test_merge_centers_of_nothing_is_empty(centers):
    got, radius = merge_centers(centers, 0.01)
    assert tuple(got.shape) == (0, 2) and tuple(radius.shape) == (0,)


@pytest.mark.parametrize(
    "centers",
    [[(0.3, -0.2)], ((0.3, -0.2),), np.array([[0.3, -0.2]]), _arr([[0.3, -0.2]])],
    ids=["list", "tuple", "numpy", "backend"],
)
def test_a_lone_center_of_any_array_like_keeps_its_exact_position(centers):
    got, radius = merge_centers(centers, 0.01)
    assert to_np(got).tolist() == [[0.3, -0.2]]
    assert to_np(radius).tolist() == [0.01]


def test_coincident_centers_share_one_hole_at_their_common_position():
    got, radius = merge_centers([(2.0, 2.0), (0.3, 0.1), (0.3, 0.1)], 0.01)
    assert to_np(got).tolist() == [[0.3, 0.1], [2.0, 2.0]]
    assert to_np(radius).tolist() == [0.01, 0.01]


def test_centers_closer_than_twice_min_img_sep_merge_at_their_mean():
    got, radius = merge_centers([(0.0, 0.0), (0.015, 0.0)], 0.01)
    assert np.allclose(to_np(got), [[0.0075, 0.0]], rtol=0, atol=1e-15)
    assert np.allclose(to_np(radius), [0.0175], rtol=0, atol=1e-15)


def test_centers_exactly_twice_min_img_sep_apart_keep_their_own_holes():
    got, _ = merge_centers([(0.0, 0.0), (0.02, 0.0)], 0.01)
    assert got.shape[0] == 2


def test_merging_repeats_until_no_two_disks_overlap():
    """``a`` and ``b`` link; ``c`` is farther than 2 * min_img_sep from both,
    yet its disk overlaps theirs once they merge, so all three share a hole."""
    pts = np.array([(0.0, 0.0), (0.019, 0.0), (0.0095, 0.025)])
    got, radius = merge_centers(pts, 0.01)
    mean = pts.mean(axis=0)
    assert got.shape[0] == 1
    assert np.allclose(to_np(got)[0], mean, rtol=0, atol=1e-15)
    want = 0.01 + np.hypot(*(pts - mean).T).max()
    assert np.isclose(to_np(radius)[0], want, rtol=0, atol=1e-15)


def test_merged_holes_are_disjoint_hold_their_centers_and_ignore_input_order():
    rng = np.random.default_rng(3)
    pts = np.concatenate(
        [rng.uniform(-1.0, 1.0, (20, 2)), rng.uniform(0.0, 0.03, (10, 2))]
    )
    got, radius = merge_centers(pts, 0.01)
    g, r = to_np(got), to_np(radius)
    gap = np.hypot(*(g[:, None, :] - g[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(gap, np.inf)
    assert (gap >= r[:, None] + r[None, :]).all()
    held = np.hypot(*(pts[:, None, :] - g[None, :, :]).transpose(2, 0, 1)) < r[None, :]
    assert (held.sum(axis=1) == 1).all()
    for seed in range(3):
        perm = np.random.default_rng(seed).permutation(len(pts))
        again, again_radius = merge_centers(pts[perm], 0.01)
        assert np.array_equal(to_np(again), g)
        assert np.array_equal(to_np(again_radius), r)


# ---------------------------------------------------------------------------
# sample_holes
# ---------------------------------------------------------------------------


def sis_raytrace(c, b):
    """A singular isothermal sphere at ``c``: ``f(c + r u) = c + (r - b) u``."""

    def raytrace(x, y):
        dx, dy = x - c[0], y - c[1]
        r = backend.sqrt(dx * dx + dy * dy)
        return x - b * dx / r, y - b * dy / r

    return raytrace


def point_mass_raytrace(c, theta_e):
    """A point mass at ``c``: ``f(c + r u) = c + (r - theta_e**2 / r) u``."""

    def raytrace(x, y):
        dx, dy = x - c[0], y - c[1]
        r2 = dx * dx + dy * dy
        return x - theta_e**2 * dx / r2, y - theta_e**2 * dy / r2

    return raytrace


def sis_in_shear_raytrace(c, b, kappa, gamma_1, gamma_2):
    """The SIS of ``sis_raytrace`` in a uniform convergence and shear about ``c``:
    the smooth pull other lenses add at an isothermal center."""
    sis = sis_raytrace(c, b)

    def raytrace(x, y):
        dx, dy = x - c[0], y - c[1]
        bx, by = sis(x, y)
        return (
            bx - (kappa + gamma_1) * dx - gamma_2 * dy,
            by - gamma_2 * dx - (kappa - gamma_1) * dy,
        )

    return raytrace


def power_law_raytrace(c, b, t):
    """A circular power law of slope ``t`` at ``c``:
    ``f(c + r u) = c + (r - b**t * r**(1 - t)) u``, the SIS at ``t = 1``."""

    def raytrace(x, y):
        dx, dy = x - c[0], y - c[1]
        k = b**t * backend.sqrt(dx * dx + dy * dy) ** -t
        return x - k * dx, y - k * dy

    return raytrace


def affine_raytrace(x, y):
    return 0.7 * x + 0.1 * y, -0.2 * x + 0.9 * y


def nan_where(raytrace, where):
    """``raytrace`` with NaN wherever ``where(x, y)`` holds."""

    def broken(x, y):
        bx, by = raytrace(x, y)
        nan = backend.where(
            where(x, y), backend.zeros_like(x) + float("nan"), backend.zeros_like(x)
        )
        return bx + nan, by + nan

    return broken


def sample(raytrace, centers, radius, min_img_sep, batch_size=None):
    """``sample_holes`` through ``make_sampler``, and every point it traced."""
    calls = []

    def recorded(x, y):
        calls.append(requested(np.stack([to_np(x), to_np(y)], axis=-1)))
        return raytrace(x, y)

    holes = sample_holes(
        make_sampler(recorded, None, None, batch_size),
        _arr(centers),
        _arr(radius),
        min_img_sep,
        batch_size,
    )
    return holes, calls


def test_an_sis_hole_curve_is_its_analytic_circle_sampled_to_min_img_sep():
    c, b, r = (0.3, -0.2), 1.0, 0.01
    holes, _ = sample(sis_raytrace(c, b), [c], [r], r)
    angle, lens, source = to_np(holes.angle), to_np(holes.lens), to_np(holes.source)
    u = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    assert to_np(holes.offsets).tolist() == [0, angle.size]
    assert angle[0] >= 0 and angle[-1] < 2 * np.pi and (np.diff(angle) > 0).all()
    assert np.allclose(lens, np.array(c) + r * u, rtol=0, atol=1e-14)
    assert np.allclose(source, np.array(c) + (r - b) * u, rtol=0, atol=1e-12)
    chords = np.hypot(*(np.roll(source, -1, axis=0) - source).T)
    assert chords.max() <= r


# An isothermal center alone, and in the uniform convergence and shear other
# lenses add there. The hole curve's size then changes by about ``r / b`` of
# itself between radii -- 1e-2 here -- which a slope between two circles reads
# as a growth of order 5e-3 rather than 0.
ISOTHERMAL = {
    "sis": sis_raytrace((0.3, -0.2), 1.0),
    "sis in shear": sis_in_shear_raytrace((0.3, -0.2), 1.0, 0.1, 0.2, -0.1),
}


@pytest.mark.parametrize("name", ISOTHERMAL)
def test_an_isothermal_center_is_a_pseudo_caustic(name):
    holes, _ = sample(ISOTHERMAL[name], [(0.3, -0.2)], [0.01], 0.01)
    growth, err = to_np(holes.growth)[0], to_np(holes.growth_err)[0]
    assert to_np(holes.pseudo_caustic).tolist() == [True]
    assert abs(growth) <= err
    assert abs(growth) < 1e-5


@pytest.mark.parametrize("t", [1.0 - 1e-4, 1.0 + 1e-4])
def test_a_power_law_a_hair_off_isothermal_is_no_pseudo_caustic(t):
    """A slope ``t`` gives ``growth = 1 - t``, resolved well below 1e-4."""
    c = (0.3, -0.2)
    holes, _ = sample(power_law_raytrace(c, 1.0, t), [c], [0.01], 0.01)
    growth, err = to_np(holes.growth)[0], to_np(holes.growth_err)[0]
    assert to_np(holes.pseudo_caustic).tolist() == [False]
    assert abs(growth - (1.0 - t)) <= err < 2.5e-5


def test_both_centers_of_two_overlapping_sie_lenses_are_pseudo_caustics():
    """Each SIE's pseudo-caustic is shifted and sheared by the other's smooth
    deflection; a slope between two circles read these as 8e-4 and -7e-4."""
    cosmology = FlatLambdaCDM(name="cosmo")
    lenses = [
        SIE(cosmology=cosmology, name=name, x0=c, y0=c, q=0.4, phi=phi, Rein=1.2, s=0.0)
        for name, c, phi in (("sie_1", 0.5, np.pi / 4), ("sie_2", 0.001, np.pi))
    ]
    lens = SinglePlane(
        cosmology=cosmology, name="lens", z_l=0.5, z_s=1.5, lenses=lenses
    )
    centers = [(0.001, 0.001), (0.5, 0.5)]
    holes, _ = sample(lens.raytrace, centers, [0.005, 0.005], 0.005)
    assert to_np(holes.pseudo_caustic).tolist() == [True, True]


def test_a_point_mass_hole_curve_grows_as_the_hole_shrinks():
    c = (0.1, 0.2)
    holes, _ = sample(point_mass_raytrace(c, 0.1), [c], [0.01], 0.01)
    assert abs(to_np(holes.growth)[0] + 1.0) < 0.01
    assert to_np(holes.pseudo_caustic).tolist() == [False]


def test_a_regular_point_hole_curve_shrinks_with_the_hole():
    holes, _ = sample(affine_raytrace, [(0.4, -0.3)], [0.01], 0.01)
    assert abs(to_np(holes.growth)[0] - 1.0) < 1e-9
    assert to_np(holes.pseudo_caustic).tolist() == [False]


def test_a_lens_mapping_every_circle_to_one_point_has_no_growth():
    def constant(x, y):
        return 0.0 * x + 0.5, 0.0 * y - 0.1

    holes, _ = sample(constant, [(0.4, -0.3)], [0.01], 0.01)
    assert np.isnan(to_np(holes.growth)[0])
    assert to_np(holes.pseudo_caustic).tolist() == [False]


def test_a_hole_curve_too_large_to_resolve_stops_at_the_cap_and_warns():
    """A 1" point mass maps a 0.005" circle to a loop of radius about 200":
    ``2**16`` samples cannot bring its chords down to 0.005"."""
    c = (0.0, 0.0)
    with pytest.warns(UserWarning, match=r"stopped at \d+ samples \(cap 65536\)"):
        holes, _ = sample(point_mass_raytrace(c, 1.0), [c], [0.005], 0.005)
    assert to_np(holes.offsets)[-1] <= HOLE_MAX_SAMPLES


def test_a_hole_frozen_below_the_cap_warns_with_its_own_sample_count():
    """A point mass off the hole's center makes the hole curve uneven.

    Refinement then bisects only some intervals, so a round can stop short
    of the cap because the next one would pass it. The warning states the
    count the hole kept, not the cap.
    """
    c = (0.0, 0.0)
    with pytest.warns(UserWarning, match=r"\(cap 65536\)") as got:
        holes, _ = sample(point_mass_raytrace((0.004, 0.0), 0.3), [c], [0.005], 0.005)
    n = int(to_np(holes.offsets)[-1])
    assert n < HOLE_MAX_SAMPLES
    assert any(f"stopped at {n} samples (cap 65536)" in str(w.message) for w in got)


def test_a_lens_not_finite_on_a_hole_circle_raises_naming_the_center():
    c = (0.4, -0.3)
    broken = nan_where(affine_raytrace, lambda x, y: x > c[0])
    with pytest.raises(
        ValueError,
        match=r"not finite on the hole circle of radius 0\.01 around \(0\.4, -0\.3\)",
    ):
        sample(broken, [c], [0.01], 0.01)


@pytest.mark.parametrize(
    "nan_within, named",
    [(0.005, r"radius 0\.0025 "), (3e-4, r"radius 0\.00015625 ")],
    ids=["quarter", "sixty-fourth"],
)
def test_every_growth_circle_down_to_a_64th_of_the_radius_must_be_finite(
    nan_within, named
):
    c = (0.4, -0.3)
    broken = nan_where(
        affine_raytrace,
        lambda x, y: (x - c[0]) ** 2 + (y - c[1]) ** 2 < nan_within**2,
    )
    with pytest.raises(ValueError, match=named):
        sample(broken, [c], [0.01], 0.01)


def test_holes_raytrace_only_their_circles_and_batching_changes_nothing():
    c, r = (0.3, -0.2), 0.01
    holes, calls = sample(sis_raytrace(c, 1.0), [c], [r], r)
    d = np.hypot(*(np.concatenate(calls) - np.array(c)).T)
    radii = r / 4.0 ** np.arange(4)
    assert np.isclose(d[:, None], radii, rtol=0, atol=1e-14).any(axis=1).all()
    batched, batched_calls = sample(sis_raytrace(c, 1.0), [c], [r], r, batch_size=100)
    assert max(len(x) for x in batched_calls) <= 100
    for field in CenterHoles._fields:
        assert np.array_equal(
            to_np(getattr(batched, field)), to_np(getattr(holes, field))
        ), field


def test_holes_sampled_together_match_holes_sampled_alone():
    """One call per round covers every hole; no hole's samples leak into another's."""
    lens = sis_raytrace((0.3, -0.2), 1.0)
    pairs = [((0.3, -0.2), 0.01), ((1.5, 1.0), 0.02)]
    both, _ = sample(lens, [c for c, _ in pairs], [r for _, r in pairs], 0.01)
    off = to_np(both.offsets)
    for h, (c, r) in enumerate(pairs):
        alone, _ = sample(lens, [c], [r], 0.01)
        for field in ("angle", "lens", "source"):
            assert np.array_equal(
                to_np(getattr(both, field))[off[h] : off[h + 1]],
                to_np(getattr(alone, field)),
            ), field
        for field in ("growth", "growth_err", "pseudo_caustic"):
            assert np.array_equal(
                to_np(getattr(both, field))[h], to_np(getattr(alone, field))[0]
            ), field


# ---------------------------------------------------------------------------
# Holes on the mesh
# ---------------------------------------------------------------------------


def sis_lens(c, b):
    """A lens-like object: the SIS of ``sis_raytrace`` and its Jacobian."""

    def jacobian(x, y):
        dx, dy = x - c[0], y - c[1]
        r = backend.sqrt(dx * dx + dy * dy)
        k = b / r**3
        a00 = 1.0 - b / r + k * dx * dx
        a01 = k * dx * dy
        a11 = 1.0 - b / r + k * dy * dy
        return backend.stack(
            (backend.stack((a00, a01), dim=-1), backend.stack((a01, a11), dim=-1)),
            dim=-2,
        )

    return SimpleNamespace(raytrace=sis_raytrace(c, b), jacobian_lens_equation=jacobian)


def recording_lens(lens):
    """``lens`` with every point each method is called on recorded."""
    calls = {"raytrace": [], "jacobian": []}

    def raytrace(x, y):
        calls["raytrace"].append(np.stack([to_np(x), to_np(y)], axis=-1))
        return lens.raytrace(x, y)

    def jacobian(x, y):
        calls["jacobian"].append(np.stack([to_np(x), to_np(y)], axis=-1))
        return lens.jacobian_lens_equation(x, y)

    return SimpleNamespace(raytrace=raytrace, jacobian_lens_equation=jacobian), calls


def _same(a, b):
    """Equal field by field, arrays bit for bit, NaN matching NaN."""
    if hasattr(a, "_fields"):
        return all(_same(getattr(a, f), getattr(b, f)) for f in a._fields)
    if hasattr(a, "shape"):
        a, b = to_np(a), to_np(b)
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
        )
    return a == b


SIS_C = (0.3001, -0.2003)  # off every lattice point of BUILD
BUILD = dict(fov=4.0, init_res=8, min_img_sep=0.02)


def test_a_build_without_centers_stores_empty_holes():
    lens = sis_lens(SIS_C, 1.0)
    mesh = build_lens_mesh(lens.raytrace, lens.jacobian_lens_equation, **BUILD)
    assert mesh.holes.centers.shape[0] == 0
    assert to_np(mesh.holes.offsets).tolist() == [0]


def test_a_build_stores_the_merged_and_sampled_holes():
    lens = sis_lens(SIS_C, 1.0)
    centers = [SIS_C, SIS_C, (1.5, 1.5)]
    mesh = to_mesh(
        build_lens_mesh(
            lens.raytrace, lens.jacobian_lens_equation, **BUILD, centers=centers
        )
    )
    want = sample_holes(
        make_sampler(lens.raytrace, None, None),
        *merge_centers(centers, mesh.min_img_sep),
        mesh.min_img_sep,
        None,
    )
    assert mesh.holes.centers.shape[0] == 2
    assert to_np(mesh.holes.radius).tolist() == [mesh.min_img_sep] * 2
    assert _same(mesh.holes, want)


def test_centers_change_nothing_but_the_holes_even_outside_the_fov():
    lens = sis_lens(SIS_C, 1.0)
    plain = build_lens_mesh(lens.raytrace, lens.jacobian_lens_equation, **BUILD)
    holed = build_lens_mesh(
        lens.raytrace, lens.jacobian_lens_equation, **BUILD, centers=[SIS_C, (5.0, 5.0)]
    )
    assert holed.holes.centers.shape[0] == 2
    for name in LensMesh._fields:
        if name != "holes":
            assert _same(getattr(holed, name), getattr(plain, name)), name


def test_holes_cost_raytraces_on_their_circles_only_and_no_jacobian():
    plain_lens, plain = recording_lens(sis_lens(SIS_C, 1.0))
    holed_lens, holed = recording_lens(sis_lens(SIS_C, 1.0))
    build_lens_mesh(plain_lens.raytrace, plain_lens.jacobian_lens_equation, **BUILD)
    mesh = build_lens_mesh(
        holed_lens.raytrace, holed_lens.jacobian_lens_equation, **BUILD, centers=[SIS_C]
    )
    base = np.concatenate(plain["raytrace"])
    extra = np.concatenate(holed["raytrace"])
    n_samples = int(to_np(mesh.holes.offsets)[-1])
    assert len(extra) - len(base) == n_samples + 4 * HOLE_GROWTH_SAMPLES
    added = extra[
        ~np.isin(extra[:, 0] + 1j * extra[:, 1], base[:, 0] + 1j * base[:, 1])
    ]
    d = np.hypot(*(added - np.array(SIS_C)).T)
    radii = mesh.min_img_sep / 4.0 ** np.arange(4)
    assert np.isclose(d[:, None], radii, rtol=0, atol=1e-12).any(axis=1).all()
    assert sum(map(len, holed["jacobian"])) == sum(map(len, plain["jacobian"]))


def test_holes_land_on_the_mesh_device(device):
    lens = sis_lens(SIS_C, 1.0)
    mesh = build_lens_mesh(
        lens.raytrace,
        lens.jacobian_lens_equation,
        **BUILD,
        centers=[SIS_C],
        device=device,
    )
    for field in CenterHoles._fields:
        assert mesh_backend.device(getattr(mesh.holes, field)) == mesh_backend.device(
            mesh.vertices_lens
        ), field


def test_the_caustic_area_of_an_sis_is_the_disk_its_pseudo_caustic_bounds():
    """The hole curve of radius ``rho`` maps to the circle of radius ``1 - rho``."""
    lens_, _ = numpy_lens(sis_map, sis_map_jacobian)
    mesh, curves = build_closed_lens_mesh(
        lens_.raytrace,
        lens_.jacobian_lens_equation,
        6.0,
        16,
        0.02,
        centers=[(0.0, 0.0)],
    )
    rho = float(to_np(to_mesh(mesh).holes.radius)[0])
    area = caustic_area(curves)
    assert isinstance(area, float)
    assert area == pytest.approx(math.pi * (1.0 - rho) ** 2, rel=1e-4)


def test_an_open_curve_leaves_the_caustic_area_unknown():
    lens_, _ = numpy_lens(sie_like, sie_like_jacobian)
    mesh = build_lens_mesh(lens_.raytrace, lens_.jacobian_lens_equation, 2.0, 8, 0.05)
    curves = critical_curves_and_caustics(mesh)
    assert not to_np(curves.closed).all()
    assert math.isnan(caustic_area(curves))


def test_a_lens_without_caustics_bounds_no_area():
    lens_, _ = numpy_lens(affine, affine_jacobian)
    mesh = build_lens_mesh(lens_.raytrace, lens_.jacobian_lens_equation, 4.0, 4, 0.25)
    assert caustic_area(critical_curves_and_caustics(mesh)) == 0.0
