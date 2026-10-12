"""Critical curves and caustics traced through the band and joined at holes, and the area they bound."""

import itertools
import math
import numpy as np
import pytest

from types import SimpleNamespace
from scipy.spatial import cKDTree
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh
from caustics.lenses.func.adaptive import (
    CenterHoles,
    CriticalBand,
    CriticalCurvesAndCaustics,
    build_closed_lens_mesh,
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
from caustics.lenses.func.adaptive.holes import empty_holes

from adaptive_maps import (
    affine,
    affine_jacobian,
    broken_where,
    build,
    f64,
    numpy_lens,
    row_fold,
    row_fold_jacobian,
    sie_like,
    sie_like_jacobian,
    sis,
    to_np,
    user_build,
)


def _band(samples, lens, det, source=None):
    """A hand-built `CriticalBand`; ``source`` defaults to ``lens``."""
    samples = np.asarray(samples, dtype=np.int64)
    return CriticalBand(
        leaves=mesh_backend.as_array(
            np.arange(samples.shape[0]), dtype=mesh_backend.int64
        ),
        samples=mesh_backend.as_array(samples, dtype=mesh_backend.int64),
        lens=f64(lens),
        source=f64(lens if source is None else source),
        det=f64(det),
        sigma_min=f64(np.ones(np.shape(det))),
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
        centers=f64([[0.0, 0.0]]),
        radius=f64([radius]),
        offsets=mesh_backend.as_array(np.array([0, n]), dtype=mesh_backend.int64),
        angle=f64(angle),
        lens=f64(lens),
        source=f64(lens + np.array(shift)),
        growth=f64([0.0]),
        growth_err=f64([0.0]),
        pseudo_caustic=mesh_backend.as_array(np.array([True]), dtype=mesh_backend.bool),
    )


def _traced(*curves):
    """Hand-built ``CriticalCurvesAndCaustics`` from ``(points, closed)`` pairs.

    ``source`` is ``2 * lens``.
    """
    pts = np.concatenate([np.asarray(p, dtype=np.float64) for p, _ in curves])
    offsets = np.cumsum([0] + [len(p) for p, _ in curves])
    return CriticalCurvesAndCaustics(
        lens=f64(pts),
        source=f64(2.0 * pts),
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
        centers=f64([_A, _B]),
        radius=f64([0.1, 0.1]),
        offsets=mesh_backend.as_array(
            np.array([0, n, 2 * n]), dtype=mesh_backend.int64
        ),
        angle=f64(np.concatenate([angle, angle])),
        lens=f64(lens),
        source=f64(lens + shift),
        growth=f64([0.0, 0.0]),
        growth_err=f64([0.0, 0.0]),
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
    mesh, _ = user_build(sie_like, sie_like_jacobian, 4.0, 16, min_img_sep)
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
    mesh, _ = build(sie_like, sie_like_jacobian, 4.0, 16, 1e-2)
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
    mesh, _ = user_build(sie_like, sie_like_jacobian, 2.0, 8, 1e-2)
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
    This also pins the rule that coincident consecutive points --
    which tracing a curve through lattice points produces -- are kept, not
    removed.
    """
    mesh, _ = user_build(row_fold, row_fold_jacobian, 4.0, 8, 2e-2)
    curves = _curves(mesh)
    assert len(curves) == 1
    lens, _, closed = curves[0]
    assert not closed
    assert np.abs(lens[:, 1] - 0.5).max() < 1e-12
    assert lens[0, 0] == 2.0 and lens[-1, 0] == -2.0
    assert (np.diff(lens, axis=0) == 0).all(axis=1).any()


def fold(p):
    """``(x, 0.6 y + y**2)``: ``det A = 0.6 + 2y``, zero on ``y = -0.3`` alone."""
    return np.stack([p[:, 0], 0.6 * p[:, 1] + p[:, 1] ** 2], axis=-1)


def fold_jacobian(p):
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 0.6 + 2.0 * p[:, 1]
    return J


def test_a_curve_ends_where_the_lens_turns_nonfinite():
    """The fold ``y = -0.3`` runs into a region where the lens is NaN.

    No band leaf has a non-finite sample, so the curve is open: one end on the
    fov boundary at ``x = -2``, the other where the band stops, within a leaf
    edge of ``x = 1``.
    """
    min_img_sep = 0.05
    fn, jac = broken_where(fold, fold_jacobian, lambda p: p[:, 0] > 1.0)
    mesh, _ = user_build(fn, jac, 4.0, 4, min_img_sep)
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
    mesh, _ = build(*sis(TWO_SIS), **TWO_SIS_BUILD, centers=TWO_SIS)
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
        sis(TWO_SIS)[0],
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
    mesh, _ = build(*sis(shifted), **TWO_SIS_BUILD, centers=shifted)
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
    mesh, _ = build(*sis(), 4.0, 8, 1e-2, centers=[(0.0, 0.0)])
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
    mesh, _ = user_build(
        *sis(WITH_COMPANION, WITH_COMPANION_B), **TWO_SIS_BUILD, centers=WITH_COMPANION
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
        sis(WITH_COMPANION, WITH_COMPANION_B)[0],
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
    mesh, _ = build(*sis(TWO_SIS), **TWO_SIS_BUILD, centers=centers)
    assert _hole_to_hole_steps(mesh) > 0
    _assert_closed_and_chord_free(mesh, critical_curves_and_caustics(mesh))


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
    field = f64([-1.0, 3.0])
    a = f64([[0.0, 0.0], [4.0, 0.0]])
    b = f64([[0.0, 1.0], [0.0, 5.0]])
    za, zb = edge_zeros(edges, field, (a, b))
    assert to_np(za).tolist() == [[1.0, 0.0]]
    assert to_np(zb).tolist() == [[0.0, 2.0]]


def test_the_caustic_area_of_an_sis_is_the_disk_its_pseudo_caustic_bounds():
    """The hole curve of radius ``rho`` maps to the circle of radius ``1 - rho``."""
    lens_ = numpy_lens(*sis())
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
    mesh, _ = user_build(sie_like, sie_like_jacobian, 2.0, 8, 0.05)
    curves = critical_curves_and_caustics(mesh)
    assert not to_np(curves.closed).all()
    assert math.isnan(caustic_area(curves))


def test_a_lens_without_caustics_bounds_no_area():
    mesh, _ = user_build(affine, affine_jacobian, 4.0, 4, 0.25)
    assert caustic_area(critical_curves_and_caustics(mesh)) == 0.0
