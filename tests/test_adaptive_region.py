"""Regions between two total magnifications: the source mesh, its contours and area, the closed build."""

import math

import numpy as np
import pytest

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, ExternalShear, Point, PseudoJaffe, SinglePlane
from caustics.lenses.func.adaptive import (
    build_closed_lens_mesh,
    build_closed_magnification_region,
    build_lens_mesh,
    critical_curves_and_caustics,
    magnification_region,
    total_magnification,
)
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh
from caustics.lenses.func.adaptive.geometry import ROOT_SHAPES
from caustics.lenses.func.adaptive.lattice import (
    lattice_fov,
    lattice_h0,
    lattice_init_res,
    lattice_xy,
    make_lattice,
    midpoint_ij,
)
from caustics.lenses.func.adaptive.curves import (
    _critical_curves_and_caustics,
    pseudo_caustics,
)
from caustics.lenses.func.adaptive.lens_mesh import (
    _extend_lens_mesh,
    _fov_boundary,
    inside_fov_image,
)
from caustics.lenses.func.adaptive.magnification import (
    _total_magnification,
    leaf_tables,
)
from caustics.lenses.func.adaptive import region as region_module
from caustics.lenses.func.adaptive.region import (
    _caustic_marks,
    _cell_validity,
    _extend_source_mesh,
    _leaf_valid,
    _region,
    _sampler,
    _source_lattice,
    _source_mesh,
    _source_split,
)
from adaptive_maps import (
    AFFINE,
    affine,
    affine_jacobian,
    assert_same,
    broken_where,
    build,
    f64,
    i64,
    numpy_lens,
    point_jacobian,
    point_raytrace,
    sie_like,
    sie_like_jacobian,
    sis_jacobian,
    sis_raytrace,
    to_np,
)


def point_radius(mu):
    """Source radius where a unit point mass has total magnification ``mu``."""
    return math.sqrt(2.0 * mu / math.sqrt(mu * mu - 1.0) - 2.0)


@pytest.fixture(scope="module")
def point():
    """A unit point mass at fov 6: its mesh and curves, on the mesh backend."""
    mesh, _ = build(
        point_raytrace, point_jacobian, fov=6.0, init_res=16, min_img_sep=0.02
    )
    return mesh, _critical_curves_and_caustics(mesh)


@pytest.fixture(scope="module")
def sis():
    """A unit SIS at fov 6, without a hole: its mesh and curves, on the mesh backend."""
    mesh, _ = build(sis_raytrace, sis_jacobian, fov=6.0, init_res=16, min_img_sep=0.02)
    return mesh, _critical_curves_and_caustics(mesh)


# Four level-0 cells of side 1 per axis, finest level 2; cell (0, 0) is invalid.
LAT = make_lattice(4.0, 0.0, 0.0, 4, 3)


_valid = np.ones((4, 4), dtype=bool)


_valid[0, 0] = False


VALID = mesh_backend.as_array(_valid)


NO_MARKS = [i64([]), i64([])]


STRADDLE = [2.0, 2.0, 2.0, 4.0, 2.0, 2.0]


BELOW = [2.0] * 6


ABOVE = [5.0] * 6


def _cell_triangle(i, j):
    """The first level-0 triangle of cell ``(i, j)`` of ``LAT``, ``(1, 3, 2)``."""
    return i64([(np.asarray(ROOT_SHAPES[0]) + (i, j)) * (1 << LAT.level)])


def _split(ij, mu, size, marks=NO_MARKS, mu_max=None):
    """``_source_split`` at ``mu_min = 3`` on one level-0 row whose six samples share ``size``."""
    values6 = f64([np.stack((np.asarray(mu), np.full(6, size)), axis=-1)])
    split = _source_split(LAT, VALID, marks, 3.0, mu_max)
    return bool(to_np(split(ij, values6, i64([0]), i64([0])))[0])


def test_a_triangle_the_threshold_crosses_splits_while_longer_than_the_leaf_images():
    """A level-0 triangle of ``LAT`` has a longest edge of sqrt(2)."""
    assert _split(_cell_triangle(1, 1), STRADDLE, size=0.5)
    assert not _split(_cell_triangle(1, 1), STRADDLE, size=2.0)
    assert not _split(_cell_triangle(1, 1), BELOW, size=0.5)


def test_a_triangle_in_an_invalid_cell_never_splits():
    assert not _split(_cell_triangle(0, 0), STRADDLE, size=0.5)
    assert not _split(_cell_triangle(0, 0), BELOW, size=2.0, marks=[i64([0]), i64([])])


def test_a_marked_cell_splits_unless_every_sample_exceeds_the_upper_threshold():
    marks = [i64([1 * 4 + 1]), i64([])]  # cell (1, 1) at level 0
    assert _split(_cell_triangle(1, 1), BELOW, size=2.0, marks=marks)
    assert not _split(_cell_triangle(1, 1), ABOVE, size=2.0, marks=marks)
    assert _split(_cell_triangle(1, 1), ABOVE, size=2.0, marks=marks, mu_max=10.0)
    assert not _split(
        _cell_triangle(1, 1), [11.0] * 6, size=2.0, marks=marks, mu_max=10.0
    )


def test_mu_max_crossings_split_too():
    samples = [9.0, 9.0, 9.0, 11.0, 9.0, 9.0]
    assert _split(_cell_triangle(1, 1), samples, size=0.5, mu_max=10.0)
    assert not _split(_cell_triangle(1, 1), samples, size=0.5)


def test_the_source_lattice_spans_the_fov_boundary_image_down_to_the_smallest_leaf_image(
    point,
):
    mesh = point[0]
    tables = leaf_tables(mesh)
    lat, max_level = _source_lattice(mesh, tables)
    assert lattice_init_res(lat) == lattice_init_res(mesh.lattice)
    poly = to_np(mesh.vertices_source[_fov_boundary(mesh)])
    corner = to_np(lattice_xy(lat, i64([0, 0])))
    assert (poly >= corner - 1e-12).all() and (
        poly <= corner + lattice_fov(lat) + 1e-12
    ).all()
    size = to_np(tables.size)
    smallest = size[np.isfinite(size) & (size > 0)].min()
    edge = math.sqrt(2.0) * lattice_h0(lat)
    assert edge / 2**max_level <= smallest < edge / 2 ** (max_level - 1)


def test_every_point_of_a_valid_cell_lies_inside_the_fov_boundary_image(sis):
    mesh = sis[0]
    lat, _ = _source_lattice(mesh, leaf_tables(mesh))
    valid = to_np(_cell_validity(mesh, lat))
    assert 0 < valid.sum() < valid.size
    corner, h0 = to_np(lattice_xy(lat, i64([0, 0]))), lattice_h0(lat)
    t = (np.arange(6) + 0.5) / 6
    u = np.stack(np.meshgrid(t, t, indexing="ij"), axis=-1).reshape(-1, 2)
    points = corner + h0 * (np.argwhere(valid)[:, None, :] + u[None])
    assert to_np(inside_fov_image(mesh, f64(points.reshape(-1, 2)))).all()


def test_a_cell_the_fov_boundary_image_runs_through_is_invalid(sis):
    mesh = sis[0]
    lat, _ = _source_lattice(mesh, leaf_tables(mesh))
    valid = to_np(_cell_validity(mesh, lat))
    corner, h0 = to_np(lattice_xy(lat, i64([0, 0]))), lattice_h0(lat)
    poly = to_np(mesh.vertices_source[_fov_boundary(mesh)])
    cell = np.clip(np.floor((poly - corner) / h0).astype(int), 0, valid.shape[0] - 1)
    assert not valid[cell[:, 0], cell[:, 1]].any()


def test_every_caustic_segment_marks_its_cell_at_every_level_coarser_than_it():
    mesh, _ = build(sie_like, sie_like_jacobian, fov=4.0, init_res=8, min_img_sep=0.05)
    curves = _critical_curves_and_caustics(mesh)
    assert to_np(curves.closed).all()
    lat, max_level = _source_lattice(mesh, leaf_tables(mesh))
    marks = [set(to_np(m).tolist()) for m in _caustic_marks(curves, lat, max_level)]
    corner, h0, r = (
        to_np(lattice_xy(lat, i64([0, 0]))),
        lattice_h0(lat),
        lattice_init_res(lat),
    )
    source, offsets = to_np(curves.source), to_np(curves.offsets)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        p = source[lo:hi]
        q = np.roll(p, -1, axis=0)
        mid, length = 0.5 * (p + q), np.linalg.norm(q - p, axis=1)
        for level in range(max_level):
            sel = length < math.sqrt(2.0) * h0 / 2**level
            ij = np.floor((mid[sel] - corner) / (h0 / 2**level)).astype(int)
            assert set((ij[:, 0] * (r << level) + ij[:, 1]).tolist()) <= marks[level]


def test_pseudo_caustics_mark_nothing():
    lens_, _ = numpy_lens(sis_raytrace, sis_jacobian)
    mesh = to_mesh(
        build_lens_mesh(
            lens_.raytrace,
            lens_.jacobian_lens_equation,
            6.0,
            16,
            0.02,
            centers=[(0.0, 0.0)],
        )
    )
    lat, max_level = _source_lattice(mesh, leaf_tables(mesh))
    marks = _caustic_marks(pseudo_caustics(mesh.holes), lat, max_level)
    assert all(m.shape[0] == 0 for m in marks)


def test_every_source_leaf_below_the_finest_level_passes_the_split(point):
    mesh, curves = point
    src = _source_mesh(mesh, curves, 3.0, None)
    rows = i64(np.flatnonzero(to_np(src.origin_level) < src.max_level))
    ij = src.vertices_ij[src.origin_leaves[rows]]
    six = mesh_backend.concatenate((ij, midpoint_ij(ij)), dim=1)
    sample = _sampler(mesh, leaf_tables(mesh))
    values6 = sample(lattice_xy(src.lattice, six.reshape(-1, 2))).reshape(-1, 6, 2)
    marks = _caustic_marks(curves, src.lattice, src.max_level)
    split = _source_split(src.lattice, src.cell_valid, marks, 3.0, None)
    assert not to_np(
        split(ij, values6, src.origin_cls[rows], src.origin_level[rows])
    ).any()


def test_source_vertices_carry_mu_tot_and_cost_no_lens_call():
    mesh, calls = build(
        point_raytrace, point_jacobian, fov=6.0, init_res=16, min_img_sep=0.02
    )
    curves = _critical_curves_and_caustics(mesh)
    made = (len(calls.raytrace), len(calls.jacobian))
    src = _source_mesh(mesh, curves, 3.0, None)
    assert (len(calls.raytrace), len(calls.jacobian)) == made
    mu, size = _total_magnification(mesh, leaf_tables(mesh), src.vertices)
    np.testing.assert_array_equal(to_np(src.values[:, 0]), to_np(mu))
    np.testing.assert_array_equal(to_np(src.values[:, 1]), to_np(size))


def _user_mesh(fn, jac, fov=6.0, init_res=16, min_img_sep=0.02, **kw):
    lens_, _ = numpy_lens(fn, jac)
    return build_lens_mesh(
        lens_.raytrace, lens_.jacobian_lens_equation, fov, init_res, min_img_sep, **kw
    )


def _signed_areas(region):
    points, offsets = to_np(region.source), to_np(region.offsets)
    areas = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        p = points[lo:hi]
        q = np.roll(p, -1, axis=0)
        areas.append(0.5 * np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]))
    return np.array(areas)


@pytest.mark.parametrize(
    "fn, jac, mu_min, radius",
    [
        (sis_raytrace, sis_jacobian, 3.0, 2.0 / 3.0),
        (sis_raytrace, sis_jacobian, 2.5, 0.8),
        (point_raytrace, point_jacobian, 3.0, point_radius(3.0)),
        (point_raytrace, point_jacobian, 1.5, point_radius(1.5)),
    ],
    ids=["sis-inside", "sis-across-pseudo-caustic", "point-3", "point-1.5"],
)
def test_the_region_of_an_axisymmetric_lens_is_its_analytic_disk(
    fn, jac, mu_min, radius
):
    mesh = _user_mesh(fn, jac)
    region = magnification_region(mesh, critical_curves_and_caustics(mesh), mu_min)
    assert to_np(region.closed).tolist() == [True]
    assert region.area == pytest.approx(math.pi * radius**2, rel=1e-2)
    assert np.abs(np.linalg.norm(to_np(region.source), axis=1) - radius).max() < 2e-2
    assert (region.mu_min, region.mu_max) == (mu_min, math.inf)
    assert to_np(region.level).tolist() == [mu_min]


def test_mu_max_cuts_a_hole_whose_contour_runs_clockwise():
    mesh = _user_mesh(point_raytrace, point_jacobian)
    region = magnification_region(mesh, critical_curves_and_caustics(mesh), 1.5, 3.0)
    want = math.pi * (point_radius(1.5) ** 2 - point_radius(3.0) ** 2)
    assert region.area == pytest.approx(want, rel=2e-2)
    assert to_np(region.level).tolist() == [1.5, 3.0]
    outer, hole = _signed_areas(region)
    assert outer > 0 > hole
    assert region.area == pytest.approx(outer + hole, rel=1e-12)


def test_a_region_reaching_past_the_valid_cells_has_nan_area_and_open_contours():
    """At fov 6 the SIS boundary's image has a radius of about 2, the disk's own."""
    mesh = _user_mesh(sis_raytrace, sis_jacobian)
    region = magnification_region(mesh, critical_curves_and_caustics(mesh), 1.5)
    assert math.isnan(region.area)
    assert not to_np(region.closed).all()


def test_a_threshold_below_the_far_field_magnification_bounds_nothing():
    mesh = _user_mesh(sis_raytrace, sis_jacobian)
    region = magnification_region(mesh, critical_curves_and_caustics(mesh), 0.5)
    assert math.isnan(region.area)
    assert to_np(region.offsets).tolist() == [0]


def test_a_lens_of_constant_magnification_gives_an_empty_or_an_unbounded_region():
    mesh = _user_mesh(affine, affine_jacobian, fov=4.0, init_res=4, min_img_sep=0.25)
    curves = critical_curves_and_caustics(mesh)
    mu = 1.0 / abs(np.linalg.det(AFFINE))
    above = magnification_region(mesh, curves, 1.1 * mu)
    assert above.area == 0.0 and to_np(above.offsets).tolist() == [0]
    assert math.isnan(magnification_region(mesh, curves, 0.9 * mu).area)


def test_a_region_inside_cells_the_broken_boundary_leaves_invalid_has_nan_area():
    """The lens is not finite on the fov's right side, so the boundary's image has a gap."""
    fn, jac = broken_where(sie_like, sie_like_jacobian, lambda p: p[:, 0] > 1.8)
    mesh = _user_mesh(fn, jac, fov=4.0, init_res=8, min_img_sep=0.05)
    assert math.isnan(
        magnification_region(mesh, critical_curves_and_caustics(mesh), 3.0).area
    )


def test_a_region_crossing_an_invalid_leaf_between_its_vertices_has_nan_area(point):
    """A caustic's jump can straddle both thresholds along one edge of an invalid leaf."""
    mesh, curves = point
    src = _source_mesh(mesh, curves, 3.0, 10.0)
    assert math.isfinite(_region(src, 3.0, 10.0).area)
    valid, leaves = to_np(_leaf_valid(src)), to_np(src.leaves)
    in_valid = np.zeros(src.vertices.shape[0], dtype=bool)
    in_valid[leaves[valid].ravel()] = True
    a, b = next(
        (tri[i], tri[(i + 1) % 3])
        for tri in leaves[~valid]
        for i in range(3)
        if in_valid[tri[i]] and in_valid[tri[(i + 1) % 3]]
    )
    values = to_np(src.values).copy()
    values[a, 0], values[b, 0] = 0.0, 100.0
    region = _region(src._replace(values=f64(values)), 3.0, 10.0)
    assert not to_np(region.closed).all()
    assert math.isnan(region.area)


def test_a_strip_by_a_caustic_in_invalid_cells_leaves_the_area_unknown():
    """At fov 4 the cored sphere's radial caustic, radius about 0.66, lies in the invalid cells
    round the small image of the fov boundary, which are never refined, so no vertex sees the
    strip of mu_tot >= 30 along its bright side."""
    mesh = _user_mesh(
        sie_like, sie_like_jacobian, fov=4.0, init_res=8, min_img_sep=0.05
    )
    region = magnification_region(mesh, critical_curves_and_caustics(mesh), 30.0)
    assert math.isnan(region.area)


COSMO = FlatLambdaCDM(name="cosmo")


def _plane(main, g1, g2):
    shear = ExternalShear(
        name="shear",
        cosmology=COSMO,
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        gamma_1=g1,
        gamma_2=g2,
    )
    return SinglePlane(
        name="plane", cosmology=COSMO, lenses=(main, shear), z_l=0.5, z_s=1.5
    )


def _fold_lens(name):
    """A lens with fold caustics: the lens, its centers and a fov its curves close in."""
    if name == "sie+shear":
        main = SIE(
            name="sie",
            cosmology=COSMO,
            z_l=0.5,
            z_s=1.5,
            x0=0.0,
            y0=0.0,
            q=0.6,
            phi=0.4,
            Rein=1.0,
        )
        return _plane(main, 0.05, -0.03), [(0.0, 0.0)], 5.0
    if name == "pseudo-jaffe+shear":
        main = PseudoJaffe(
            name="pj",
            cosmology=COSMO,
            z_l=0.5,
            z_s=1.5,
            x0=0.0,
            y0=0.0,
            mass=1e12,
            Rc=0.1,
            Rs=5.0,
        )
        return _plane(main, 0.03, 0.06), None, 5.0
    main = Point(name="pt", cosmology=COSMO, z_l=0.5, z_s=1.5, x0=0.0, y0=0.0, Rein=1.0)
    return _plane(main, 0.2, 0.1), None, 6.0


@pytest.fixture(
    scope="module", params=["sie+shear", "pseudo-jaffe+shear", "point+shear"]
)
def fold(request):
    lens, centers, fov = _fold_lens(request.param)
    return build_closed_lens_mesh(
        lens.raytrace, lens.jacobian_lens_equation, fov, 16, 0.02, centers=centers
    )


@pytest.mark.filterwarnings("ignore:Lens model")
@pytest.mark.parametrize("mu_min, mu_max", [(3.0, None), (30.0, None), (3.0, 10.0)])
def test_a_region_s_area_matches_counting_mu_tot_on_a_fine_grid(fold, mu_min, mu_max):
    """At mu_min = 30 the region is strips by the folds that the caustic marks find."""
    mesh, curves = fold
    region = magnification_region(mesh, curves, mu_min, mu_max)
    assert to_np(region.closed).all() and math.isfinite(region.area)
    points, offsets = to_np(region.source), to_np(region.offsets)
    lo = points.min(axis=0) - 0.05
    n = 600
    dx = float((points.max(axis=0) + 0.05 - lo).max()) / n
    axis = (np.arange(n) + 0.5) * dx
    gx, gy = np.meshgrid(lo[0] + axis, lo[1] + axis, indexing="ij")
    mu = to_np(total_magnification(gx, gy, mesh))
    upper = math.inf if mu_max is None else mu_max
    counted = np.count_nonzero((mu >= mu_min) & (mu <= upper)) * dx * dx
    perimeter = sum(
        np.linalg.norm(np.roll(points[a:b], -1, axis=0) - points[a:b], axis=1).sum()
        for a, b in zip(offsets[:-1], offsets[1:])
    )
    assert abs(region.area - counted) <= perimeter * dx


@pytest.fixture(scope="module")
def grown():
    """An SIS source mesh for mu_min = 1.5 at fov 6, and its extension with the lens mesh to fov 9."""
    lens_, _ = numpy_lens(sis_raytrace, sis_jacobian)
    mesh = to_mesh(
        build_lens_mesh(lens_.raytrace, lens_.jacobian_lens_equation, 6.0, 16, 0.02)
    )
    src = _source_mesh(mesh, _critical_curves_and_caustics(mesh), 1.5, None)
    bigger = _extend_lens_mesh(
        mesh,
        lens_.raytrace,
        lens_.jacobian_lens_equation,
        9.0,
        device=None,
        batch_size=None,
    )
    return src, bigger


def test_extending_a_source_mesh_samples_no_vertex_it_holds_and_keeps_every_one(
    grown, monkeypatch
):
    src, bigger = grown
    seen = []
    real = region_module._sampler

    def recording(mesh, tables):
        sample = real(mesh, tables)

        def record(xy):
            seen.append(to_np(xy))
            return sample(xy)

        return record

    monkeypatch.setattr(region_module, "_sampler", recording)
    before = to_np(src.values).copy()
    out = _extend_source_mesh(
        src, bigger, _critical_curves_and_caustics(bigger), 1.5, None
    )
    old = {tuple(p) for p in to_np(src.vertices).tolist()}
    assert seen and not any(tuple(p) in old for p in np.concatenate(seen).tolist())
    assert old <= {tuple(p) for p in to_np(out.vertices).tolist()}
    np.testing.assert_array_equal(to_np(src.values), before)


def test_an_extended_source_mesh_bounds_the_region_the_grown_lens_mesh_holds(grown):
    src, bigger = grown
    assert math.isnan(_region(src, 1.5, None).area)
    region = _region(
        _extend_source_mesh(
            src, bigger, _critical_curves_and_caustics(bigger), 1.5, None
        ),
        1.5,
        None,
    )
    assert to_np(region.closed).all()
    assert region.area == pytest.approx(4.0 * math.pi, rel=1e-2)


def test_the_closed_build_grows_the_fov_until_the_region_is_bounded():
    lens_, _ = numpy_lens(sis_raytrace, sis_jacobian)
    mesh, curves, region = build_closed_magnification_region(
        lens_.raytrace,
        lens_.jacobian_lens_equation,
        6.0,
        16,
        0.02,
        1.5,
        centers=[(0.0, 0.0)],
    )
    assert lattice_fov(to_mesh(mesh).lattice) > 6.0
    assert to_np(region.closed).all()
    assert region.area == pytest.approx(4.0 * math.pi, rel=1e-2)


def test_a_bounded_region_costs_no_extension_beyond_the_closed_lens_mesh():
    lens_, _ = numpy_lens(sis_raytrace, sis_jacobian)
    args = (lens_.raytrace, lens_.jacobian_lens_equation, 6.0, 16, 0.02)
    mesh, curves, region = build_closed_magnification_region(
        *args, 3.0, centers=[(0.0, 0.0)]
    )
    closed_mesh, closed_curves = build_closed_lens_mesh(*args, centers=[(0.0, 0.0)])
    assert_same(mesh, closed_mesh)
    assert_same(curves, closed_curves)
    assert_same(region, magnification_region(closed_mesh, closed_curves, 3.0))


def test_the_closed_build_warns_when_no_fov_bounds_the_region():
    lens_, _ = numpy_lens(sis_raytrace, sis_jacobian)
    with pytest.warns(
        UserWarning, match="still reaches the edge of the valid source domain"
    ):
        _, _, region = build_closed_magnification_region(
            lens_.raytrace,
            lens_.jacobian_lens_equation,
            6.0,
            16,
            0.02,
            0.5,
            max_iters=1,
            centers=[(0.0, 0.0)],
        )
    assert math.isnan(region.area)
