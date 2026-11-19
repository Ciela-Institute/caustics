"""The lens mesh: its sampler, build, extension and closed build."""

import pickle
import warnings
import weakref
import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE
from caustics.lenses.func.adaptive.band import (
    MEASURE_OFFSET,
    RESOLVED_SPREAD,
    CriticalBand,
    in_band,
)
from caustics.lenses.func.adaptive.criterion import (
    LEAF_APPROX_PARITY_UNRESOLVED,
    LEAF_CONVERGED,
    LEAF_CONVERGENCE_FAILED,
    LEAF_JACOBIAN_PARITY_UNRESOLVED,
    LEAF_RAYTRACE_NONFINITE,
    affine_error,
    child_shape_matrices,
    fitted_jacobian,
    lens_status,
    parity_from_children,
)
from caustics.lenses.func.adaptive.geometry import (
    CORNER_SECTOR,
    build_device,
    child_matrix_tables,
    to_device,
)
from caustics.lenses.func.adaptive.index import index_hits
from caustics.lenses.func.adaptive.lattice import (
    lattice_key,
    lattice_xy,
)
from caustics.lenses.func.adaptive.curves import critical_curves_and_caustics
from caustics.lenses.func.adaptive import lens_mesh as lens_mesh_module
from caustics.lenses.func.adaptive.lens_mesh import (
    _fov_boundary,
    _freeze,
    _seed,
    build_closed_lens_mesh,
    build_lens_mesh,
    extend_lens_mesh,
    inside_fov,
    inside_fov_image,
    leaf_grow,
    make_sampler,
)

from adaptive_maps import (
    AFFINE,
    affine,
    affine_jacobian,
    assert_same,
    broken_where,
    build,
    collapse,
    collapse_jacobian,
    f64,
    fresh_equivalent,
    holders_mean,
    i64,
    localised_fold,
    localised_fold_jacobian,
    numpy_lens,
    point_mass,
    row_fold,
    row_fold_jacobian,
    sie_like,
    sie_like_jacobian,
    sie_with_subhalo,
    sis,
    to_np,
    user_build,
)


@pytest.fixture(scope="module")
def sie_mesh():
    lens = SIE(
        name="sie",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        q=0.4,
        phi=np.pi / 5,
        Rein=1.0,
        s=1e-3,
    )
    return to_mesh(build_lens_mesh(lens.raytrace, 5.0, 32, 1e-2))


def test_the_sampler_hands_the_raytrace_float64_points_on_the_device(device):
    seen = []

    def raytrace(x, y):
        seen.append((x.dtype, backend.device(x)))
        return 0.5 * x, 0.5 * y

    xy = f64([[0.0, 1.0], [2.0, -3.0]])
    out = make_sampler(raytrace, device)(xy)
    want = backend.device(backend.as_array([0.0], device=device))
    ((rd, rdev),) = seen
    assert rd == backend.float64 and rdev == want
    assert out.dtype == mesh_backend.float64 and mesh_backend.device(
        out
    ) == mesh_backend.device(xy)
    assert to_np(out).tolist() == [[0.0, 0.5], [1.0, -1.5]]


def test_the_build_samples_no_point_twice():
    _, lens_ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.1)
    points = np.concatenate(lens_.calls.raytrace)
    assert len({tuple(p) for p in points}) == len(points)


def test_no_build_path_calls_the_lens_jacobian():
    mesh, lens_ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    extend_lens_mesh(mesh, lens_.raytrace, 6.0)
    assert len(lens_.calls.raytrace) > 0 and lens_.calls.jacobian == []
    closed = numpy_lens(sie_like, sie_like_jacobian)
    grown, _ = build_closed_lens_mesh(closed.raytrace, 2.0, 4, 0.05)
    assert grown.lattice.fov > 2.0  # an extension ran
    assert closed.calls.jacobian == []


def test_a_mesh_has_images_and_a_fitted_jacobian_at_its_vertices_and_nothing_of_the_lens_s():
    mesh, _ = build(sie_like, sie_like_jacobian)
    assert (
        "vertices_det" not in mesh._fields and "vertices_sigma_min" not in mesh._fields
    )
    assert (
        mesh.vertices_source.dtype
        == mesh.vertices_jacobian.dtype
        == mesh_backend.float64
    )
    assert tuple(mesh.vertices_jacobian.shape) == (mesh.vertices_lens.shape[0], 2, 2)


def test_a_float32_lens_still_builds_and_traces_its_fold():
    f32 = backend.float32

    def raytrace(x, y):
        out = localised_fold(np.stack((to_np(x), to_np(y)), axis=-1)).astype(np.float32)
        out = backend.as_array(out, dtype=f32, device=backend.device(x))
        return out[:, 0], out[:, 1]

    xy = f64([[0.3, -0.3], [-1.2, 0.7], [1.9, 0.01]])
    assert make_sampler(raytrace, None)(xy).dtype == mesh_backend.float64
    mesh = to_mesh(build_lens_mesh(raytrace, 4.0, 4, 0.05))
    want, _ = build(localised_fold, localised_fold_jacobian, 4.0, 4, 0.05)
    leaves_ij = [to_np(m.vertices_ij)[to_np(m.leaves)] for m in (mesh, want)]
    assert np.array_equal(*leaves_ij)
    got, ref = critical_curves_and_caustics(mesh), critical_curves_and_caustics(want)
    assert got.closed.shape[0] == ref.closed.shape[0] > 0
    a, b = to_np(got.lens), to_np(ref.lens)
    assert np.linalg.norm(a[:, None] - b[None], axis=-1).min(axis=1).max() < 1e-3


def test_localised_fold_s_kink_lies_on_no_lattice_row():
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    y = to_np(mesh.vertices_lens)[:, 1]
    assert not np.any(np.isclose(np.abs(y), 0.55, rtol=0, atol=1e-12))
    p = np.array([[0.3, -0.55 - 1e-9], [0.3, -0.55 + 1e-9]])
    assert np.allclose(localised_fold(p)[0], localised_fold(p)[1], atol=1e-8)
    J = localised_fold_jacobian(p)
    assert J[0, 1, 1] > 0 > J[1, 1, 1]


def test_finest_leaves_are_no_longer_than_the_mesh_s_min_img_sep(sie_mesh):
    mesh = sie_mesh
    tri = to_np(mesh.vertices_lens)[to_np(mesh.leaves)]
    longest = np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=-1).max(axis=1)
    level = to_np(mesh.origin_level)[to_np(mesh.leaf_origin)]
    finest = level == mesh.lattice.level - 1
    assert finest.any()
    assert longest[finest].max() <= mesh.min_img_sep * (1 + 1e-12)


def test_finest_level_midpoints_are_band_samples_but_never_vertices():
    mesh, _ = build(row_fold, row_fold_jacobian, init_res=8, min_img_sep=2e-2)
    lat, band = mesh.lattice, mesh.critical_band
    ij = np.rint((to_np(band.lens) - to_np(lat.lo)) / lat.scale).astype(np.int64)
    ij = ij + lat.origin
    odd = (ij % 2 == 1).any(axis=1)
    vertices = {tuple(p) for p in to_np(mesh.vertices_ij)}
    assert odd.any() and (~odd).any()
    assert all((tuple(p) in vertices) != o for p, o in zip(ij, odd))


@pytest.mark.parametrize("batch_size", [1, 7])
def test_the_build_does_not_depend_on_the_batch_size(batch_size):
    want, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.1)
    got, lens_ = build(
        localised_fold, localised_fold_jacobian, min_img_sep=0.1, batch_size=batch_size
    )
    assert (
        max(len(c) for c in lens_.calls.raytrace + lens_.calls.jacobian) <= batch_size
    )
    assert_same(want, got)


BUILD = dict(fov=4.0, init_res=3, min_img_sep=0.5, max_depth=3)


def leaf_status(mesh):
    return to_np(mesh.origin_status)[to_np(mesh.leaf_origin)]


def leaf_level(mesh):
    return to_np(mesh.origin_level)[to_np(mesh.leaf_origin)]


@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_the_index_holds_exactly_the_leaves_with_a_finite_raytrace(bad):
    """A leaf whose raytrace is non-finite carries that flag alone and a
    non-finite vertex, every other leaf has finite vertices, only finest-level
    leaves carry a flag, and the index -- unconverged leaves included -- holds
    exactly the finite ones."""
    fn, jac = broken_where(
        localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.0, bad
    )
    mesh, _ = build(fn, jac, min_img_sep=0.05)
    status, level = to_np(mesh.origin_status), to_np(mesh.origin_level)
    raytrace_bad = (status & LEAF_RAYTRACE_NONFINITE) != 0
    finite_vertices = np.isfinite(
        to_np(mesh.vertices_source)[to_np(mesh.origin_leaves)]
    ).all(axis=(1, 2))
    assert raytrace_bad.any()
    assert (status[raytrace_bad] == LEAF_RAYTRACE_NONFINITE).all()
    assert not finite_vertices[raytrace_bad].any()
    assert finite_vertices[~raytrace_bad].all()
    assert (level[status != LEAF_CONVERGED] == mesh.lattice.level - 1).all()
    leaf = leaf_status(mesh)
    finite = (leaf & LEAF_RAYTRACE_NONFINITE) == 0
    assert (
        finite & (leaf != LEAF_CONVERGED)
    ).any(), "fixture must leave a finite leaf unconverged"
    assert (
        np.unique(to_np(mesh.index.leaves)).tolist() == np.flatnonzero(finite).tolist()
    )


def test_leaf_grow_is_the_origin_s_deviation_where_it_did_not_converge(sie_mesh):
    mesh = sie_mesh
    grow = to_np(leaf_grow(mesh))
    origin = to_np(mesh.leaf_origin)
    converged = to_np(mesh.origin_status)[origin] == LEAF_CONVERGED
    assert (grow[converged] == 0).all()
    deviation = to_np(mesh.origin_deviation)[origin]
    assert np.array_equal(grow[~converged], deviation[~converged])
    assert (grow[~converged] > 0).any()


def test_leaf_grow_is_the_origin_s_deviation_beside_a_leaf_left_out_of_the_index():
    """On every finite leaf sharing a vertex with a non-finite one, converged or not; elsewhere a converged leaf reaches nothing.

    Next to a point mass ``sigma_min`` is large, so the leaves around its
    non-finite center converge with deviations far above ``min_img_sep``.
    """
    mesh, _ = build(*point_mass(), min_img_sep=0.2, centers=[[0.0, 0.0]])
    grow = to_np(leaf_grow(mesh))
    status, leaves = leaf_status(mesh), to_np(mesh.leaves)
    out = (status & LEAF_RAYTRACE_NONFINITE) != 0
    beside = ~out & np.isin(leaves, leaves[out]).any(axis=1)
    converged = status == LEAF_CONVERGED
    assert (beside & converged).any(), "fixture must converge next to the center"
    deviation = to_np(mesh.origin_deviation)[to_np(mesh.leaf_origin)]
    assert np.array_equal(grow[beside], deviation[beside])
    assert (deviation[beside & converged] > mesh.min_img_sep).all()
    assert (grow[converged & ~beside] == 0).all()


def test_build_halves_the_requested_min_img_sep():
    mesh, _ = user_build(affine, affine_jacobian, 4.0, 2, 0.4, max_depth=3)
    assert mesh.min_img_sep == pytest.approx(0.2)


def test_every_vertex_carries_its_image_as_float64_exactly():
    mesh, _ = user_build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    for name in ("vertices_lens", "vertices_source"):
        assert getattr(mesh, name).dtype == backend.float64, name
    xy = to_np(mesh.vertices_lens)
    assert np.array_equal(to_np(mesh.vertices_source), localised_fold(xy))


def test_build_returns_a_consistent_mesh_for_an_affine_map():
    mesh, _ = build(affine, affine_jacobian)
    L = to_np(mesh.leaves).shape[0]
    assert L == 2 * 4**2
    assert (leaf_status(mesh) == LEAF_CONVERGED).sum() == L
    assert to_np(mesh.origin_leaves).shape[0] == L
    assert np.array_equal(to_np(mesh.leaf_origin), np.arange(L))
    src, lens_xy = to_np(mesh.vertices_source), to_np(mesh.vertices_lens)
    assert np.allclose(src, lens_xy @ AFFINE.T, rtol=1e-10, atol=1e-12)


def test_two_builds_with_centers_are_identical_holes_included():
    c = (0.3001, -0.2003)  # off every lattice point

    fn, jac = sis([c])
    a, _ = build(fn, jac, 4.0, 8, 0.02, centers=[c])
    b, _ = build(fn, jac, 4.0, 8, 0.02, centers=[c])
    assert a.holes.centers.shape[0] == 1
    assert_same(a, b)


def test_depth_limit_warns_and_names_the_required_max_depth():
    with pytest.warns(
        UserWarning, match=r"Lens mesh is depth-limited.*max_depth >= \d+"
    ):
        mesh, _ = build(
            localised_fold, localised_fold_jacobian, min_img_sep=1e-4, max_depth=2
        )
    assert mesh.lattice.level - 1 == 2


def test_depth_limited_warning_never_bare_quotes_the_halved_value():
    """The warning quotes what the caller passed, and the halved value only labelled."""
    with pytest.warns(UserWarning, match=r"min_img_sep=0\.0002 arcsec") as record:
        build(affine, affine_jacobian, min_img_sep=2e-4, max_depth=1)
    assert not any("min_img_sep=0.0001" in str(w.message) for w in record)


def test_parity_invalid_partitions_the_max_level_leaves(sie_mesh):
    """Finest-level leaves converge or carry only criterion flags; coarser ones all converge.

    The cored SIE is finite everywhere and its Jacobian is never exactly
    singular at a sample, so no non-finite flag can occur.
    """
    mesh = sie_mesh
    assert np.isfinite(to_np(mesh.vertices_source)).all()
    status, level = leaf_status(mesh), leaf_level(mesh)
    at_max = level == mesh.lattice.level - 1
    assert at_max.any()
    criterion_flags = (
        LEAF_CONVERGENCE_FAILED
        | LEAF_APPROX_PARITY_UNRESOLVED
        | LEAF_JACOBIAN_PARITY_UNRESOLVED
    )
    assert not (status[at_max] & ~criterion_flags).any()
    assert (status[at_max] != LEAF_CONVERGED).any()
    assert (status[at_max] == LEAF_CONVERGED).any()
    assert (status[~at_max] == LEAF_CONVERGED).all()


def test_parity_band_is_bounded_by_a_small_multiple_of_min_img_sep(sie_mesh):
    """Halving min_img_sep keeps the condemned band near the center within 3.5x the request."""
    requested_min_img_sep = 1e-2
    mesh = sie_mesh
    status = leaf_status(mesh)
    leaves = to_np(mesh.leaves)
    vl = to_np(mesh.vertices_lens)
    invalid = np.flatnonzero(status != LEAF_CONVERGED)
    assert invalid.size > 0
    radius = np.linalg.norm(vl[leaves[invalid]].mean(axis=1), axis=-1)
    near_center = radius < 0.05
    assert near_center.any()
    core = radius[near_center]
    assert core.max() - core.min() <= 3.5 * requested_min_img_sep


def lattice_samples(mesh):
    """Lattice-exact positions of every finite finest-level origin's six samples."""
    level, status = to_np(mesh.origin_level), to_np(mesh.origin_status)
    rows = np.flatnonzero(
        (level == mesh.lattice.level - 1) & ((status & LEAF_RAYTRACE_NONFINITE) == 0)
    )
    return rows, origin_samples(mesh, rows)


BAND_CASES = [
    pytest.param(
        localised_fold,
        localised_fold_jacobian,
        dict(init_res=4, min_img_sep=0.05),
        id="localised_fold",
    ),
    pytest.param(
        row_fold, row_fold_jacobian, dict(init_res=8, min_img_sep=2e-2), id="row_fold"
    ),
    pytest.param(
        *broken_where(localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.0),
        dict(init_res=4, min_img_sep=0.05),
        id="nan_region",
    ),
]


def six_ij(mesh, rows):
    """Lattice coordinates of the six samples of origins ``rows``, ``(n, 6, 2)``: vertices, then midpoints."""
    v = to_np(mesh.vertices_ij)[to_np(mesh.origin_leaves)[rows]]
    m = np.stack(
        [(v[:, (i + 1) % 3] + v[:, (i + 2) % 3]) // 2 for i in range(3)], axis=1
    )
    return np.concatenate((v, m), axis=1)


def fitted_det6(mesh, rows):
    J = to_np(mesh.vertices_jacobian)[to_np(mesh.origin_leaves)[rows]]

    def det2(A):  # the explicit 2x2 formula: np.linalg.det goes through slogdet
        return A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]

    det = det2(J)
    mid = np.stack(
        [det2(0.5 * (J[:, (i + 1) % 3] + J[:, (i + 2) % 3])) for i in range(3)],
        axis=1,
    )
    spread = np.max(
        [
            np.linalg.norm(J[:, a] - J[:, b], axis=(1, 2))
            for a, b in ((0, 1), (1, 2), (2, 0))
        ],
        axis=0,
    )
    return np.concatenate((det, mid), axis=1), spread


def independent_measured(mesh):
    """The samples a mesh must measure: those of its unresolved finest-level origins with finite images."""
    finest = np.flatnonzero(to_np(mesh.origin_level) == mesh.lattice.level - 1)
    _, spread = fitted_det6(mesh, finest)
    finite = (to_np(mesh.origin_status)[finest] & LEAF_RAYTRACE_NONFINITE) == 0
    rows = finest[~(spread < RESOLVED_SPREAD) & finite]
    return {tuple(p) for p in six_ij(mesh, rows).reshape(-1, 2).tolist()}


def independent_band(mesh):
    """Finest-level origins whose six det change class: fitted, or measured where the table has the sample."""
    finest = np.flatnonzero(to_np(mesh.origin_level) == mesh.lattice.level - 1)
    det6, _ = fitted_det6(mesh, finest)
    table = {
        tuple(p): d
        for p, d in zip(
            to_np(mesh.critical_band.measured_ij).tolist(),
            to_np(mesh.critical_band.measured_det).tolist(),
        )
    }
    ij = six_ij(mesh, finest)
    for r in range(finest.size):
        for s in range(6):
            det6[r, s] = table.get(tuple(ij[r, s]), det6[r, s])
    ok = (
        np.isfinite(det6).all(axis=1) & (det6 >= 0).any(axis=1) & (det6 < 0).any(axis=1)
    )
    return finest[ok], det6[ok]


@pytest.mark.parametrize("fn, jac, kw", BAND_CASES)
def test_the_band_is_the_sign_change_leaves_recomputed_independently(fn, jac, kw):
    mesh, _ = build(fn, jac, **kw)
    band = mesh.critical_band
    rows, det6 = independent_band(mesh)
    assert np.array_equal(to_np(band.leaves), rows) and rows.size > 0
    np.testing.assert_array_equal(to_np(band.det[band.samples]), det6)
    assert to_np(in_band(band.det[band.samples])).all()


SINGULAR = dict(fov=4.0, init_res=8, min_img_sep=0.02)


@pytest.mark.parametrize("centers", [None, [[0.0, 0.0]]])
def test_the_measured_samples_are_those_of_the_unresolved_finite_finest_leaves(centers):
    fn, jac = sis()
    kw = {} if centers is None else dict(centers=f64(centers))
    mesh, _ = build(fn, jac, **SINGULAR, **kw)
    got = {tuple(p) for p in to_np(mesh.critical_band.measured_ij).tolist()}
    assert got == independent_measured(mesh) and len(got) > 0
    key = to_np(lattice_key(mesh.lattice, mesh.critical_band.measured_ij))
    assert (np.diff(key) > 0).all()


def test_a_measured_det_is_the_central_difference_of_the_raytrace_and_has_the_lens_s_sign():
    fn, jac = sis()
    mesh, _ = build(fn, jac, **SINGULAR, centers=f64([[0.0, 0.0]]))
    band = mesh.critical_band
    xy = to_np(lattice_xy(mesh.lattice, band.measured_ij))
    d = MEASURE_OFFSET * mesh.min_img_sep
    Jx = (fn(xy + [d, 0.0]) - fn(xy - [d, 0.0])) / (2 * d)
    Jy = (fn(xy + [0.0, d]) - fn(xy - [0.0, d])) / (2 * d)
    want = Jx[:, 0] * Jy[:, 1] - Jy[:, 0] * Jx[:, 1]
    got = to_np(band.measured_det)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    outside = np.hypot(*xy.T) > mesh.min_img_sep
    assert (np.sign(got[outside]) == np.sign(numpy_det(jac, xy[outside]))).all()


def test_a_lens_the_fit_resolves_measures_nothing():
    mesh, lens_ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    assert mesh.critical_band.measured_ij.shape[0] == 0
    asked = np.concatenate(lens_.calls.raytrace)
    assert np.unique(asked, axis=0).shape[0] == asked.shape[0]


def test_band_samples_are_lattice_exact_and_carry_the_lens_values():
    mesh, _ = build(row_fold, row_fold_jacobian, fov=4.0, init_res=8, min_img_sep=2e-2)
    band = mesh.critical_band
    leaves, samples = to_np(band.leaves), to_np(band.samples)
    xy, source = to_np(band.lens), to_np(band.source)
    assert leaves.size > 0
    vertices = to_np(mesh.vertices_lens)[to_np(mesh.origin_leaves)[leaves]]
    assert np.array_equal(xy[samples[:, :3]], vertices)
    rows, want = lattice_samples(mesh)
    position = {r: i for i, r in enumerate(rows.tolist())}
    assert np.array_equal(xy[samples], want[[position[r] for r in leaves.tolist()]])
    assert np.unique(xy, axis=0).shape[0] == xy.shape[0]
    assert np.array_equal(source, row_fold(xy))
    assert (np.diff(leaves) > 0).all()
    lat = mesh.lattice
    ij = np.rint((xy - to_np(lat.lo)) / lat.scale).astype(np.int64) + lat.origin
    assert (np.diff(ij[:, 0] * (lat.n + 1) + ij[:, 1]) > 0).all()


@pytest.mark.parametrize(
    "fn, jac, kw",
    [
        (affine, affine_jacobian, dict()),
        (collapse, collapse_jacobian, dict(min_img_sep=0.5)),
    ],
    ids=["affine", "kappa_one_sheet"],
)
def test_band_is_empty_without_a_sign_change(fn, jac, kw):
    mesh, _ = build(fn, jac, **kw)
    band = mesh.critical_band
    assert tuple(band.leaves.shape) == (0,)
    assert tuple(band.samples.shape) == (0, 6)
    assert tuple(band.lens.shape) == (0, 2)
    assert tuple(band.source.shape) == (0, 2)
    assert tuple(band.det.shape) == (0,)


def test_a_fold_on_a_lattice_row_is_one_curve_along_the_row():
    mesh, _ = build(row_fold, row_fold_jacobian, fov=4.0, init_res=8, min_img_sep=2e-2)
    curves = critical_curves_and_caustics(mesh)
    assert curves.offsets.shape[0] - 1 == 1
    assert np.abs(to_np(curves.lens)[:, 1] - 0.5).max() < 1e-9


@pytest.mark.filterwarnings("ignore:Lens mesh is depth-limited")
def test_a_depth_limited_mesh_still_traces_a_closed_tangential_curve():
    mesh, _ = build(
        sie_like, sie_like_jacobian, fov=4.0, init_res=8, min_img_sep=1e-3, max_depth=4
    )
    curves = critical_curves_and_caustics(mesh)
    assert to_np(curves.closed).any()


def test_a_fold_is_traced_straight_at_a_leaf_edge_of_two_to_the_minus_twenty():
    h = 2.0**-20
    mesh, _ = build(
        row_fold,
        row_fold_jacobian,
        fov=64 * h,
        init_res=4,
        min_img_sep=2 * h,
        x0=0.0,
        y0=0.5 + 3.3 * h,
    )
    curves = critical_curves_and_caustics(mesh)
    assert curves.offsets.shape[0] - 1 == 1
    assert np.abs(to_np(curves.lens)[:, 1] - 0.5).max() < h


def test_a_mesh_survives_a_pickle_round_trip():
    mesh, _ = user_build(affine, affine_jacobian, **BUILD)
    again = pickle.loads(pickle.dumps(mesh))
    for name in ("index", "critical_band", "holes", "lattice"):
        assert type(getattr(again, name)) is type(getattr(mesh, name)), name
    for name in ("vertices_lens", "vertices_source", "leaves", "origin_status"):
        assert np.array_equal(to_np(getattr(again, name)), to_np(getattr(mesh, name)))


def origin_samples(mesh, rows):
    """Lattice-exact lens-plane positions of origins ``rows``' six samples, ``(n, 6, 2)``."""
    lat = mesh.lattice
    vij = to_np(mesh.vertices_ij)[to_np(mesh.origin_leaves)[rows]] - lat.origin
    mij = np.stack(
        [
            (vij[:, 1] + vij[:, 2]) // 2,
            (vij[:, 2] + vij[:, 0]) // 2,
            (vij[:, 0] + vij[:, 1]) // 2,
        ],
        axis=1,
    )
    six = np.concatenate([vij, mij], axis=1).astype(np.float64)
    return to_np(lat.lo) + six * lat.scale


def numpy_det(jac, xy):
    J = jac(xy.reshape(-1, 2)).reshape(*xy.shape[:-1], 2, 2)
    return J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]


def _gaussian_bump(p, w=0.08, amp=1.0, c=(0.13, 0.07)):
    """``0.5 p`` plus a narrow Gaussian bump along ``(1, 0.3)``."""
    r2 = ((p - np.asarray(c)) ** 2).sum(axis=-1)
    return p * 0.5 + (amp * np.exp(-r2 / (2 * w**2)))[:, None] * np.array([1.0, 0.3])


def _gaussian_bump_jacobian(p, w=0.08, amp=1.0, c=(0.13, 0.07)):
    d = p - np.asarray(c)
    g = amp * np.exp(-(d**2).sum(axis=-1) / (2 * w**2))
    grad = -(g / w**2)[:, None] * d
    return 0.5 * np.eye(2) + np.array([1.0, 0.3])[None, :, None] * grad[:, None, :]


def test_a_kappa_one_sheet_samples_every_lattice_point_once_and_converges_nothing():
    """``A == 0`` everywhere: every triangle fails the deviation test, and its dets, all zero, have no sign."""
    mesh, lens_ = build(collapse, collapse_jacobian, min_img_sep=1.0)
    level, status = to_np(mesh.origin_level), to_np(mesh.origin_status)
    assert (level == mesh.lattice.level - 1).all()
    assert (status == (LEAF_CONVERGENCE_FAILED | LEAF_JACOBIAN_PARITY_UNRESOLVED)).all()
    assert np.isfinite(to_np(mesh.vertices_source)).all()
    assert sum(len(c) for c in lens_.calls.raytrace) == (mesh.lattice.n + 1) ** 2


def test_a_nonfinite_subregion_is_split_down_to_the_finest_level():
    fn, jac = broken_where(affine, affine_jacobian, lambda p: p[:, 0] > 0.5)
    mesh, _ = build(fn, jac, min_img_sep=1.0)
    max_level = mesh.lattice.level - 1
    assert max_level > 0
    status, level = to_np(mesh.origin_status), to_np(mesh.origin_level)
    vs = to_np(mesh.vertices_source)[to_np(mesh.origin_leaves)]
    nonfinite = status == LEAF_RAYTRACE_NONFINITE
    good = status == LEAF_CONVERGED
    assert nonfinite.any() and good.any()
    assert (level[nonfinite] == max_level).all()
    assert not np.isfinite(vs[nonfinite]).all()
    assert np.isfinite(vs[good]).all()
    assert (good | nonfinite).all()


def test_only_the_six_triangles_at_a_point_singularity_stay_nonfinite():
    """A red split hands the bad vertex to one child, so six triangles share it at every level."""
    mesh, _ = build(*sis(), min_img_sep=0.1)
    assert mesh.lattice.level - 1 == 5
    status, level = to_np(mesh.origin_status), to_np(mesh.origin_level)
    nonfinite = (status & LEAF_RAYTRACE_NONFINITE) != 0
    assert nonfinite.sum() == 6
    assert (status[nonfinite] == LEAF_RAYTRACE_NONFINITE).all()
    assert (level[nonfinite] == 5).all()


def test_nonfinite_vertices_are_flagged_even_when_the_finest_level_is_zero():
    fn, jac = broken_where(affine, affine_jacobian, lambda p: p[:, 0] > 0.5)
    mesh, _ = build(fn, jac, min_img_sep=4.0)
    assert mesh.lattice.level - 1 == 0
    status = to_np(mesh.origin_status)
    vs = to_np(mesh.vertices_source)[to_np(mesh.origin_leaves)]
    assert (status == LEAF_RAYTRACE_NONFINITE).any()
    assert (status == LEAF_CONVERGED).any()
    assert set(np.unique(status).tolist()) <= {LEAF_CONVERGED, LEAF_RAYTRACE_NONFINITE}
    assert np.isfinite(vs[status == LEAF_CONVERGED]).all()
    for row in np.flatnonzero(status == LEAF_RAYTRACE_NONFINITE):
        assert not np.isfinite(vs[row]).all()


def test_a_nonfinite_midpoint_with_finite_vertices_is_condemned_alone():
    """A NaN band around ``x = 0.5`` hits the eight triangles' midpoints only."""
    fn, jac = broken_where(
        affine, affine_jacobian, lambda p: np.abs(p[:, 0] - 0.5) < 0.1
    )
    mesh, _ = build(fn, jac, min_img_sep=4.0)
    assert mesh.lattice.level - 1 == 0
    status = to_np(mesh.origin_status)
    vs = to_np(mesh.vertices_source)[to_np(mesh.origin_leaves)]
    nonfinite = np.flatnonzero(status == LEAF_RAYTRACE_NONFINITE)
    assert nonfinite.size == 8
    assert np.isfinite(vs[nonfinite]).all()
    assert (status == LEAF_CONVERGED).any()


def test_finest_level_flags_match_both_parity_tests_row_for_row():
    """Child parity and Jacobian parity recomputed from lattice-exact samples."""
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.1)
    max_level = mesh.lattice.level - 1
    level, status = to_np(mesh.origin_level), to_np(mesh.origin_status)
    rows = np.flatnonzero(level == max_level)
    assert rows.size > 0
    xy = origin_samples(mesh, rows)
    beta6 = localised_fold(xy.reshape(-1, 2)).reshape(-1, 6, 2)
    child_ok = to_np(
        parity_from_children(child_shape_matrices(f64(beta6[:, :3]), f64(beta6[:, 3:])))
    )
    det = numpy_det(localised_fold_jacobian, xy)
    usable = (np.isfinite(det) & (det != 0)).all(axis=1)
    mixed = usable & ~((det > 0).all(axis=1) | (det < 0).all(axis=1))
    st = status[rows]
    approx = (st & LEAF_APPROX_PARITY_UNRESOLVED) != 0
    jacobian = (st & LEAF_JACOBIAN_PARITY_UNRESOLVED) != 0
    for flag in (approx, jacobian):
        assert flag.any() and not flag.all()
    assert approx.tolist() == (~child_ok).tolist()
    assert jacobian.tolist() == mixed.tolist()
    assert (level[status != LEAF_CONVERGED] == max_level).all()


@pytest.mark.parametrize(
    "fn,jac,kw",
    [
        (localised_fold, localised_fold_jacobian, dict(min_img_sep=0.1)),
        (*sis(), dict(min_img_sep=0.1)),
        (_gaussian_bump, _gaussian_bump_jacobian, dict(init_res=8, min_img_sep=0.04)),
    ],
    ids=["localised_fold", "sis", "gaussian_bump"],
)
def test_converged_leaves_pass_jacobian_parity_at_their_own_samples(fn, jac, kw):
    """No origin converges with a critical curve between its samples, balance-forced ones included."""
    mesh, _ = build(fn, jac, **kw)
    level, status = to_np(mesh.origin_level), to_np(mesh.origin_status)
    rows = np.flatnonzero(status == LEAF_CONVERGED)
    assert (level[rows] < mesh.lattice.level - 1).any()
    det = numpy_det(jac, origin_samples(mesh, rows))
    assert np.isfinite(det).all() and (det != 0).all()
    assert ((det > 0).all(axis=1) | (det < 0).all(axis=1)).all()


def test_a_subhalo_beside_the_critical_curve_is_resolved_and_audited():
    fn, jac, center = sie_with_subhalo()
    mesh, _ = build(
        fn,
        jac,
        fov=4.0,
        init_res=8,
        min_img_sep=5e-3,
        centers=f64([[0.0, 0.0], center]),
    )
    converged = np.flatnonzero(to_np(mesh.origin_status) == LEAF_CONVERGED)
    det = numpy_det(
        jac, to_np(origin_samples(mesh, i64(converged))).reshape(-1, 2)
    ).reshape(-1, 6)
    assert ((det > 0).all(axis=1) | (det < 0).all(axis=1)).all()
    curves = critical_curves_and_caustics(mesh)
    points = to_np(curves.lens)
    near = np.hypot(*(points - center).T) < 4 * 0.05
    assert near.any()
    # |grad det A| at the points, by central differences of jac with step 1e-6
    dx, dy = np.array([1e-6, 0.0]), np.array([0.0, 1e-6])
    grad = np.hypot(
        numpy_det(jac, points + dx) - numpy_det(jac, points - dx),
        numpy_det(jac, points + dy) - numpy_det(jac, points - dy),
    ) / (2 * 1e-6)
    off = np.abs(numpy_det(jac, points)) / grad
    traced = np.ones(points.shape[0], bool)  # not on a hole circle
    for c, r in zip(to_np(mesh.holes.centers), to_np(mesh.holes.radius)):
        traced &= np.hypot(*(points - c).T) > r + 1e-9
    assert off[traced].max() < 0.1 * mesh.min_img_sep


def test_coverage_does_not_drop_at_level_transitions():
    """Every point a uniform finest-level mesh covers, the adaptive mesh covers too."""
    fov, init_res, sep = 4.0, 4, 0.1
    mesh, _ = build(
        localised_fold,
        localised_fold_jacobian,
        fov=fov,
        init_res=init_res,
        min_img_sep=sep,
    )
    ml = mesh.lattice.level - 1
    assert ml >= 3 and len(set(leaf_level(mesh).tolist())) > 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uniform, _ = build(
            localised_fold,
            localised_fold_jacobian,
            fov=fov,
            init_res=init_res * 2**ml,
            min_img_sep=sep,
            max_depth=0,
        )
    grid = np.linspace(-0.9, 0.9, 120)
    beta = f64(np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2))

    def covered(m):
        qidx, _, _ = index_hits(m.index, m.vertices_source, m.leaves, beta)
        return np.bincount(to_np(qidx), minlength=beta.shape[0]) > 0

    hit_a, hit_u = covered(mesh), covered(uniform)
    assert int((hit_u & ~hit_a).sum()) == 0


def test_the_criterion_is_blind_to_structure_below_the_sampling_scale():
    """A bump inside every sample's blind radius is invisible at init_res 2, and seen at 32."""
    center = np.array([[-2.0, -2.0], [0.0, 0.0], [-2.0, 0.0]]).mean(axis=0)

    def bumped(p):
        r2 = ((p - center) ** 2).sum(axis=-1)
        bump = (2.0 * np.exp(-r2 / (2 * 0.08**2)))[:, None] * np.array([1.0, 0.0])
        return p * 0.5 + bump

    def bumped_jacobian(p):
        d = p - center
        g = 2.0 * np.exp(-(d**2).sum(axis=-1) / (2 * 0.08**2))
        grad = -(g / 0.08**2)[:, None] * d
        return 0.5 * np.eye(2) + np.array([1.0, 0.0])[None, :, None] * grad[:, None, :]

    coarse, _ = build(bumped, bumped_jacobian, init_res=2, min_img_sep=0.05)
    fine, _ = build(bumped, bumped_jacobian, init_res=32, min_img_sep=0.05)
    assert fine.lattice.level - 1 >= 1
    assert (to_np(coarse.origin_level) == 0).all()
    assert (to_np(coarse.origin_status) == LEAF_CONVERGED).all()
    assert (to_np(fine.origin_level) > 0).any()


def test_an_extension_samples_no_point_the_old_mesh_holds():
    mesh, _ = build(row_fold, row_fold_jacobian, init_res=8, min_img_sep=2e-2)
    assert mesh.critical_band.leaves.shape[0] > 0
    lens_ = numpy_lens(row_fold, row_fold_jacobian)
    extend_lens_mesh(mesh, lens_.raytrace, 6.0)
    old = np.concatenate((to_np(mesh.vertices_lens), to_np(mesh.critical_band.lens)))
    old = {tuple(p) for p in old}
    assert not any(tuple(p) in old for p in np.concatenate(lens_.calls.raytrace))


# Extension and the closed build. The fixtures use dyadic fovs, centers and
# cell sizes, where a fresh build's lattice coincides with an extended one.


def called_at(calls, method):
    """Every point ``method`` was called on, stacked, shape ``(K, 2)``."""
    batches = getattr(calls, method)
    return np.concatenate(batches, axis=0) if batches else np.zeros((0, 2))


# An SIS whose tangential curve, radius 0.6 about C_IN, lies inside [-1, 1]**2.
C_IN = (0.3001, -0.2003)


def seam_fold(seam):
    """Curved, with a fold at ``x = seam - 0.4``, on ``[seam - 0.6, seam]``; affine elsewhere."""

    def fn(p):
        x = p[:, 0]
        u = np.clip(x - (seam - 0.6), 0.0, 0.6)
        tail = np.maximum(x - seam, 0.0)
        return np.stack([0.6 * x - 5.0 * u**3 - 5.4 * tail, p[:, 1]], axis=-1)

    def jac(p):
        x = p[:, 0]
        u = np.clip(x - (seam - 0.6), 0.0, 0.6)
        J = np.zeros((p.shape[0], 2, 2))
        J[:, 0, 0] = np.where(x > seam, -4.8, 0.6 - 15.0 * u**2)
        J[:, 1, 1] = 1.0
        return J

    return fn, jac


def mirrored(fn, jac, seam):
    """``fn`` reflected in the line ``x = seam``."""

    def reflect(p):
        q = p.copy()
        q[:, 0] = 2.0 * seam - q[:, 0]
        return q

    def fn_r(p):
        return fn(reflect(p))

    def jac_r(p):
        J = jac(reflect(p))
        J[:, :, 0] = -J[:, :, 0]
        return J

    return fn_r, jac_r


def ring_fold(seam):
    """`seam_fold` mirrored: affine up to ``x = seam``, folded just past it."""
    return mirrored(*seam_fold(seam), seam)


def edge_bump(p):
    """A fold confined to the right edge of ``[-2, 2]**2``, exactly affine outside it."""
    x, y = p[:, 0], p[:, 1]
    g = np.where((x > 1.4) & (x < 2.0), 60.0 * (x - 1.4) ** 2 * (2.0 - x) ** 2, 0.0)
    h = np.where(np.abs(y) < 2.0, (1.0 - y**2 / 4.0) ** 2, 0.0)
    return np.stack([0.6 * x + g * h, y], axis=-1)


def edge_bump_jacobian(p):
    x, y = p[:, 0], p[:, 1]
    inx, iny = (x > 1.4) & (x < 2.0), np.abs(y) < 2.0
    g = np.where(inx, 60.0 * (x - 1.4) ** 2 * (2.0 - x) ** 2, 0.0)
    h = np.where(iny, (1.0 - y**2 / 4.0) ** 2, 0.0)
    dg = np.where(inx, 120.0 * (x - 1.4) * (2.0 - x) * (3.4 - 2.0 * x), 0.0)
    dh = np.where(iny, -y * (1.0 - y**2 / 4.0), 0.0)
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 0.6 + dg * h
    J[:, 0, 1] = g * dh
    J[:, 1, 1] = 1.0
    return J


def test_a_mesh_records_its_lattice_and_its_vertices_compacted_in_lattice_key_order():
    mesh, _ = build(localised_fold, localised_fold_jacobian, 4.0, 4, 0.05, x0=0.25)
    lat = mesh.lattice
    assert lat.origin == 0 and lat.n == 4 << lat.level
    assert (lat.fov, lat.init_res) == (4.0, 4)
    assert to_np(mesh.vertices_ij).dtype == np.int64
    assert np.array_equal(
        to_np(mesh.vertices_lens), to_np(lattice_xy(lat, mesh.vertices_ij))
    )
    assert (np.diff(to_np(lattice_key(lat, mesh.vertices_ij))) > 0).all()
    assert np.unique(to_np(mesh.leaves)).tolist() == list(
        range(mesh.vertices_lens.shape[0])
    )


def test_origin_cls_is_the_orientation_class_of_each_origin_leaf():
    """``P = 2**(L - d) * R @ G[c]`` in lattice units, for every origin."""
    mesh, _ = build(localised_fold, localised_fold_jacobian, 4.0, 4, 0.05)
    _, G, _, _, _ = child_matrix_tables()
    ij = to_np(mesh.vertices_ij)[to_np(mesh.origin_leaves)]
    P = np.stack((ij[:, 1] - ij[:, 0], ij[:, 2] - ij[:, 0]), axis=-1)
    level = to_np(mesh.origin_level)
    R = np.array([[1, 0], [1, 1]])
    scale = (1 << (mesh.lattice.level - level))[:, None, None]
    assert len(set(level.tolist())) > 1
    assert np.array_equal(P, scale * (R @ to_np(G)[to_np(mesh.origin_cls)]))


def test_seeding_a_mesh_and_freezing_it_again_reproduces_it():
    """Seed then freeze, with nothing added, is the identity, and calls no lens."""
    fn, jac = broken_where(
        localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.5
    )
    mesh, lens_ = build(fn, jac, 4.0, 4, 0.05, centers=[(1.0001, 0.1003)])
    assert mesh.critical_band.leaves.shape[0] > 0
    assert mesh.holes.radius.shape[0] == 1
    assert ((to_np(mesh.origin_status) & LEAF_RAYTRACE_NONFINITE) != 0).any()
    lens_.calls.raytrace.clear()
    lens_.calls.jacobian.clear()
    lat = to_device(mesh.lattice, build_device())
    cache, store, seed = _seed(mesh, lat, 0)
    sample = make_sampler(lens_.raytrace, None)
    again = _freeze(
        lat,
        cache,
        store,
        i64([]),
        sample,
        mesh.min_img_sep,
        mesh.holes,
        None,
        seed,
    )
    assert_same(again, mesh)
    assert not lens_.calls.raytrace and not lens_.calls.jacobian


@pytest.mark.parametrize("extend", [False, True], ids=["build", "extension"])
def test_every_stepped_pass_holds_at_most_status_rows_rows_per_step(
    extend, monkeypatch
):
    """
    Each step of every pass a freeze takes in steps holds at most ``STATUS_ROWS`` rows, and the mesh does not change.

    Each pass is seen through the call it makes once per step, on that
    step's rows: the status pass through ``_status_rows``; the refit of the
    old rows an extension splits through ``_six_values`` called outside
    ``_status_rows``; the measuring pass's leaf selection through
    ``resolved``; the band pass through ``in_band``. The extension is
    ``ring_fold(2.0)``'s, which splits old rows.
    """
    fn, jac = ring_fold(2.0) if extend else (localised_fold, localised_fold_jacobian)

    def run():
        mesh, lens_ = build(fn, jac, 4.0, 4, 0.05)
        if extend:
            mesh = extend_lens_mesh(mesh, lens_.raytrace, 6.0)
        return mesh

    steps = {"status": [], "split": [], "measure": [], "band": []}
    in_status_rows = []
    status_rows = lens_mesh_module._status_rows
    six_values = lens_mesh_module._six_values
    resolved = lens_mesh_module.resolved
    in_band_ = lens_mesh_module.in_band

    def spy_status_rows(lat, cache, store, rows, *rest):
        steps["status"].append(rows.shape[0])
        in_status_rows.append(True)
        try:
            return status_rows(lat, cache, store, rows, *rest)
        finally:
            in_status_rows.pop()

    def spy_six_values(lat, cache, store, rows, *rest):
        if not in_status_rows:
            steps["split"].append(rows.shape[0])
        return six_values(lat, cache, store, rows, *rest)

    def spy_resolved(jacobian3):
        steps["measure"].append(jacobian3.shape[0])
        return resolved(jacobian3)

    def spy_in_band(det6):
        steps["band"].append(det6.shape[0])
        return in_band_(det6)

    monkeypatch.setattr(lens_mesh_module, "_status_rows", spy_status_rows)
    monkeypatch.setattr(lens_mesh_module, "_six_values", spy_six_values)
    monkeypatch.setattr(lens_mesh_module, "resolved", spy_resolved)
    monkeypatch.setattr(lens_mesh_module, "in_band", spy_in_band)
    whole = run()
    freezes = 2 if extend else 1
    calls = {name: len(got) for name, got in steps.items()}
    rows = {name: sum(got) for name, got in steps.items()}
    assert calls == dict(
        status=freezes, split=int(extend), measure=freezes, band=freezes
    ), "the default cap must hold each freeze's pass in one step"
    assert whole.critical_band.leaves.shape[0] > 0, "fixture must have a band"
    if extend:
        assert rows["split"] > 7, "fixture must split more old rows than one step holds"
    monkeypatch.setattr(lens_mesh_module, "STATUS_ROWS", 7)
    for got in steps.values():
        got.clear()
    assert_same(run(), whole, rounding=("vertices_jacobian", "critical_band.det"))
    for name, got in steps.items():
        assert sum(got) == rows[name] and max(got, default=0) <= 7, (name, got)
        assert len(got) > calls[name] or rows[name] == 0, (name, got)


@pytest.mark.parametrize("extend", [False, True], ids=["build", "extension"])
def test_the_refinement_and_the_status_table_are_freed_before_the_index_is_built(
    extend, monkeypatch
):
    """The cache, the store and the table the status pass reads are gone by the index build, the largest step."""
    held, alive = [], []
    refine = lens_mesh_module.refine
    status_rows = lens_mesh_module._status_rows
    build_index = lens_mesh_module.build_index

    def spy_refine(*args):
        cache, store = refine(*args)
        held.extend(weakref.ref(a) for a in (*cache, *store))
        return cache, store

    def spy_status_rows(lat, cache, store, rows, table_keys, table_values, sep):
        held.extend((weakref.ref(table_keys), weakref.ref(table_values)))
        return status_rows(lat, cache, store, rows, table_keys, table_values, sep)

    def spy_build_index(*args):
        alive.append(sum(ref() is not None for ref in held))
        held.clear()
        return build_index(*args)

    monkeypatch.setattr(lens_mesh_module, "refine", spy_refine)
    monkeypatch.setattr(lens_mesh_module, "_status_rows", spy_status_rows)
    monkeypatch.setattr(lens_mesh_module, "build_index", spy_build_index)
    mesh, lens_ = build(localised_fold, localised_fold_jacobian, 4.0, 4, 0.05)
    if extend:
        extend_lens_mesh(mesh, lens_.raytrace, 6.0)
    assert alive == [0] * (2 if extend else 1)


@pytest.mark.skipif(backend.backend != "torch", reason="reads torch storages")
def test_the_mesh_s_arrays_hold_no_memory_beyond_their_own_values():
    """A field that is a view keeps its whole base alive: the mesh's views share bases they fill."""

    def arrays(value):
        if hasattr(value, "_fields"):
            for v in value:
                yield from arrays(v)
        elif hasattr(value, "untyped_storage"):
            yield value

    mesh, _ = user_build(localised_fold, localised_fold_jacobian, 4.0, 4, 0.05)
    held = {
        a.untyped_storage().data_ptr(): a.untyped_storage().nbytes()
        for a in arrays(mesh)
    }
    assert sum(held.values()) <= sum(a.numel() * a.element_size() for a in arrays(mesh))


EQUIVALENCE = {
    "sie_like": (
        sie_like,
        sie_like_jacobian,
        dict(fov=2.0, init_res=4, min_img_sep=0.05),
        3.0,
    ),
    "localised_fold": (
        localised_fold,
        localised_fold_jacobian,
        dict(fov=4.0, init_res=4, min_img_sep=0.05),
        6.0,
    ),
    "row_fold": (
        row_fold,
        row_fold_jacobian,
        dict(fov=4.0, init_res=8, min_img_sep=2e-2),
        6.0,
    ),
    "nan_across_seam": (
        *broken_where(localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.5),
        dict(fov=4.0, init_res=4, min_img_sep=0.05),
        6.0,
    ),
    "affine": (
        affine,
        affine_jacobian,
        dict(fov=4.0, init_res=4, min_img_sep=0.05),
        6.0,
    ),
    "seam_fold": (*seam_fold(2.0), dict(fov=4.0, init_res=4, min_img_sep=0.05), 6.0),
    "ring_fold": (*ring_fold(2.0), dict(fov=4.0, init_res=4, min_img_sep=0.05), 6.0),
    "edge_bump": (
        edge_bump,
        edge_bump_jacobian,
        dict(fov=4.0, init_res=4, min_img_sep=0.05),
        6.0,
    ),
    "off_center": (
        localised_fold,
        localised_fold_jacobian,
        dict(fov=4.0, init_res=8, min_img_sep=0.05, x0=0.5, y0=-0.25),
        6.0,
    ),
    "depth_limited": (
        localised_fold,
        localised_fold_jacobian,
        dict(fov=4.0, init_res=4, min_img_sep=0.05, max_depth=3),
        6.0,
    ),
}


ROUNDING = ("vertices_jacobian", "critical_band.det")


def assert_same_point_sets(a, b, tol, what):
    """Every point of ``a`` within ``tol`` of a point of ``b``, and every point of ``b`` of one of ``a``."""
    a, b = to_np(a), to_np(b)
    assert a.shape[0] > 0 and b.shape[0] > 0, what
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    assert d.min(axis=1).max() <= tol and d.min(axis=0).max() <= tol, what


@pytest.mark.filterwarnings("ignore:Lens mesh is depth-limited")
@pytest.mark.parametrize("case", list(EQUIVALENCE))
def test_an_extension_is_the_fresh_build_of_the_larger_fov(case):
    fn, jac, kw, target = EQUIVALENCE[case]
    mesh, lens_ = build(fn, jac, **kw)
    got = extend_lens_mesh(mesh, lens_.raytrace, target)
    want, _ = build(fn, jac, **fresh_equivalent(kw, target))
    assert got.lattice.init_res > mesh.lattice.init_res
    if case != "row_fold":
        assert_same(got, want, rounding=ROUNDING)
        return
    # The fold's det is within rounding of zero on its lattice row, so band
    # membership can differ between the routes: compare the curves instead.
    for name in got._fields:
        if name != "critical_band":
            assert_same(getattr(got, name), getattr(want, name), name, ROUNDING)
    # A measured det depends on its point alone, so the table still matches.
    assert_same(got.critical_band.measured_ij, want.critical_band.measured_ij)
    assert_same(got.critical_band.measured_det, want.critical_band.measured_det)
    a, b = critical_curves_and_caustics(got), critical_curves_and_caustics(want)
    for field in ("lens", "source"):
        assert_same_point_sets(getattr(a, field), getattr(b, field), 1e-9, field)


def origin_triples(mesh, shift=0):
    """Each origin as its sorted lattice coordinates, shifted by ``shift`` on both axes."""
    ij = to_np(mesh.vertices_ij)[to_np(mesh.origin_leaves)] + shift
    return [tuple(sorted(map(tuple, t))) for t in ij.tolist()]


def changed_vertices(mesh, got):
    """Old vertices, by index in ``mesh``, that gained a holder or lost one to a split."""
    ring = (got.lattice.init_res - mesh.lattice.init_res) // 2
    shift = ring << mesh.lattice.level
    old, new = origin_triples(mesh, shift), origin_triples(got)
    old_set, new_set = set(old), set(new)
    index = {
        tuple(p): v for v, p in enumerate((to_np(mesh.vertices_ij) + shift).tolist())
    }
    changed, split = set(), set()
    for triple in old_set - new_set:  # an old origin the extension split
        split.update(index[p] for p in triple)
    for triple in new_set - old_set:  # a new origin
        changed.update(index[p] for p in triple if p in index)
    return changed | split, split


@pytest.mark.filterwarnings("ignore:Lens mesh is depth-limited")
def test_some_equivalence_case_splits_a_holder_of_an_old_vertex():
    """So that the comparison with the fresh build is known to cross a subtraction."""
    splitting = []
    for case, (fn, jac, kw, target) in EQUIVALENCE.items():
        mesh, lens_ = build(fn, jac, **kw)
        got = extend_lens_mesh(mesh, lens_.raytrace, target)
        if changed_vertices(mesh, got)[1]:
            splitting.append(case)
    assert splitting


def test_an_extension_keeps_an_untouched_vertex_s_jacobian_and_updates_the_rest():
    fn, jac, kw, target = EQUIVALENCE["seam_fold"]
    mesh, lens_ = build(fn, jac, **kw)
    noise = f64(
        np.random.default_rng(0).normal(size=tuple(mesh.vertices_jacobian.shape))
    )
    # Random bits: (x * c) / c != x for many doubles.
    marked = mesh._replace(vertices_jacobian=noise)
    got = extend_lens_mesh(marked, lens_.raytrace, target)
    want, _ = build(fn, jac, **fresh_equivalent(kw, target))
    changed, _ = changed_vertices(mesh, got)
    ring = (got.lattice.init_res - mesh.lattice.init_res) // 2
    shift = ring << mesh.lattice.level
    new_index = {tuple(p): v for v, p in enumerate(to_np(got.vertices_ij).tolist())}
    at = np.array(
        [new_index[tuple(p)] for p in (to_np(mesh.vertices_ij) + shift).tolist()]
    )
    kept = np.array(sorted(set(range(at.size)) - changed))
    moved = np.array(sorted(changed))
    assert kept.size and moved.size
    G, W = to_np(got.vertices_jacobian), to_np(want.vertices_jacobian)
    M = to_np(marked.vertices_jacobian)
    assert np.array_equal(G[at[kept]], M[kept])  # stored bits, untouched
    assert not np.array_equal(G[at[moved]], M[moved])  # re-formed
    # Each re-formed vertex starts from the marked mean, so each differs from
    # the fresh build's.
    assert not np.isclose(G[at[moved]], W[at[moved]]).all(axis=(1, 2)).any()


def test_an_old_leaf_pulled_into_the_band_has_only_its_midpoints_raytraced_again():
    fn, jac, kw, target = EQUIVALENCE["seam_fold"]
    mesh, _ = build(fn, jac, **kw)
    band = mesh.critical_band
    keep = np.arange(band.leaves.shape[0]) != 0
    used = np.unique(to_np(band.samples)[keep])
    renumber = np.full(band.lens.shape[0], -1)
    renumber[used] = np.arange(used.shape[0])
    doctored = mesh._replace(
        critical_band=CriticalBand(
            leaves=band.leaves[i64(np.flatnonzero(keep))],
            samples=i64(renumber[to_np(band.samples)[keep]]),
            lens=band.lens[i64(used)],
            source=band.source[i64(used)],
            det=band.det[i64(used)],
            measured_ij=band.measured_ij,
            measured_det=band.measured_det,
        )
    )
    lens_ = numpy_lens(fn, jac)
    got = extend_lens_mesh(doctored, lens_.raytrace, target)
    want, _ = build(fn, jac, **fresh_equivalent(kw, target))
    assert_same(got, want, rounding=ROUNDING)

    def as_complex(xy):
        return xy[:, 0] + 1j * xy[:, 1]  # lattice points are exact, so == is too

    asked = as_complex(np.concatenate(lens_.calls.raytrace))
    vertices = as_complex(to_np(mesh.vertices_lens))
    kept = as_complex(to_np(band.lens)[used])
    assert not np.isin(asked, np.concatenate((vertices, kept))).any()
    dropped = as_complex(
        to_np(band.lens)[np.setdiff1d(np.arange(band.lens.shape[0]), used)]
    )
    again = dropped[~np.isin(dropped, vertices)]
    assert again.shape[0] >= 1
    assert all((asked == p).sum() == 1 for p in again)


def test_extensions_chain():
    kw = dict(fov=4.0, init_res=8, min_img_sep=0.05)
    mesh, lens_ = build(localised_fold, localised_fold_jacobian, **kw)
    rt = lens_.raytrace
    twice = extend_lens_mesh(extend_lens_mesh(mesh, rt, 6.0), rt, 8.0)
    fresh, _ = build(
        localised_fold, localised_fold_jacobian, **fresh_equivalent(kw, 8.0)
    )
    assert_same(twice, fresh, rounding=ROUNDING)


def test_an_extension_carries_its_holes_and_traces_none_of_them_again():
    """One center lies outside the original fov, so its hole exists from the start."""
    c_in, c_out = (0.3001, -0.2003), (1.8001, 0.2003)
    fn, jac = sis([c_in], 0.6)
    kw = dict(fov=2.0, init_res=4, min_img_sep=0.05, centers=[c_in, c_out])
    mesh, lens_ = build(fn, jac, **kw)
    assert mesh.holes.radius.shape[0] == 2
    lens_.calls.raytrace.clear()
    got = extend_lens_mesh(mesh, lens_.raytrace, 4.0)
    want, _ = build(fn, jac, **fresh_equivalent(kw, 4.0))
    assert_same(got, want, rounding=("vertices_jacobian",))
    traced = called_at(lens_.calls, "raytrace")
    for c in (c_in, c_out):
        d = np.hypot(*(traced - np.array(c)).T)
        for r in (mesh.min_img_sep, mesh.min_img_sep / 4):
            assert not np.isclose(d, r, rtol=0, atol=1e-12).any()


def _in_any_triangle(points, triangles):
    """True where an integer point lies in or on some integer triangle."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    p = points[:, None, :]

    def cross(u, v, w):
        return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (
            v[..., 1] - u[..., 1]
        ) * (w[..., 0] - u[..., 0])

    d = np.stack((cross(a, b, p), cross(b, c, p), cross(c, a, p)), axis=-1)
    return ((d >= 0).all(axis=-1) | (d <= 0).all(axis=-1)).any(axis=1)


def test_an_extension_calls_the_lens_only_in_the_ring_and_in_old_leaves_it_splits():
    """No old vertex is sampled again, and every call strictly inside the old
    domain lies in an old leaf the extension split, which ``ring_fold(2.0)``
    makes it do."""
    fn, jac = ring_fold(2.0)
    mesh, lens_ = build(fn, jac, 4.0, 4, 0.05)
    lens_.calls.raytrace.clear()
    ext = extend_lens_mesh(mesh, lens_.raytrace, 6.0)
    lat = ext.lattice
    pad = lat.origin
    lo, hi = pad, pad + mesh.lattice.n
    ij = (
        np.rint(
            (called_at(lens_.calls, "raytrace") - to_np(lat.lo)) / lat.scale
        ).astype(np.int64)
        + lat.origin
    )
    old_vertices = to_np(mesh.vertices_ij) + pad
    assert not set(map(tuple, ij.tolist())) & set(map(tuple, old_vertices.tolist()))
    inner = ij[((ij > lo) & (ij < hi)).all(axis=1)]
    kept = {
        tuple(sorted(map(tuple, t)))
        for t in to_np(ext.vertices_ij)[to_np(ext.origin_leaves)].tolist()
    }
    split = np.array(
        [
            t
            for t in old_vertices[to_np(mesh.origin_leaves)].tolist()
            if tuple(sorted(map(tuple, t))) not in kept
        ]
    )
    assert inner.shape[0] > 0 and split.shape[0] > 0, "ring_fold must split old leaves"
    assert _in_any_triangle(inner, split).all()


@pytest.fixture
def affine_mesh():
    """``h0 = 0.5``: each ring adds ``1.0`` to the fov."""
    return build(affine, affine_jacobian, 4.0, 8, 0.05)


@pytest.mark.parametrize(
    "fov, want",
    [
        (5.9, 6.0),
        (6.0, 6.0),
        (6, 6.0),
        (float(np.nextafter(6.0, np.inf)), 6.0),
        (4.2, 5.0),
    ],
)
def test_the_requested_fov_rounds_up_to_whole_cells(affine_mesh, fov, want):
    mesh, lens_ = affine_mesh
    ext = extend_lens_mesh(mesh, lens_.raytrace, fov)
    assert ext.lattice.fov == want
    assert ext.lattice.init_res == round(want / 0.5)


@pytest.mark.parametrize("fov", [4.0, 4.0 - 1e-12, 3.9])
def test_a_fov_the_mesh_already_covers_returns_the_mesh_itself(affine_mesh, fov):
    mesh, lens_ = affine_mesh
    lens_.calls.raytrace.clear()
    lens_.calls.jacobian.clear()
    got = extend_lens_mesh(mesh, lens_.raytrace, fov)
    assert got is mesh
    assert not lens_.calls.raytrace and not lens_.calls.jacobian


def test_an_extension_int64_keys_cannot_hold_raises():
    """At finest level 25 the lattice keys in int64 up to ``init_res = 45``."""
    mesh, lens_ = build(affine, affine_jacobian, 1.0, 1, 1e-7)
    assert mesh.lattice.level - 1 == 25
    rt = lens_.raytrace
    assert extend_lens_mesh(mesh, rt, 45.0).lattice.init_res == 45
    with pytest.raises(ValueError, match="lattice too fine.*Extend by less"):
        extend_lens_mesh(mesh, rt, 46.0)


def _messages(record):
    """The ``UserWarning`` messages recorded, in order."""
    return [str(w.message) for w in record if issubclass(w.category, UserWarning)]


def test_an_extension_warns_depth_limited_as_the_fresh_build_would():
    kw = dict(fov=4.0, init_res=4, min_img_sep=0.05, max_depth=2)
    with pytest.warns(UserWarning, match="depth-limited"):
        mesh, lens_ = build(localised_fold, localised_fold_jacobian, **kw)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        extend_lens_mesh(mesh, lens_.raytrace, 6.0)
    with warnings.catch_warnings(record=True) as want:
        warnings.simplefilter("always")
        build(localised_fold, localised_fold_jacobian, **fresh_equivalent(kw, 6.0))
    assert any("depth-limited" in m for m in _messages(got))
    assert _messages(got) == _messages(want)


def test_an_extension_of_a_mesh_on_a_device_stays_on_it(device):
    kw = dict(fov=4.0, init_res=4, min_img_sep=0.05, device=device)
    mesh, lens_ = build(localised_fold, localised_fold_jacobian, **kw)
    got = extend_lens_mesh(mesh, lens_.raytrace, 6.0)
    want, _ = build(
        localised_fold, localised_fold_jacobian, **fresh_equivalent(kw, 6.0)
    )
    on = backend.device(backend.as_array([0.0], device=device))
    assert backend.device(got.vertices_lens) == on
    assert backend.device(got.leaves) == backend.device(want.leaves)
    assert_same(got, want, rounding=("vertices_jacobian",))


def test_batch_size_reaches_every_path_an_extension_traces():
    """``batch_size`` caps every lens call an extension makes -- the ring's, and
    the midpoints of the old leaves the balance splits, which `edge_bump`
    needs -- and the mesh still equals the fresh build."""
    kw = dict(fov=4.0, init_res=4, min_img_sep=0.05, batch_size=5)
    mesh, lens_ = build(edge_bump, edge_bump_jacobian, **kw)
    lens_.calls.raytrace.clear()
    lens_.calls.jacobian.clear()
    got = extend_lens_mesh(mesh, lens_.raytrace, 6.0, batch_size=5)
    assert (
        lens_.calls.raytrace
        and max(len(xy) for xy in lens_.calls.raytrace + lens_.calls.jacobian) <= 5
    )
    want, _ = build(edge_bump, edge_bump_jacobian, **fresh_equivalent(kw, 6.0))
    assert_same(got, want, rounding=("vertices_jacobian",))


def closed_build(fn, jac, fov, init_res, min_img_sep, **kw):
    """``build_closed_lens_mesh`` on a recording lens, with every warning kept."""
    lens_ = numpy_lens(fn, jac)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mesh, curves = build_closed_lens_mesh(
            lens_.raytrace,
            fov,
            init_res,
            min_img_sep,
            **kw,
        )
    return (
        to_mesh(mesh),
        to_mesh(curves),
        [str(w.message) for w in record],
        lens_,
    )


def fov_and_init_res(mesh):
    return mesh.lattice.fov, mesh.lattice.init_res


def test_the_fov_boundary_runs_counter_clockwise_along_leaf_edges_through_every_edge_vertex():
    """`edge_bump` refines the right side alone, so the sides hold unequal numbers of vertices."""
    mesh, _ = build(edge_bump, edge_bump_jacobian, 4.0, 4, 0.05)
    order = to_np(_fov_boundary(mesh))
    xy = to_np(mesh.vertices_lens)
    on_edge = np.flatnonzero((np.abs(xy[:, 0]) == 2.0) | (np.abs(xy[:, 1]) == 2.0))
    assert (xy[on_edge, 0] == 2.0).sum() > (xy[on_edge, 0] == -2.0).sum()
    assert sorted(order.tolist()) == on_edge.tolist()
    leaves = to_np(mesh.leaves)
    edges = {
        (min(a, b), max(a, b))
        for t in leaves.tolist()
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))
    }
    steps = zip(order.tolist(), np.roll(order, -1).tolist())
    assert all((min(a, b), max(a, b)) in edges for a, b in steps)
    x, y = xy[order, 0], xy[order, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area == 16.0


def test_a_curve_the_fov_cuts_is_grown_until_it_closes():
    """The tangential curve, radius ~1.18, crosses every side of the fov-2 domain.
    With ``h0 = 0.125`` and ``growth = 1.05`` each step adds one cell a side:
    fov 2 -> 2.25, still cut, -> 2.5, closed."""
    kw = dict(fov=2.0, init_res=16, min_img_sep=0.05)
    mesh, curves, messages, _ = closed_build(
        sie_like, sie_like_jacobian, growth=1.05, **kw
    )
    assert fov_and_init_res(mesh) == (2.5, 20)
    assert curves.closed.shape[0] > 0 and to_np(curves.closed).all()
    assert messages == []
    start, lens_ = build(sie_like, sie_like_jacobian, **kw)
    assert_same(
        mesh,
        extend_lens_mesh(start, lens_.raytrace, 2.5),
        rounding=("vertices_jacobian",),
    )
    assert_same(curves, critical_curves_and_caustics(mesh))


@pytest.mark.parametrize("x0, y0", [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)])
def test_a_curve_cut_by_any_one_side_of_the_fov_grows_it(x0, y0):
    """Off-center by 0.5, only the fov-3 domain's near side cuts the curve; the
    default ``growth`` asks for fov 4.5, two whole cells of ``h0 = 0.5`` a side."""
    mesh, curves, messages, _ = closed_build(
        sie_like, sie_like_jacobian, 3.0, 6, 0.05, x0=x0, y0=y0
    )
    assert fov_and_init_res(mesh) == (5.0, 10)
    assert to_np(curves.closed).all()
    assert messages == []


def test_a_mesh_whose_fov_cuts_no_curve_is_the_plain_build():
    kw = dict(fov=3.0, init_res=6, min_img_sep=0.05)
    mesh, curves, messages, _ = closed_build(sie_like, sie_like_jacobian, **kw)
    assert messages == []
    want, _ = build(sie_like, sie_like_jacobian, **kw)
    assert_same(mesh, want)
    assert_same(curves, critical_curves_and_caustics(want))


def test_a_curve_open_inside_the_fov_does_not_grow_it():
    """A non-finite strip across the tangential curve leaves it open inside the fov;
    growing cannot close it, so the fov stays, and nothing warns."""
    fn, jac = broken_where(
        sie_like,
        sie_like_jacobian,
        lambda p: (p[:, 0] > 0.9) & (p[:, 0] < 1.3) & (np.abs(p[:, 1]) < 0.2),
    )
    mesh, curves, messages, _ = closed_build(fn, jac, 3.0, 6, 0.05)
    assert not to_np(curves.closed).all()
    assert fov_and_init_res(mesh) == (3.0, 6)
    assert messages == []


@pytest.mark.parametrize("max_iters, fov, init_res", [(0, 2.0, 16), (1, 2.25, 18)])
def test_running_out_of_iterations_warns_and_returns_the_last_mesh(
    max_iters, fov, init_res
):
    mesh, curves, messages, _ = closed_build(
        sie_like, sie_like_jacobian, 2.0, 16, 0.05, growth=1.05, max_iters=max_iters
    )
    assert fov_and_init_res(mesh) == (fov, init_res)
    assert not to_np(curves.closed).all()
    assert len(messages) == 1 and "still cuts" in messages[0]
    assert_same(curves, critical_curves_and_caustics(mesh))


def test_build_options_reach_the_build_and_every_extension():
    """``batch_size`` caps every raytrace call of the build and its extensions,
    and the result equals the build extended by hand."""
    kw = dict(fov=2.0, init_res=4, min_img_sep=0.05, batch_size=5)
    mesh, _, messages, lens_ = closed_build(sie_like, sie_like_jacobian, **kw)
    assert messages == []
    assert max(len(xy) for xy in lens_.calls.raytrace) <= 5
    start, lens_ = build(sie_like, sie_like_jacobian, **kw)
    want = extend_lens_mesh(start, lens_.raytrace, 3.0, batch_size=5)
    assert_same(mesh, want)


def test_a_center_outside_the_fov_widens_it_before_the_build_and_warns():
    """The hole at x = 1.8001, radius 0.025, needs a half-width above 1.8251: two
    cells of ``h0 = 0.5`` a side. It happens before the build, even with ``max_iters = 0``.
    """
    c_out = (1.8001, 0.2003)
    fn, jac = sis([C_IN], 0.6)
    mesh, curves, messages, _ = closed_build(
        fn, jac, 2.0, 4, 0.05, centers=[C_IN, c_out], max_iters=0
    )
    assert fov_and_init_res(mesh) == (4.0, 8)
    assert len(messages) == 1
    assert "fov=4" in messages[0] and "init_res=8" in messages[0]
    assert to_np(curves.closed).all()
    want, _ = build(fn, jac, 4.0, 8, 0.05, centers=[C_IN, c_out])
    assert_same(mesh, want)


@pytest.mark.parametrize(
    "edge",
    [
        pytest.param([(0.9901, 0.2003)], id="center-inside-hole-across"),
        # Closer than twice the halved 0.05, so one hole: mean x 0.9601,
        # radius 0.025 + 0.0225, reaching 1.0076.
        pytest.param([(0.9601, 0.2003), (0.9601, 0.2453)], id="merged-pair"),
    ],
)
def test_a_hole_across_the_fov_edge_widens_it(edge):
    fn, jac = sis([C_IN], 0.6)
    mesh, _, messages, _ = closed_build(fn, jac, 2.0, 4, 0.05, centers=[C_IN, *edge])
    assert fov_and_init_res(mesh) == (3.0, 6)
    assert len(messages) == 1 and "init_res=6" in messages[0]


def test_holes_inside_the_fov_leave_it_alone():
    """The hole at x = 0.9601 reaches 0.9851 with the build's halved radius 0.025: inside.
    The SIS's boundary image stays 0.3999 or more from ``C_IN``, clear of its
    hole curve, the circle of radius ``0.3 - 0.025``."""
    fn, jac = sis([C_IN], 0.3)
    centers = [C_IN, (0.9601, 0.2003)]
    mesh, curves, messages, _ = closed_build(fn, jac, 2.0, 4, 0.05, centers=centers)
    assert messages == []
    assert to_np(curves.closed).all()
    want, _ = build(fn, jac, 2.0, 4, 0.05, centers=centers)
    assert_same(mesh, want)


def test_a_boundary_image_across_a_pseudo_caustic_grows_the_fov():
    """The SIS maps a point ``d`` from ``C_IN`` to ``d - 0.4`` from it, and its hole
    curve is the circle of radius ``0.4 - 0.025``. At fov 2 the critical curve,
    radius 0.4, is closed, but the right side's image comes within 0.2999 of
    ``C_IN`` and the far corner's lies 1.369 from it; at fov 3 the nearest is 0.7999."""
    fn, jac = sis([C_IN], 0.4)
    kw = dict(fov=2.0, init_res=4, min_img_sep=0.05, centers=[C_IN])
    mesh, curves, messages, _ = closed_build(fn, jac, **kw)
    assert fov_and_init_res(mesh) == (3.0, 6)
    assert curves.closed.shape[0] > 0 and to_np(curves.closed).all()
    assert messages == []
    start, lens_ = build(fn, jac, **kw)
    assert_same(mesh, extend_lens_mesh(start, lens_.raytrace, 3.0))


def test_the_crossing_test_finds_the_crossing_one_hole_segment_at_a_time(monkeypatch):
    monkeypatch.setattr("caustics.lenses.func.adaptive.lens_mesh.CROSSING_PAIRS", 1)
    fn, jac = sis([C_IN], 0.4)
    mesh, _, _, _ = closed_build(fn, jac, 2.0, 4, 0.05, centers=[C_IN])
    assert fov_and_init_res(mesh) == (3.0, 6)


def test_a_hole_curve_that_is_no_pseudo_caustic_grows_nothing():
    """A point mass's hole curve, a loop of radius ``0.15**2 / 0.025 - 0.025 = 0.875``
    about ``C_IN``, crosses the boundary's image, which runs from 0.668 to 1.756
    from it; its growth is -1, so it is no pseudo-caustic."""
    fn, jac = point_mass(C_IN, 0.15)
    mesh, _, messages, _ = closed_build(fn, jac, 2.0, 4, 0.05, centers=[C_IN])
    assert not to_np(mesh.holes.pseudo_caustic).any()
    assert fov_and_init_res(mesh) == (2.0, 4)
    assert messages == []


def test_running_out_of_iterations_with_a_pseudo_caustic_crossed_warns():
    fn, jac = sis([C_IN], 0.4)
    mesh, _, messages, _ = closed_build(
        fn, jac, 2.0, 4, 0.05, centers=[C_IN], max_iters=0
    )
    assert fov_and_init_res(mesh) == (2.0, 4)
    assert len(messages) == 1 and "pseudo-caustic" in messages[0]


def test_a_closed_build_warns_at_the_callers_line():
    lens_ = numpy_lens(sie_like, sie_like_jacobian)
    with pytest.warns(UserWarning, match="still cuts") as record:
        build_closed_lens_mesh(
            lens_.raytrace,
            2.0,
            16,
            0.05,
            growth=1.05,
            max_iters=0,
        )
    hits = [w for w in record if "still cuts" in str(w.message)]
    assert [w.filename for w in hits] == [__file__]


@pytest.mark.parametrize("n, batch_size", [(37, None), (37, 40), (3, 3)])
def test_the_sampler_hands_the_lens_padded_batches_and_returns_one_row_each(
    n, batch_size
):
    seen = []

    def raytrace(x, y):
        seen.append(x.shape[0])
        return 2.0 * x, 3.0 * y

    xy = np.random.default_rng(5).uniform(-1, 1, (n, 2))
    out = to_np(make_sampler(raytrace, None, batch_size)(f64(xy)))
    assert seen == [backend.padded_size(n, batch_size)]
    np.testing.assert_array_equal(out, np.column_stack([2 * xy[:, 0], 3 * xy[:, 1]]))


def _bowl(p):
    return p + 0.1 * p**2


def _bowl_jacobian(p):
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0 + 0.2 * p[:, 0]
    J[:, 1, 1] = 1.0 + 0.2 * p[:, 1]
    return J


def test_sigma_min_of_the_fitted_jacobian_drives_refinement():
    """The bowl converges at level 0; with its second image coordinate a billion
    times smaller, so that its fitted Jacobian's ``sigma_min`` is too while the
    first coordinate's deviation is unchanged, every leaf fails down to the
    finest level."""
    honest, _ = build(_bowl, _bowl_jacobian, init_res=2, min_img_sep=1.0, max_depth=3)
    assert (to_np(honest.origin_level) == 0).all()
    assert (to_np(honest.origin_status) == LEAF_CONVERGED).all()

    def starved_bowl(p):
        return _bowl(p) * np.array([1.0, 1e-9])

    def starved_jacobian(p):
        J = _bowl_jacobian(p)
        J[:, 1, :] *= 1e-9
        return J

    starved, _ = build(
        starved_bowl, starved_jacobian, init_res=2, min_img_sep=1.0, max_depth=3
    )
    finest = starved.lattice.level - 1
    assert finest > 0
    assert (to_np(starved.origin_level) == finest).all()
    assert (to_np(starved.origin_status) == LEAF_CONVERGENCE_FAILED).all()


def test_origin_deviation_and_sigma_min_match_their_samples_recomputed_independently():
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.1)
    rows = np.arange(to_np(mesh.origin_leaves).shape[0])
    xy = origin_samples(mesh, rows)
    beta6 = localised_fold(xy.reshape(-1, 2)).reshape(-1, 6, 2)
    _, dev, sig, _ = lens_status(
        f64(beta6),
        mesh.origin_cls,
        mesh.origin_level,
        mesh.lattice.h0,
        mesh.min_img_sep,
    )
    assert np.array_equal(to_np(mesh.origin_deviation), to_np(dev))
    assert np.array_equal(to_np(mesh.origin_sigma_min), to_np(sig))


def test_affine_error_is_below_sep_on_every_converged_origin(sie_mesh):
    mesh = sie_mesh
    error = to_np(affine_error(mesh.origin_deviation, mesh.origin_sigma_min))
    converged = to_np(mesh.origin_status) == LEAF_CONVERGED
    assert (error[converged] < mesh.min_img_sep).all()
    assert (error[~converged] >= mesh.min_img_sep).any()


def test_inside_fov_keeps_points_inside_and_on_the_boundary_and_drops_those_outside():
    mesh, _ = build(affine, affine_jacobian)
    theta = f64([[0.0, 0.0], [2.0, -2.0], [2.0, 0.5], [2.0 + 1e-12, 0.0], [0.0, -2.5]])
    assert to_np(inside_fov(mesh, theta)).tolist() == [True, True, True, False, False]


def test_inside_fov_image_is_inside_the_image_of_the_fov_boundary():
    """The fov ``[-2, 2]^2`` maps to the parallelogram ``AFFINE @ [-2, 2]^2``."""
    mesh, _ = build(affine, affine_jacobian)
    beta = f64([[0.0, 0.0], [1.0, 0.5], [5.0, 0.0], [np.nan, 0.0]])
    assert to_np(inside_fov_image(mesh, beta)).tolist() == [True, True, False, False]


def test_a_clockwise_image_of_the_fov_boundary_still_has_an_inside():
    """A reflecting lens maps the boundary clockwise: winding -1, still inside."""

    def flip(p):
        return p * np.array([1.0, -1.0])

    def flip_jacobian(p):
        return np.tile(np.diag([1.0, -1.0]), (p.shape[0], 1, 1))

    mesh, _ = build(flip, flip_jacobian)
    beta = f64([[0.5, 0.5], [3.0, 0.0]])
    assert to_np(inside_fov_image(mesh, beta)).tolist() == [True, False]


FITTED = [(sie_like, sie_like_jacobian), (localised_fold, localised_fold_jacobian)]


@pytest.mark.parametrize("fn, jac", FITTED)
def test_a_vertex_carries_the_mean_of_its_holders_fitted_jacobians(fn, jac):
    mesh, _ = build(fn, jac, min_img_sep=0.05)
    assert mesh.leaves.shape[0] > mesh.origin_leaves.shape[0]  # hanging nodes
    assert mesh.vertices_jacobian.dtype == mesh_backend.float64
    np.testing.assert_allclose(
        to_np(mesh.vertices_jacobian), holders_mean(mesh, fn), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("fn, jac", FITTED)
def test_no_two_holders_of_a_vertex_cover_one_sector(fn, jac):
    mesh, _ = build(fn, jac, min_img_sep=0.05)
    sector = to_np(CORNER_SECTOR)[to_np(mesh.origin_cls)]
    key = to_np(mesh.origin_leaves) * 6 + sector
    assert np.unique(key).size == key.size


def test_a_vertex_any_of_whose_holders_has_a_nonfinite_image_is_nan():
    fn, jac = broken_where(
        sie_like, sie_like_jacobian, lambda p: np.hypot(p[:, 0] - 0.7, p[:, 1]) < 0.1
    )
    mesh, _ = build(fn, jac, min_img_sep=0.05)
    bad_origin = (to_np(mesh.origin_status) & LEAF_RAYTRACE_NONFINITE) != 0
    bad = np.zeros(mesh.vertices_lens.shape[0], bool)
    bad[to_np(mesh.origin_leaves)[bad_origin].reshape(-1)] = True
    assert bad.any() and not bad.all()
    J = to_np(mesh.vertices_jacobian)
    assert np.isnan(J[bad]).all() and np.isfinite(J[~bad]).all()


def one_holder_fit(mesh, fn):
    """Each vertex's corner fit from the first origin, in ``mesh.origin_leaves`` order, that holds it, numpy ``(V, 2, 2)``."""
    flat = to_np(mesh.origin_leaves).reshape(-1)
    vertex, first = np.unique(flat, return_index=True)
    assert np.array_equal(vertex, np.arange(mesh.vertices_lens.shape[0]))
    row, corner = np.divmod(first, 3)
    beta6 = fn(origin_samples(mesh, row).reshape(-1, 2)).reshape(-1, 6, 2)
    A = fitted_jacobian(
        f64(beta6),
        mesh.origin_cls[i64(row)],
        mesh.origin_level[i64(row)],
        mesh.lattice.h0,
    )
    return to_np(A)[np.arange(row.shape[0]), corner]


# Twice the measured values, rounded up to one significant figure.
MEDIAN_BOUND = 3e-4  # measured median error 1.494e-4
MAX_BOUND = 0.4  # measured largest error 0.1843


def test_the_mean_is_closer_to_the_lens_than_one_holder_s_fit():
    mesh, _ = build(sie_like, sie_like_jacobian, min_img_sep=0.05)
    true = sie_like_jacobian(to_np(mesh.vertices_lens))
    err = np.abs(to_np(mesh.vertices_jacobian) - true).max(axis=(1, 2))
    one = np.abs(one_holder_fit(mesh, sie_like) - true).max(axis=(1, 2))
    assert np.median(err) < 0.6 * np.median(one)
    assert np.median(err) < MEDIAN_BOUND and err.max() < MAX_BOUND


def test_two_builds_give_the_same_vertex_jacobians_bit_for_bit():
    a, _ = build(sie_like, sie_like_jacobian, min_img_sep=0.05)
    b, _ = build(sie_like, sie_like_jacobian, min_img_sep=0.05)
    assert_same(a, b)
