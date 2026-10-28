"""forward_raytrace, and the queries and dedup it is built from."""

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Point
from caustics.lenses.func.adaptive.criterion import (
    LEAF_CONVERGED,
    LEAF_JACOBIAN_NONFINITE,
    LEAF_RAYTRACE_NONFINITE,
    affine_error,
)
from caustics.lenses.func.adaptive.geometry import area2, contains, triangle_weights
from caustics.lenses.func.adaptive.images import (
    dedup_representatives,
    forward_raytrace,
    mesh_query,
    mesh_seeds,
    near_seed,
)
from caustics.lenses.func.adaptive.index import index_hits
from caustics.lenses.func.adaptive.lens_mesh import build_lens_mesh, leaf_grow
from caustics.lenses.func.adaptive.curves import critical_curves_and_caustics

from adaptive_maps import (
    AFFINE,
    affine,
    affine_jacobian,
    build,
    f64,
    i64,
    localised_fold,
    localised_fold_jacobian,
    localised_fold_xy,
    point_mass,
    power_law_lens,
    SIE_LIKE_LENS,
    sis,
    SIS_LENS,
    to_np,
)

RNG = np.random.default_rng(20260904)


NONFINITE = LEAF_RAYTRACE_NONFINITE | LEAF_JACOBIAN_NONFINITE


def fr(mesh, beta, raytrace, **kw):
    """`forward_raytrace` at ``(B, 2)`` points, its images stacked ``(K, 2)``."""
    beta = backend.as_array(beta, dtype=backend.float64)
    x, y, counts = forward_raytrace(beta[:, 0], beta[:, 1], raytrace, mesh, **kw)
    return backend.stack((x, y), dim=-1), counts


def query_np(mesh, beta):
    beta = mesh_backend.as_array(
        np.asarray(beta, dtype=np.float64), dtype=mesh_backend.float64
    )
    idx, off, bary = mesh_query(mesh, beta)
    return to_np(idx), to_np(off), to_np(bary)


def leaf_status(mesh):
    return to_np(mesh.origin_status)[to_np(mesh.leaf_origin)]


BUILD = dict(fov=6.0, init_res=4, min_img_sep=0.1, max_depth=6)


def _missed(images, counts, want, tol):
    """Sources ``b`` none of whose returned images lies within ``tol`` of ``want[b]``."""
    images = to_np(images)
    off = np.concatenate(([0], np.cumsum(to_np(counts))))
    return [
        b
        for b, w in enumerate(np.asarray(want))
        if off[b + 1] == off[b]
        or np.linalg.norm(images[off[b] : off[b + 1]] - w, axis=-1).min() > tol
    ]


@pytest.fixture(scope="module")
def mesh():
    return to_mesh(
        build_lens_mesh(
            SIE_LIKE_LENS.raytrace, SIE_LIKE_LENS.jacobian_lens_equation, **BUILD
        )
    )


@pytest.fixture(scope="module")
def beta():
    rng = np.random.default_rng(21)
    return mesh_backend.as_array(
        rng.uniform(-1.5, 1.5, (64, 2)), dtype=mesh_backend.float64
    )


# The images of three sources under `_sie_build`'s SIE, found by the legacy
# triangle-subdivision `forward_raytrace` (commit b47823c) and deduplicated at
# 1e-2: a reference found without the adaptive mesh.
SIE_IMAGES = {
    (0.2, 0.2): [
        [-0.406354964, -0.322011113],
        [-0.104330279, 1.195786],
        [-0.000538395485, -0.000447346014],
        [0.924434245, 0.55215627],
        [1.08214688, -0.315099537],
    ],
    (0.05, -0.05): [
        [-0.673647225, -0.403054774],
        [-0.570067227, 0.834818602],
        [-5.65010305e-05, 3.58657526e-05],
        [0.606700599, 0.532241642],
        [0.696002543, -0.916224718],
    ],
    (1.4, 1.1): [[1.98252189, 1.65288591]],
}


def _sie_build(device=None):
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
    if device is not None:
        lens = lens.to(device)
    mesh = build_lens_mesh(
        lens.raytrace,
        lens.jacobian_lens_equation,
        fov=5.0,
        init_res=32,
        min_img_sep=1e-2,
        device=device,
    )
    return lens, to_mesh(mesh)


@pytest.fixture(scope="module")
def sie():
    return _sie_build()


def test_images_solve_the_lens_equation(mesh):
    beta = f64([[0.05, 0.02], [0.4, -0.3]])
    img, counts = fr(mesh, beta, SIE_LIKE_LENS.raytrace)
    _img_np = to_np(img)
    bx, by = SIE_LIKE_LENS.raytrace(img[:, 0], img[:, 1])
    got = np.stack((to_np(bx), to_np(by)), axis=-1)
    want = np.repeat(to_np(beta), to_np(counts), axis=0)
    assert np.abs(got - want).max() < 1e-6


def test_forward_raytrace_is_invariant_to_batch_size(mesh):
    beta = f64(np.random.default_rng(4).uniform(-1, 1, (12, 2)))
    whole = [to_np(t) for t in fr(mesh, beta, SIE_LIKE_LENS.raytrace)]
    for size in (1, 3, 100):
        got = [
            to_np(t) for t in fr(mesh, beta, SIE_LIKE_LENS.raytrace, batch_size=size)
        ]
        assert got[1].tolist() == whole[1].tolist()
        assert np.allclose(got[0], whole[0])


def test_forward_raytrace_hands_every_chunk_one_lm_map(mesh, monkeypatch):
    """One map per call, so a jitted solve compiles once per call, not per chunk."""
    import caustics.lenses.func.adaptive.images as images

    maps = []
    solve = images.batch_lm
    monkeypatch.setattr(
        images, "batch_lm", lambda X, Y, f, **kw: maps.append(f) or solve(X, Y, f, **kw)
    )
    beta = f64(np.random.default_rng(4).uniform(-1, 1, (12, 2)))
    fr(mesh, beta, SIE_LIKE_LENS.raytrace, batch_size=3)
    assert len(maps) > 1 and all(f is maps[0] for f in maps)


def test_forward_raytrace_raises_for_a_source_outside_the_image_of_the_fov_boundary(
    mesh,
):
    """A NaN source included."""
    with pytest.raises(ValueError, match="1 source position"):
        fr(mesh, f64([[1e6, 1e6]]), SIE_LIKE_LENS.raytrace)
    with pytest.raises(ValueError, match="1 source position"):
        fr(mesh, f64([[0.05, 0.02], [1e6, 1e6]]), SIE_LIKE_LENS.raytrace)
    with pytest.raises(ValueError, match="1 source position"):
        fr(mesh, f64([[0.05, 0.02], [1e6, 1e6]]), SIE_LIKE_LENS.raytrace, batch_size=1)
    with pytest.raises(ValueError, match="1 source position"):
        fr(mesh, f64([[np.nan, 0.0]]), SIE_LIKE_LENS.raytrace)


def test_every_sie_image_has_a_seed_within_min_img_sep(device):
    """Every image in :data:`SIE_IMAGES` has a seed (:func:`mesh_seeds`) within ``min_img_sep``.

    The central image included: for ``sp = [0.2, 0.2]`` and ``[0.05, -0.05]``
    it lies at radius ~7.0e-4 and ~6.7e-5, inside the softening radius
    ``s = 1e-3``, where the Jacobian is large but -- the radial critical curve
    running at r ~ 0.025 -- of one sign, so the leaves holding it converge.
    Measured, the worst seed lies 1.1e-3 from its image.
    """
    _, mesh = _sie_build(device)
    for sp, images in SIE_IMAGES.items():
        expected = np.asarray(images)
        idx, _, bary = mesh_query(mesh, f64([sp]))
        seed = to_np(mesh_seeds(mesh, idx, bary))
        nearest = np.linalg.norm(expected[:, None, :] - seed[None, :, :], axis=-1).min(
            axis=1
        )
        assert (
            nearest < 1e-2
        ).all(), f"{sp}: uncovered image, worst seed distance {nearest.max():.3e}"


def test_forward_raytrace_returns_exactly_the_sie_s_images(sie):
    """Every image comes back, the central one inside the softening radius
    ``s = 1e-3`` included, and nothing else: the residual and leaf-or-ball
    filters keep a stalled solve out. The reference is :data:`SIE_IMAGES`."""
    lens, mesh = sie
    for sp in ((0.2, 0.2), (0.05, -0.05)):
        images, counts = fr(mesh, f64([sp]), lens.raytrace)
        images = to_np(images)
        assert to_np(counts).tolist() == [images.shape[0]] and images.shape[0] > 0
        expected = np.asarray(SIE_IMAGES[sp])
        gap = np.linalg.norm(expected[:, None, :] - images[None, :, :], axis=-1)
        assert gap.min(axis=1).max() < 1e-2, f"{sp}: an image did not come back"
        assert (
            gap.min(axis=0).max() < 1e-2
        ), f"{sp}: an image that is not the SIE's came back"


def test_forward_raytrace_finds_every_sis_inner_image_beyond_min_img_sep():
    """Every inner image more than ``min_img_sep`` from the SIS center comes back.

    ``|theta_minus| = b - beta``, so a source at ``beta = b - k * min_img_sep``
    has its inner image ``k * min_img_sep`` from the center, ``k > 1`` here.
    Near the center a finest leaf's source image is a curved arc, and a source
    in the bulge between the arc and the leaf's straight edge is inside no
    triangle. Seeding only from converged leaves, with a radius of
    ``min_img_sep``, misses 56 of these 600. Images within ``min_img_sep`` of
    the center may or may not come back;
    nothing is asserted about them.
    """
    sep = 0.05
    mesh = build_lens_mesh(
        SIS_LENS.raytrace,
        SIS_LENS.jacobian_lens_equation,
        fov=4.0,
        init_res=4,
        min_img_sep=sep,
    )
    rng = np.random.default_rng(7)
    k = rng.uniform(1.0, 4.0, 600)
    phi = rng.uniform(0.0, 2 * np.pi, 600)
    u = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    images, counts = fr(mesh, (1.0 - k * sep)[:, None] * u, SIS_LENS.raytrace)
    missed = _missed(images, counts, -(k * sep)[:, None] * u, sep / 4)
    assert (
        not missed
    ), f"{len(missed)} of 600 inner images missed, at k = {np.round(k[missed], 3).tolist()[:10]}"


def test_forward_raytrace_finds_every_image_just_outside_a_center_steeper_than_isothermal():
    """Every image just outside the center's hole of a power law of slope ``t = 1.2`` comes back.

    Its deflection, ``r**-0.2``, diverges at the center, so the leaves there
    are non-finite and left out of the index. The leaves sharing a vertex
    with them have a large ``sigma_min``, so they converge although their
    source images bulge far past their straight edges -- on the side of the
    leaves left out, where no indexed triangle covers the bulge. The images
    lie ``0.5`` to ``1.5`` times ``min_img_sep`` from the center, just
    outside its hole. Growing only the leaves that did not converge misses
    9 of these 1000.
    """
    sep = 0.05
    lens_ = power_law_lens(1.2)
    mesh = build_lens_mesh(
        lens_.raytrace,
        lens_.jacobian_lens_equation,
        fov=8.0,
        init_res=8,
        min_img_sep=sep,
        centers=[[0.0, 0.0]],
    )
    rng = np.random.default_rng(3)
    r = rng.uniform(0.5, 1.5, 1000) * sep
    phi = rng.uniform(0.0, 2 * np.pi, 1000)
    theta = r[:, None] * np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    bx, by = lens_.raytrace(
        backend.as_array(theta[:, 0]), backend.as_array(theta[:, 1])
    )
    images, counts = fr(mesh, np.stack((to_np(bx), to_np(by)), axis=-1), lens_.raytrace)
    missed = _missed(images, counts, theta, sep / 4)
    assert (
        not missed
    ), f"{len(missed)} of 1000 images missed, at r / min_img_sep = {np.round(r[missed] / sep, 3).tolist()[:10]}"


def test_forward_raytrace_finds_every_sie_image_beyond_min_img_sep_of_a_critical_curve(
    sie,
):
    """Every image at least the mesh's ``min_img_sep`` from a critical curve comes back.

    Its fold partner is then at least the requested ``min_img_sep`` away.
    Built backwards: lens-plane points ``theta`` between ``1.25`` and ``3``
    times the mesh's ``min_img_sep`` from the vertices of the mesh's critical
    curves, which lie at most ``min_img_sep`` apart, so each ``theta`` is at
    least ``min_img_sep`` from the curve itself; their images are the sources.
    Seeding only from converged leaves, with a radius of ``min_img_sep``,
    misses 21 of these 3000.
    """
    lens_, mesh = sie
    sep = mesh.min_img_sep
    curve = to_np(critical_curves_and_caustics(mesh).lens)
    rng = np.random.default_rng(1)
    theta = curve[rng.integers(0, len(curve), 12000)] + rng.uniform(
        -3 * sep, 3 * sep, (12000, 2)
    )
    d = np.stack([np.linalg.norm(curve - t, axis=-1).min() for t in theta])
    theta = theta[(d >= 1.25 * sep) & (d <= 3 * sep)][:3000]
    assert theta.shape[0] == 3000
    bx, by = lens_.raytrace(
        backend.as_array(theta[:, 0]), backend.as_array(theta[:, 1])
    )
    images, counts = fr(mesh, np.stack((to_np(bx), to_np(by)), axis=-1), lens_.raytrace)
    missed = _missed(images, counts, theta, sep / 2)
    assert not missed, f"{len(missed)} of 3000 images missed"


def test_forward_raytrace_finds_one_image_of_a_source_below_the_fold_s_caustic():
    """Below the caustic ``beta_y = -0.34`` the fold has one image, ``y = beta_y / 0.6``.

    Every point of the fold strip ``|y| < 0.5`` maps at least as far from the
    source as the source lies below the caustic, so a stalled solve at the
    fold point is kept only for sources nearer the caustic than the default
    ``residual_tol``, ``1e-6``, and those are left out. Three sources sit
    just beyond that margin.
    """
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    tol = 1e-6
    rng = np.random.default_rng(5)
    bx = rng.uniform(-1.5, 1.5, 200)
    near = [2e-6, 1e-5, 1e-3]
    by = -0.34 - np.concatenate((rng.uniform(1.5 * tol, 0.2, 197), near))
    images, counts = fr(mesh, np.stack((bx, by), axis=-1), localised_fold_xy)
    assert to_np(counts).tolist() == [1] * 200
    assert np.allclose(to_np(images), np.stack((bx, by / 0.6), axis=-1), atol=1e-6)


def test_forward_raytrace_recovers_the_analytic_point_mass_pair():
    """The one case with a closed form: exactly two images, at known positions.

    ``theta = (b +- sqrt(b^2 + 4 Rein^2)) / 2``, both on the x axis. Asserting
    the cardinality is only defensible here, where it is a theorem rather than a
    measurement.
    """
    lens = Point(
        name="pt",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=1.0,
        s=1e-6,
    )
    mesh = build_lens_mesh(
        lens.raytrace,
        lens.jacobian_lens_equation,
        fov=8.0,
        init_res=64,
        min_img_sep=1e-2,
    )
    b = 0.4
    images, counts = fr(mesh, f64([[b, 0.0]]), lens.raytrace)
    images = to_np(images)
    assert to_np(counts).tolist() == [2], f"expected 2 images, got {images}"
    expected = np.sort([(b + np.sqrt(b**2 + 4)) / 2, (b - np.sqrt(b**2 + 4)) / 2])
    assert np.allclose(np.sort(images[:, 0]), expected, atol=1e-4)
    assert np.abs(images[:, 1]).max() < 1e-4


def test_forward_raytrace_batches_independently(sie):
    """A batched call equals looping one source at a time.

    Falsifiable against the two bugs the ragged layout invites: targets paired
    with the wrong seeds, and dedup merging images of different sources.
    """
    lens, mesh = sie
    points = [[0.2, 0.2], [0.05, -0.05], [0.4, -0.3]]
    images, counts = fr(mesh, f64(points), lens.raytrace)
    counts = to_np(counts)
    assert counts.shape == (3,)
    assert counts.sum() == to_np(images).shape[0]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for i, sp in enumerate(points):
        one, one_counts = fr(mesh, f64([sp]), lens.raytrace)
        assert to_np(one_counts).tolist() == [counts[i]], f"{sp}: count differs"
        block = to_np(images)[offsets[i] : offsets[i + 1]]
        assert np.allclose(block, to_np(one), atol=1e-8), f"{sp}: images differ"


def test_query_seeds_an_inner_image_that_runs_into_the_lens_center():
    """The inner image runs continuously into the SIS center as the source nears the cut: ``|theta_minus| = b - beta``.

    Hand-derived: at ``beta = 0.8`` and ``b = 1`` the images are
    ``theta = 1.8`` and ``theta = -0.2``, since ``1.8 * (1 - 1/1.8) = 0.8`` and
    ``-0.2 * (1 - 1/0.2) = 0.8``. The inner one lies by the non-finite center,
    where the leaves are finest; the query must still seed it.
    """
    mesh, _ = build(*sis(), min_img_sep=0.05)
    idx, offsets, bary = query_np(mesh, np.array([[0.8, 0.0]]))
    seed = to_np(
        mesh_seeds(mesh, mesh_backend.as_array(idx), mesh_backend.as_array(bary))
    )
    assert offsets.shape[0] == 2 and seed.shape[0] > 0
    for image in ([-0.2, 0.0], [1.8, 0.0]):
        gap = np.linalg.norm(seed - np.asarray(image), axis=1).min()
        assert gap <= 0.05, f"no seed within min_img_sep of {image}, closest {gap:.3g}"


def test_query_returns_the_leaves_around_a_lattice_point_where_det_a_is_exactly_zero():
    """A critical curve through a lattice point must not cost the leaves around it.

    The SIS's critical curve ``|theta| = 1`` runs through the level-0 vertex
    ``(1, 0)``, where ``A = diag(1, 0)`` and ``det A`` is exactly zero. Hand-derived:
    ``beta = (0.005, 0)`` has its images at ``theta = 1.005`` and
    ``theta = -0.995`` on the x axis, since ``1.005 * (1 - 1/1.005) = 0.005`` and
    ``-0.995 * (1 - 1/0.995) = 0.005``, both well inside the finest leaves that
    share ``(1, 0)`` and ``(-1, 0)``, whose legs are ``1/64``. The query must
    return a leaf containing each: the solve alone does not show it, since
    unconverged neighbours reach the images too.
    """
    mesh, _ = build(*sis(), min_img_sep=0.05)
    idx, _, _ = query_np(mesh, np.array([[0.005, 0.0]]))
    tri = mesh_backend.as_array(to_np(mesh.vertices_lens)[to_np(mesh.leaves)[idx]])
    for image in ([1.005, 0.0], [-0.995, 0.0]):
        at = mesh_backend.as_array(np.repeat([image], idx.shape[0], axis=0))
        held = to_np(contains(triangle_weights(tri, at)))
        assert held.any(), f"no queried leaf holds {image}"


def test_forward_raytrace_returns_the_converged_root_of_an_image_by_a_point_caustic():
    """The SIS's caustic is a point, so near it a residual says little about position.

    Hand-derived: ``beta = 1e-4 (cos 30, sin 30)`` has its images at
    ``theta = (1 + 1e-4) u`` and ``theta = (1e-4 - 1) u``, ``u`` the direction of
    ``beta``. There the tangential eigenvalue ``1 - 1/|theta|`` is about
    ``1e-4``, so a root up to ``1e-6 / 1e-4 = 1e-2`` from an image still passes
    ``residual_tol`` and joins that image's cluster, beside the converged root.
    """
    t, u = 1e-4, np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    mesh = build_lens_mesh(
        SIS_LENS.raytrace,
        SIS_LENS.jacobian_lens_equation,
        fov=4.0,
        init_res=8,
        min_img_sep=0.02,
    )
    images, counts = fr(mesh, f64([t * u]), SIS_LENS.raytrace)
    assert to_np(counts).tolist() == [2]
    for image in ((1 + t) * u, (t - 1) * u):
        gap = np.linalg.norm(to_np(images) - image, axis=1).min()
        assert gap < 1e-8, f"no image within 1e-8 of {image}, closest {gap:.3g}"


def test_query_hits_a_failing_leaf_within_its_deviation_at_its_nearest_point(mesh):
    """A point just outside a non-converged leaf's source triangle, within its
    deviation, hits the leaf, with the coordinates of the triangle's nearest point."""
    status = leaf_status(mesh)
    grow = to_np(leaf_grow(mesh))
    vs, leaves = to_np(mesh.vertices_source), to_np(mesh.leaves)
    tri = vs[leaves]
    edge = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    pick = np.flatnonzero(
        ((status & NONFINITE) == 0)
        & (status != LEAF_CONVERGED)
        & (grow > 0)
        & (edge > 0)
    )[:20]
    assert pick.size > 0, "fixture must have non-converged finite leaves"
    a, b, c = tri[pick, 0], tri[pick, 1], tri[pick, 2]
    mid = (a + b) / 2
    normal = np.stack(((b - a)[:, 1], -(b - a)[:, 0]), axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    inward = ((c - mid) * normal).sum(axis=1) > 0
    outward = np.where(inward[:, None], -normal, normal)
    points = mid + 0.5 * grow[pick][:, None] * outward
    idx, off, bary = query_np(mesh, points)
    for k, leaf in enumerate(pick):
        block = idx[off[k] : off[k + 1]]
        assert leaf in block, f"leaf {leaf} not hit by a point within its reach"
        j = off[k] + int(np.flatnonzero(block == leaf)[0])
        assert (bary[j] >= 0).all() and np.isclose(bary[j].sum(), 1.0)
        assert np.allclose(bary[j] @ tri[leaf], mid[k], rtol=0, atol=1e-9)


def test_query_returns_a_leaf_holding_each_point_next_to_the_leaves_left_out_of_the_index():
    """Every point of a leaf sharing a vertex with a non-finite leaf is queried back to a leaf holding it.

    Next to a point mass ``sigma_min`` is large, so the leaves around its
    non-finite center converge although their source images bulge far past
    their straight edges, into the images of the center's leaves, which are
    left out of the index. Forty points in each such leaf; growing only the
    leaves that did not converge returns no leaf holding 165 of these 720.
    """
    fn, jac = point_mass()
    mesh, _ = build(fn, jac, min_img_sep=0.2, centers=[[0.0, 0.0]])
    status = leaf_status(mesh)
    vl, leaves = to_np(mesh.vertices_lens), to_np(mesh.leaves)
    out = (status & NONFINITE) != 0
    ring = np.flatnonzero(~out & np.isin(leaves, leaves[out]).any(axis=1))
    assert (
        status[ring] == LEAF_CONVERGED
    ).any(), "fixture must converge next to the center"
    w = np.random.default_rng(8).dirichlet((1.0, 1.0, 1.0), (ring.size, 40))
    theta = np.einsum("lnk,lkd->lnd", w, vl[leaves[ring]]).reshape(-1, 2)
    idx, off, _ = query_np(mesh, fn(theta))
    owner = np.repeat(np.arange(theta.shape[0]), np.diff(off))
    held = to_np(
        contains(
            triangle_weights(
                mesh_backend.as_array(vl[leaves[idx]]),
                mesh_backend.as_array(theta[owner]),
            )
        )
    )
    unheld = np.setdiff1d(np.arange(theta.shape[0]), owner[held])
    assert (
        unheld.size == 0
    ), f"no queried leaf holds {unheld.size} of {theta.shape[0]} points"


def test_query_handles_empty_input_and_misses():
    mesh, _ = build(affine, affine_jacobian)
    idx, off, bary = query_np(mesh, np.zeros((0, 2)))
    assert off.tolist() == [0] and idx.shape == (0,) and bary.shape == (0, 3)
    far = np.array([[1e6, 1e6], [-1e6, 0.0]])
    idx, off, bary = query_np(mesh, far)
    assert off.tolist() == [0, 0, 0]


def test_query_is_invariant_to_point_order():
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(97, 2))
    ref = query_np(mesh, beta)
    perm = RNG.permutation(97)
    pidx, poff, pbary = query_np(mesh, beta[perm])
    for pos, old in enumerate(perm):
        assert np.array_equal(
            pidx[poff[pos] : poff[pos + 1]], ref[0][ref[1][old] : ref[1][old + 1]]
        )
        # `bary` too, not just the indices: a permutation-dependent bug that
        # scrambled coordinates while leaving leaf ids correct would otherwise
        # slip through this check.
        assert np.array_equal(
            pbary[poff[pos] : poff[pos + 1]], ref[2][ref[1][old] : ref[1][old + 1]]
        )


def test_query_finds_the_affine_preimage():
    mesh, _ = build(affine, affine_jacobian, fov=4.0, init_res=4, min_img_sep=0.25)
    lens_pts = RNG.uniform(-1.8, 1.8, size=(200, 2))
    beta = lens_pts @ AFFINE.T
    idx, off, bary = query_np(mesh, beta)
    assert (np.diff(off) >= 1).all(), "every interior point must hit a leaf"
    leaves = to_np(mesh.leaves)
    vl = to_np(mesh.vertices_lens)
    seed = np.einsum("kj,kjd->kd", bary, vl[leaves[idx]])
    first = seed[off[:-1]]
    assert np.allclose(first, lens_pts, atol=1e-9)


def test_bary_reconstructs_beta_on_every_hit_leaf():
    """Barycentric coordinates invert the source-plane map on every hit whose
    triangle contains the point; a widened hit's coordinates instead give the
    triangle's nearest point, within the leaf's deviation of beta."""
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(200, 2))
    idx, off, bary = query_np(mesh, beta)
    area = np.abs(to_np(area2(mesh.vertices_source[mesh.leaves])))[idx]
    vs = to_np(mesh.vertices_source)
    leaves = to_np(mesh.leaves)
    tri = vs[leaves[idx]]
    owner = np.repeat(np.arange(len(beta)), np.diff(off))
    assert idx.shape[0] > 0, "fixture returned no candidates"
    assert np.isfinite(bary).all() and (bary >= 0).all() and (bary <= 1).all()
    assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-12)
    recon = np.einsum("kj,kjd->kd", bary, tri)
    inside = to_np(
        contains(
            triangle_weights(
                mesh_backend.as_array(tri), mesh_backend.as_array(beta[owner])
            )
        )
    )
    good = area > 1e-10
    assert good[
        inside
    ].all(), (
        "fixture produced a degenerate contained leaf; the ~good branch needs writing"
    )
    assert np.allclose(recon[inside], beta[owner][inside], atol=1e-8)
    grow = to_np(leaf_grow(mesh))
    assert (
        np.linalg.norm(recon[~inside] - beta[owner][~inside], axis=1)
        <= grow[idx[~inside]] + 1e-12
    ).all()


def _dedup(points, counts, tol):
    """``dedup_representatives`` with every residual equal: each cluster keeps its earliest point."""
    residual2 = f64(np.zeros(points.shape[0]))
    return to_np(dedup_representatives(points, residual2, counts, tol))


def test_dedup_collapses_points_closer_than_the_tolerance():
    pts = f64([[0.0, 0.0], [0.05, 0.0], [1.0, 0.0]])
    keep = _dedup(pts, np.array([3]), 0.1)
    assert keep.tolist() == [True, False, True]


def test_dedup_keeps_the_point_of_smallest_residual_in_each_cluster():
    """Not the earliest: the earliest root of an image can be the least converged."""
    pts = f64([[0.0, 0.0], [0.005, 0.0], [1.0, 0.0], [1.005, 0.0]])
    residual2 = f64([4e-12, 1e-30, 1.0, 1.0])
    keep = to_np(dedup_representatives(pts, residual2, np.array([4]), 0.01))
    assert keep.tolist() == [False, True, True, False]


def test_dedup_keeps_points_separated_by_exactly_the_tolerance():
    pts = f64([[0.0, 0.0], [0.1, 0.0]])
    keep = _dedup(pts, np.array([2]), 0.1)
    assert keep.tolist() == [True, True]


def test_dedup_counts_connected_components_regardless_of_point_order():
    """Three collinear points spaced ``0.9 * tol`` apart form one connected
    component. Greedy returns 2 for the order below and 1 for ``[1, 0, 2]`` --
    the answer would depend on the order ``mesh_query`` happened to emit
    candidates in, which is not something a multiplicity map may depend on.
    """
    p = np.array([[0.0, 0.0], [0.009, 0.0], [0.018, 0.0]])
    counts = np.array([3])
    base = _dedup(f64(p), counts, 0.01).sum()
    assert base == 1, f"one chained component expected, got {base}"
    for order in ([1, 0, 2], [2, 1, 0], [0, 2, 1], [2, 0, 1]):
        got = _dedup(f64(p[order]), counts, 0.01).sum()
        assert got == base, f"order {order} gave {got}, not {base}"


def test_dedup_is_unchanged_by_bucketing_on_randomised_blocks():
    """The bucketed kernel must agree with a per-block reference exactly.

    Blocks are grouped by count and run at their own M rather than padded to
    the global maximum, so the risk is a scatter that puts one block's answer
    on another block's rows. Running each block *alone* through the same
    function is the independent reference: a single-block call has nothing to
    mis-scatter.

    What this does NOT check: both sides call the same clustering kernel
    (:func:`~caustics.lenses.func.adaptive.dedup_block_group` via
    :func:`~caustics.lenses.func.adaptive.dedup_representatives`), so a bug
    inside that kernel itself -- a wrong sentinel, a wrong iteration bound --
    reproduces identically on both sides and is invisible to this comparison.
    """
    rng = np.random.default_rng(20260910)
    # The all-empty vector is explicit: 20 random draws from this seed never
    # produce one, and it is the case that reaches the `total == 0` early exit.
    cases = [np.zeros(4, dtype=np.int64)]
    cases += [rng.integers(0, 6, size=rng.integers(1, 12)) for _ in range(20)]
    for counts in cases:
        pts = rng.normal(scale=0.01, size=(int(counts.sum()), 2)).reshape(-1, 2)
        res = f64(rng.random(int(counts.sum())))
        got = to_np(dedup_representatives(f64(pts), res, counts, 0.01))
        starts = np.cumsum(counts) - counts
        want = np.concatenate(
            [
                to_np(
                    dedup_representatives(
                        f64(pts[s : s + c]), res[s : s + c], np.array([c]), 0.01
                    )
                )
                for s, c in zip(starts, counts)
            ]
            + [np.zeros(0, dtype=bool)]
        )
        assert got.tolist() == want.tolist(), f"counts={counts.tolist()}"


def test_dedup_keeps_exactly_one_point_per_singleton_block():
    """Blocks of one bypass the clustering kernel; they must still be kept."""
    points = f64([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    keep = _dedup(points, np.array([1, 1, 1]), 0.01)
    assert keep.tolist() == [True, True, True]


def test_index_hits_are_the_query_hits_with_raw_weights(mesh, beta):
    grow = leaf_grow(mesh)
    qidx, tri, w = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, beta, grow)
    idx, off, _ = mesh_query(mesh, beta)
    counts = np.diff(to_np(off))
    assert np.array_equal(to_np(tri), to_np(idx))
    assert np.array_equal(to_np(qidx), np.repeat(np.arange(beta.shape[0]), counts))
    expected = triangle_weights(mesh.vertices_source[mesh.leaves[tri]], beta[qidx])
    assert np.array_equal(to_np(w), to_np(expected))


def test_forward_raytrace_reads_a_scalar_source_and_empty_sources(mesh):
    x, y, counts = forward_raytrace(0.05, 0.02, SIE_LIKE_LENS.raytrace, mesh)
    assert counts.shape == (1,) and int(to_np(counts)[0]) >= 1
    assert x.shape == y.shape == (int(to_np(counts)[0]),)
    x, y, counts = forward_raytrace(f64([]), f64([]), SIE_LIKE_LENS.raytrace, mesh)
    assert x.shape == y.shape == (0,) and counts.shape == (0,)


def test_near_seed_accepts_a_root_within_the_leaf_s_radius_and_rejects_one_beyond(mesh):
    """The radius is ``max(r, min_img_sep)``: ``r`` on a failing leaf whose ``r``
    is large, ``min_img_sep`` on a converged leaf. The roots lie outside their
    leaves, so the leaf test cannot pass them."""
    status = leaf_status(mesh)
    sep = mesh.min_img_sep
    r = to_np(affine_error(mesh.origin_deviation, mesh.origin_sigma_min))[
        to_np(mesh.leaf_origin)
    ]
    lens_tri = to_np(mesh.vertices_lens)[to_np(mesh.leaves)]
    diameter = np.linalg.norm(lens_tri - lens_tri[:, [1, 2, 0]], axis=-1).max(axis=1)
    finite = (status & NONFINITE) == 0
    wide = np.flatnonzero(
        finite & (status != LEAF_CONVERGED) & np.isfinite(r) & (r > 2 * sep)
    )
    # NOTE: the fixture's leaves are no smaller than its finest level allows,
    # so the selector is `diameter < sep`: no point of a triangle of diameter
    # `d` lies more than `2 d / 3` from its centroid, under `0.67 * sep`, so
    # the roots at `0.9` and `1.1` times the radius lie outside every
    # selected leaf.
    tight = np.flatnonzero((status == LEAF_CONVERGED) & (diameter < sep))
    assert wide.size and tight.size, "fixture must have both kinds of leaf"
    chosen = np.concatenate((wide[:10], tight[:10]))
    radius = np.maximum(r[chosen], sep)
    seed = lens_tri[chosen].mean(axis=1)

    def near(scale):
        root = seed + (scale * radius)[:, None] * np.array([1.0, 0.0])
        return to_np(near_seed(mesh, i64(chosen), f64(seed), f64(root)))

    assert near(0.9).all()
    assert not near(1.1).any()
