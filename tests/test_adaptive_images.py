"""forward_raytrace, and the queries and dedup it is built from."""

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Point
from caustics.lenses.func import forward_raytrace_rootfind
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
    index_cell_ranges,
    lens,
    localised_fold,
    localised_fold_jacobian,
    sis_jacobian,
    sis_raytrace,
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


def _sie_like(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    return x - 1.2 * x / r, y - 1.2 * y / r


def _sie_like_jacobian(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    k = 1.2 / r**3
    xx, yy, xy = 1.0 - 1.2 / r + k * x * x, 1.0 - 1.2 / r + k * y * y, k * x * y
    return backend.stack(
        (backend.stack((xx, xy), dim=-1), backend.stack((xy, yy), dim=-1)), dim=-2
    )


SIE_LIKE = lens(_sie_like, _sie_like_jacobian)


BUILD = dict(fov=6.0, init_res=4, min_img_sep=0.1, max_depth=6)


def _sis(x, y):
    """A singular isothermal sphere of Einstein radius 1, non-finite at the origin."""
    r = (x * x + y * y) ** 0.5
    return x - x / r, y - y / r


def _sis_jacobian(x, y):
    r = (x * x + y * y) ** 0.5
    k = 1.0 / r**3
    xx, yy, xy = 1.0 - 1.0 / r + k * x * x, 1.0 - 1.0 / r + k * y * y, k * x * y
    return backend.stack(
        (backend.stack((xx, xy), dim=-1), backend.stack((xy, yy), dim=-1)), dim=-2
    )


SIS = lens(_sis, _sis_jacobian)


def _fold(x, y):
    """``localised_fold`` on backend arrays: inside ``|y| < 0.5``, ``beta_y = 0.6 y + y**2 - 0.25``."""
    bend = backend.where(backend.abs(y) < 0.5, y * y - 0.25, 0.0 * y)
    return x, 0.6 * y + bend


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
        build_lens_mesh(SIE_LIKE.raytrace, SIE_LIKE.jacobian_lens_equation, **BUILD)
    )


@pytest.fixture(scope="module")
def beta():
    rng = np.random.default_rng(21)
    return mesh_backend.as_array(
        rng.uniform(-1.5, 1.5, (64, 2)), dtype=mesh_backend.float64
    )


def _beta(points):
    return mesh_backend.as_array(
        np.asarray(points, dtype=np.float64), dtype=mesh_backend.float64
    )


def dedup(points, tol):
    """Greedy clustering; returns one representative per cluster, sorted."""
    keep = []
    for p in points:
        if all(np.linalg.norm(p - q) >= tol for q in keep):
            keep.append(p)
    return np.array(sorted(keep, key=tuple)) if keep else np.zeros((0, 2))


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
    beta = _beta([[0.05, 0.02], [0.4, -0.3]])
    img, counts = fr(mesh, beta, _sie_like)
    _img_np = to_np(img)
    bx, by = _sie_like(img[:, 0], img[:, 1])
    got = np.stack((to_np(bx), to_np(by)), axis=-1)
    want = np.repeat(to_np(beta), to_np(counts), axis=0)
    assert np.abs(got - want).max() < 1e-6


def test_forward_raytrace_is_invariant_to_batch_size(mesh):
    beta = _beta(np.random.default_rng(4).uniform(-1, 1, (12, 2)))
    whole = [to_np(t) for t in fr(mesh, beta, _sie_like)]
    for size in (1, 3, 100):
        got = [to_np(t) for t in fr(mesh, beta, _sie_like, batch_size=size)]
        assert got[1].tolist() == whole[1].tolist()
        assert np.allclose(got[0], whole[0])


def test_forward_raytrace_raises_for_a_source_outside_the_image_of_the_fov_boundary(
    mesh,
):
    with pytest.raises(ValueError, match="1 source position"):
        fr(mesh, _beta([[1e6, 1e6]]), _sie_like)
    with pytest.raises(ValueError, match="1 source position"):
        fr(mesh, _beta([[0.05, 0.02], [1e6, 1e6]]), _sie_like)
    with pytest.raises(ValueError, match="1 source position"):
        fr(mesh, _beta([[0.05, 0.02], [1e6, 1e6]]), _sie_like, batch_size=1)


def test_forward_raytrace_raises_for_a_nan_source(mesh):
    with pytest.raises(ValueError, match="1 source position"):
        fr(mesh, _beta([[np.nan, 0.0]]), _sie_like)


def test_forward_raytrace_handles_empty_input(mesh):
    img, counts = fr(
        mesh,
        mesh_backend.as_array(np.zeros((0, 2)), dtype=mesh_backend.float64),
        _sie_like,
    )
    assert to_np(counts).size == 0
    assert to_np(img).shape == (0, 2)


def test_sie_candidates_recover_forward_raytrace_images(device):
    """Spec test 26, split into the two contracts this module actually owns.

    1. **Coverage** -- every image ``forward_raytrace`` finds has a candidate
       seed within ``min_img_sep``. That is exactly what :func:`mesh_seeds`
       promises: the hit leaf's own affine map is the one step 7 bounds, so
       the seed is accurate to ``min_img_sep`` by construction. Measured
       across the images at these three source points, the worst distance is
       1.1e-3 against a 1e-2 tolerance -- roughly ten times better than the
       guarantee.
    2. **No spurious images** -- every candidate the root-finder converges on
       is a genuine image.

    Coverage includes the central image. For ``sp = [0.2, 0.2]`` and
    ``sp = [0.05, -0.05]`` this SIE has one at radius ~7.0e-4 and ~6.7e-5
    respectively, *inside* its own softening radius ``s = 1e-3``, where the
    Jacobian is large but -- the radial critical curve running at r ~ 0.025 --
    of one sign. The frozen oracle's quadratic-vertex parity check read the
    core's curvature as a fold and condemned the leaves holding those images,
    leaving no seed within 0.52 and 0.78 arcsec of them, so this test used to
    exclude everything inside the softening radius. The Jacobian test finds
    no sign change there, the leaves converge, and the two central images now
    have seeds 8.4e-4 and 1.1e-4 away -- so Contract 1 covers every image, and
    fails loudly should the core ever be condemned again.
    """
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
    ).to(device)
    mesh = to_mesh(
        build_lens_mesh(
            lens.raytrace,
            lens.jacobian_lens_equation,
            fov=5.0,
            init_res=32,
            min_img_sep=1e-2,
            device=device,
        )
    )
    for sp in ([0.2, 0.2], [0.05, -0.05], [1.4, 1.1]):
        sx = backend.as_array(sp[0], device=device)
        sy = backend.as_array(sp[1], device=device)
        ex, ey = lens.forward_raytrace(sx, sy)
        expected = dedup(np.stack([to_np(ex), to_np(ey)], axis=-1), 1e-2)
        assert expected.shape[0] > 0, f"{sp}: no reference images"
        idx, offsets, bary = mesh_query(mesh, mesh_backend.as_array(np.asarray([sp])))
        seed = to_np(mesh_seeds(mesh, idx, bary))
        assert seed.shape[0] >= expected.shape[0], "candidates must cover the images"
        refined = forward_raytrace_rootfind(
            backend.as_array(seed[:, 0], device=device),
            backend.as_array(seed[:, 1], device=device),
            sx,
            sy,
            lens.raytrace,
        )
        refined = to_np(refined)
        bx, by = lens.raytrace(
            backend.as_array(refined[:, 0], device=device),
            backend.as_array(refined[:, 1], device=device),
        )
        residual = np.linalg.norm(
            np.stack([to_np(bx), to_np(by)], -1) - np.asarray(sp),
            axis=-1,
        )
        got = dedup(refined[residual < 1e-3], 1e-2)
        # Contract 1: coverage. Every image, the central one included, has a
        # seed within min_img_sep.
        nearest = np.linalg.norm(expected[:, None, :] - seed[None, :, :], axis=-1).min(
            axis=1
        )
        assert (
            nearest < 1e-2
        ).all(), f"{sp}: uncovered image, worst seed distance {nearest.max():.3e}"
        # Contract 2: no spurious images among those the root-finder converged on.
        assert got.shape[0] > 0, f"{sp}: nothing converged"
        for p in got:
            assert (
                np.linalg.norm(expected - p, axis=-1).min() < 1e-2
            ), f"{sp}: converged to {p}, which is not a forward_raytrace image"


def test_build_and_query_run_on_the_configured_device(device):
    """Build and query complete on the configured device and return sane CSR.

    Note what this does **not** verify: every array is converted through
    ``backend.to_numpy`` before inspection, so this test cannot distinguish
    "computed on the requested device" from "computed elsewhere and converted
    back". It pins that the pipeline runs end to end under the ``device``
    fixture and returns coherent results, not placement.
    """
    lens = SIE(
        name="sie",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        q=0.7,
        phi=0.0,
        Rein=1.0,
        s=1e-3,
    ).to(device)
    mesh = build_lens_mesh(
        lens.raytrace,
        lens.jacobian_lens_equation,
        fov=4.0,
        init_res=8,
        min_img_sep=0.1,
        device=device,
    )
    idx, off, bary = mesh_query(
        to_mesh(mesh), mesh_backend.as_array(np.array([[0.1, 0.1], [3.0, 3.0]]))
    )
    off_np = to_np(off)
    bary_np = to_np(bary)
    assert off_np.shape == (3,)
    # The hit/miss pair is what makes this falsifiable. `off.shape` is
    # `(B + 1,)` for any B by the CSR contract, and `np.isfinite` is vacuously
    # True on an empty array, so shape-plus-finiteness alone would pass even if
    # the query silently returned nothing for both points. Measured: [0, 3, 3].
    assert off_np[1] > off_np[0], "the interior source point must hit a leaf"
    assert off_np[2] == off_np[1], "the far exterior point must hit nothing"
    assert bary_np.shape[0] == off_np[-1], "bary rows must match the CSR total"
    assert np.isfinite(bary_np).all()


def test_forward_raytrace_finds_no_spurious_sie_images(sie):
    """Every returned image is an image `lens.forward_raytrace` also finds.

    The residual and leaf-or-ball filters keep a stalled Levenberg-Marquardt
    solve from being returned: the ball is `min_img_sep` on a converged leaf,
    and on a leaf that did not converge it is `max(r, min_img_sep)`, which
    can be large. These sources lie away from the caustics.
    """
    lens, mesh = sie
    for sp in ([0.2, 0.2], [0.05, -0.05]):
        images, counts = fr(mesh, _beta([sp]), lens.raytrace)
        images = to_np(images)
        assert to_np(counts).tolist() == [images.shape[0]]
        assert images.shape[0] > 0, f"{sp}: no images found"
        # The reference path is float32-only: `LensBase.forward_raytrace` raises
        # "expected scalar type Float but found Double" on float64 input. The mesh
        # itself is float64, so only this comparison call is narrowed.
        ex, ey = lens.forward_raytrace(backend.as_array(sp[0]), backend.as_array(sp[1]))
        expected = dedup(np.stack([to_np(ex), to_np(ey)], axis=-1), 1e-2)
        for p in images:
            assert (
                np.linalg.norm(expected - p, axis=-1).min() < 1e-2
            ), f"{sp}: returned {p}, which is not a forward_raytrace image"


def test_forward_raytrace_covers_every_sie_image(sie):
    """The converse contract: no image is dropped by the filters or the dedup.

    Every image, including the central one inside the softening radius
    ``s = 1e-3`` (radius 7.0e-4 and 6.7e-5 for the two source points below).
    The frozen oracle's criterion condemned the core leaves holding it, so
    this test used to be scoped to the images outside that radius; under the
    Jacobian test those leaves converge (see
    ``test_sie_candidates_recover_forward_raytrace_images`` for the full
    derivation), and all five images of each source come back. Measured,
    every image is covered to within 4.4e-5.
    """
    lens, mesh = sie
    for sp in ([0.2, 0.2], [0.05, -0.05]):
        images, _ = fr(mesh, _beta([sp]), lens.raytrace)
        images = to_np(images)
        # The reference path is float32-only: `LensBase.forward_raytrace` raises
        # "expected scalar type Float but found Double" on float64 input. The mesh
        # itself is float64, so only this comparison call is narrowed.
        ex, ey = lens.forward_raytrace(backend.as_array(sp[0]), backend.as_array(sp[1]))
        expected = dedup(np.stack([to_np(ex), to_np(ey)], axis=-1), 1e-2)
        assert expected.shape[0] > 0, f"{sp}: reference found no images"
        nearest = np.linalg.norm(expected[:, None, :] - images[None, :, :], axis=-1)
        worst = nearest.min(axis=1).max()
        assert worst < 1e-2, f"{sp}: uncovered image, worst distance {worst:.3e}"


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
        SIS.raytrace, SIS.jacobian_lens_equation, fov=4.0, init_res=4, min_img_sep=sep
    )
    rng = np.random.default_rng(7)
    k = rng.uniform(1.0, 4.0, 600)
    phi = rng.uniform(0.0, 2 * np.pi, 600)
    u = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    images, counts = fr(mesh, (1.0 - k * sep)[:, None] * u, _sis)
    missed = _missed(images, counts, -(k * sep)[:, None] * u, sep / 4)
    assert (
        not missed
    ), f"{len(missed)} of 600 inner images missed, at k = {np.round(k[missed], 3).tolist()[:10]}"


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
    images, counts = fr(mesh, np.stack((bx, by), axis=-1), _fold)
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
    images, counts = fr(mesh, _beta([[b, 0.0]]), lens.raytrace)
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
    images, counts = fr(mesh, _beta(points), lens.raytrace)
    counts = to_np(counts)
    assert counts.shape == (3,)
    assert counts.sum() == to_np(images).shape[0]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for i, sp in enumerate(points):
        one, one_counts = fr(mesh, _beta([sp]), lens.raytrace)
        assert to_np(one_counts).tolist() == [counts[i]], f"{sp}: count differs"
        block = to_np(images)[offsets[i] : offsets[i + 1]]
        assert np.allclose(block, to_np(one), atol=1e-8), f"{sp}: images differ"


def test_query_seeds_an_inner_image_that_runs_into_the_lens_center():
    """The coverage the old terminate-on-non-finite policy destroyed.

    For the SIS the inner image runs continuously into the lens center as the
    source approaches the cut: ``|theta_minus| = b - beta``. Terminating the six
    level-0 triangles that share the origin therefore removed a hexagon of
    half-width ``fov / init_res`` from the spatial index -- 1.0 arcsec on this
    fixture -- and with it the seed for every inner image inside it, exactly
    where a grid-and-Newton forward_raytrace is already weakest.

    Hand-derived, not read off the mesh: at ``beta = 0.8`` and ``b = 1`` the two
    images are ``theta = 1.8`` and ``theta = -0.2``, since
    ``1.8 * (1 - 1/1.8) = 0.8`` and ``-0.2 * (1 - 1/0.2) = 0.8``. The inner one
    sits 5x deeper inside the old hexagon than its half-width, so the old policy
    returns only the outer seed, 2.0 arcsec away.

    Compose the public ``mesh_query`` and ``mesh_seeds`` interfaces so this
    regression exercises the same seeding path used by forward raytracing.
    """
    mesh, _ = build(sis_raytrace, sis_jacobian, min_img_sep=0.05)
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
    mesh, _ = build(sis_raytrace, sis_jacobian, min_img_sep=0.05)
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
        SIS.raytrace, SIS.jacobian_lens_equation, fov=4.0, init_res=8, min_img_sep=0.02
    )
    images, counts = fr(mesh, _beta([t * u]), _sis)
    assert to_np(counts).tolist() == [2]
    for image in ((1 + t) * u, (t - 1) * u):
        gap = np.linalg.norm(to_np(images) - image, axis=1).min()
        assert gap < 1e-8, f"no image within 1e-8 of {image}, closest {gap:.3g}"


def test_query_covers_points_on_the_source_bbox_upper_edge():
    """Regression: the upper bbox edge used to return zero candidates.

    `cell = span / [nx, ny]`, so a point at `x == hi_x` yields `u_x == nx`. The
    old cell-index containment test rejected it, while `build_index` clips leaf
    registration to `nx - 1` -- so leaves whose AABB reaches `hi` were indexed
    but unreachable. Measured before the fix: 18 of 18 upper-edge vertices
    returned nothing where brute-force containment found candidates.
    """
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    vs = to_np(mesh.vertices_source)
    leaves = to_np(mesh.leaves)
    status = leaf_status(mesh)
    hi = to_np(mesh.index.hi)
    on_edge = np.flatnonzero((vs[:, 0] == hi[0]) | (vs[:, 1] == hi[1]))
    assert on_edge.size > 0, "fixture must have vertices on the upper bbox edge"
    for v in on_edge:
        beta = vs[v]
        tri = mesh_backend.as_array(vs[leaves])
        pts = mesh_backend.as_array(np.repeat(beta[None], leaves.shape[0], axis=0))
        truth = to_np(contains(triangle_weights(tri, pts)))
        # Every leaf with a finite raytrace and Jacobian is indexed (see
        # `LensMesh`'s docstring), converged or not.
        expected = set(np.flatnonzero(truth & ((status & NONFINITE) == 0)).tolist())
        idx, off, _ = query_np(mesh, beta[None])
        assert (
            set(idx[off[0] : off[1]].tolist()) >= expected
        ), f"upper-edge point {beta} lost candidates"


def test_query_matches_brute_force_containment_on_multi_cell_leaves():
    """The completeness claim, on a mesh whose leaves span several cells.

    `build_index` registers each leaf in every cell its box covers at its
    level, not just its three vertex cells -- and no other test distinguishes
    those, since the vertex-cell test checks only vertices and the crack test's
    uniform reference shares `build_index` so a common bug cancels. On this
    fixture every one of the 5,349 finite leaves spans three or more cells on
    an axis at its level.
    """
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    vs = to_np(mesh.vertices_source)
    leaves = to_np(mesh.leaves)
    status = leaf_status(mesh)
    tri = vs[leaves]
    finite = (status & NONFINITE) == 0
    grow = to_np(leaf_grow(mesh.origin_status, mesh.origin_deviation, mesh.leaf_origin))
    _, i0, i1 = index_cell_ranges(mesh.index, tri[finite], grow[finite])
    span = i1 - i0 + 1
    assert (span >= 3).any(), "fixture must contain leaves spanning several cells"
    beta = RNG.uniform(-0.9, 0.9, size=(200, 2))
    idx, off, _ = query_np(mesh, beta)
    tri_b = mesh_backend.as_array(tri)
    for b in range(beta.shape[0]):
        pts = mesh_backend.as_array(np.repeat(beta[b][None], leaves.shape[0], axis=0))
        truth = to_np(contains(triangle_weights(tri_b, pts)))
        # Every leaf with a finite raytrace and Jacobian is indexed (see
        # `LensMesh`'s docstring), converged or not.
        expected = set(np.flatnonzero(truth & ((status & NONFINITE) == 0)).tolist())
        assert set(idx[off[b] : off[b + 1]].tolist()) >= expected


def test_query_hits_a_failing_leaf_within_its_deviation_at_its_nearest_point(mesh):
    """A point just outside a non-converged leaf's source triangle, within its
    deviation, hits the leaf, with the coordinates of the triangle's nearest point."""
    status = leaf_status(mesh)
    grow = to_np(leaf_grow(mesh.origin_status, mesh.origin_deviation, mesh.leaf_origin))
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


def test_query_csr_is_well_formed_on_a_folded_mesh():
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    beta = RNG.uniform(-2.5, 2.5, size=(64, 2))
    idx, off, bary = query_np(mesh, beta)
    assert off.shape == (65,) and off[0] == 0 and off[-1] == idx.shape[0]
    assert (np.diff(off) >= 0).all()
    assert bary.shape == (idx.shape[0], 3)
    for b in range(64):
        block = idx[off[b] : off[b + 1]]
        assert (np.diff(block) > 0).all(), "blocks must be strictly ascending"


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


def test_bary_is_in_the_simplex_on_every_leaf():
    """The simplex guarantee end to end, on a curved mesh."""
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    beta = RNG.uniform(-0.4, 0.4, size=(40, 2))
    idx, off, bary = query_np(mesh, beta)
    assert idx.shape[0] > 0, "fixture returned no candidates"
    assert np.isfinite(bary).all()
    assert (bary >= 0).all() and (bary <= 1).all()
    assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-12)


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
    grow = to_np(leaf_grow(mesh.origin_status, mesh.origin_deviation, mesh.leaf_origin))
    assert (
        np.linalg.norm(recon[~inside] - beta[owner][~inside], axis=1)
        <= grow[idx[~inside]] + 1e-12
    ).all()


def _pts(x):
    return mesh_backend.as_array(
        np.asarray(x, dtype=np.float64), dtype=mesh_backend.float64
    )


def _dedup(points, counts, tol):
    """``dedup_representatives`` with every residual equal: each cluster keeps its earliest point."""
    residual2 = _pts(np.zeros(points.shape[0]))
    return to_np(dedup_representatives(points, residual2, counts, tol))


def test_dedup_collapses_points_closer_than_the_tolerance():
    pts = _pts([[0.0, 0.0], [0.05, 0.0], [1.0, 0.0]])
    keep = _dedup(pts, np.array([3]), 0.1)
    assert keep.tolist() == [True, False, True]


def test_dedup_keeps_the_point_of_smallest_residual_in_each_cluster():
    """Not the earliest: the earliest root of an image can be the least converged."""
    pts = _pts([[0.0, 0.0], [0.005, 0.0], [1.0, 0.0], [1.005, 0.0]])
    residual2 = _pts([4e-12, 1e-30, 1.0, 1.0])
    keep = to_np(dedup_representatives(pts, residual2, np.array([4]), 0.01))
    assert keep.tolist() == [False, True, True, False]


def test_dedup_keeps_points_separated_by_exactly_the_tolerance():
    pts = _pts([[0.0, 0.0], [0.1, 0.0]])
    keep = _dedup(pts, np.array([2]), 0.1)
    assert keep.tolist() == [True, True]


def _cross2(u, v):
    """Scalar cross product of 2-D vectors.

    ``np.cross`` on 2-vectors is deprecated in NumPy 2.0 and emits a
    ``DeprecationWarning`` per call. This is the same value, computed
    component-wise, and is bit-identical to the ``np.cross`` result.
    """
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def test_seeds_lie_inside_their_lens_triangle_on_a_folded_mesh():
    """RENAMED from ``test_seeds_lie_inside_their_lens_triangle``: this checks
    the stronger inside-the-triangle sign condition on a folded, multi-level
    mesh, distinct from the bounding-box check the same-named test above
    (added verbatim from the Task 14 brief) performs on the module-level
    ``mesh``/``beta`` fixtures.

    ADAPTED: the oracle's ``Mesh.seeds`` accepted ``beta`` (query-and-seed) or
    ``leaf_indices``/``bary`` (gather-only) behind a ``_check_call`` guard, so
    the legacy test compared the two modes against each other and also
    checked the CSR ``offsets`` the ``beta`` mode returned. ``mesh_seeds`` has
    only the gather-only form, so that comparison has no counterpart and is
    dropped; ``triangles_lens(leaf_indices=idx)`` (also dropped) is replaced
    by the same manual ``vertices_lens[leaves[idx]]`` gather ``mesh_seeds``
    itself performs.
    """
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(50, 2))
    idx, _, bary = mesh_query(mesh, beta)
    seed = to_np(mesh_seeds(mesh, idx, bary))
    assert seed.shape[0] > 0, "fixture returned no candidates"
    idx_np = to_np(idx)
    leaves = to_np(mesh.leaves)
    vl = to_np(mesh.vertices_lens)
    tri = vl[leaves[idx_np]]
    for k in range(len(seed)):
        w = np.array(
            [
                _cross2(tri[k, (i + 1) % 3] - seed[k], tri[k, (i + 2) % 3] - seed[k])
                for i in range(3)
            ]
        )
        assert (w >= -1e-9).all() or (w <= 1e-9).all()


def test_dedup_counts_connected_components_regardless_of_point_order():
    """RENAMED from ``test_dedup_counts_connected_components_not_greedy_clusters``:
    this checks invariance across four explicit permutations, distinct from
    the single-order check in the same-named test above (added verbatim from
    the Task 14 brief).

    Three collinear points spaced ``0.9 * tol`` apart form one connected
    component. Greedy returns 2 for the order below and 1 for ``[1, 0, 2]`` --
    the answer would depend on the order ``mesh_query`` happened to emit
    candidates in, which is not something a multiplicity map may depend on.
    """
    p = np.array([[0.0, 0.0], [0.009, 0.0], [0.018, 0.0]])
    counts = np.array([3])
    base = _dedup(_pts(p), counts, 0.01).sum()
    assert base == 1, f"one chained component expected, got {base}"
    for order in ([1, 0, 2], [2, 1, 0], [0, 2, 1], [2, 0, 1]):
        got = _dedup(_pts(p[order]), counts, 0.01).sum()
        assert got == base, f"order {order} gave {got}, not {base}"


def test_dedup_keeps_identical_points_in_different_blocks_distinct():
    """RENAMED from ``test_dedup_never_merges_across_blocks`` to avoid
    shadowing the same-named test above (added verbatim from the Task 14
    brief); both exercise the same contract on equivalent input.

    Two source points whose images coincide must not collapse into one. The
    padded ``(B, M, M)`` formulation makes cross-block bleed the natural bug
    here, and it would silently halve a multiplicity map.
    """
    points = _pts([[0.0, 0.0], [0.0, 0.0]])
    keep = _dedup(points, np.array([1, 1]), 0.01)
    assert keep.sum() == 2, "identical points in different blocks are distinct"


def test_dedup_handles_ragged_blocks_and_empty_blocks():
    """Padding must not invent images in a block that found none."""
    points = _pts([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0005]])
    keep = _dedup(points, np.array([1, 0, 2]), 0.01)
    assert keep.tolist() == [True, True, False]


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
        res = _pts(rng.random(int(counts.sum())))
        got = to_np(dedup_representatives(_pts(pts), res, counts, 0.01))
        starts = np.cumsum(counts) - counts
        want = np.concatenate(
            [
                to_np(
                    dedup_representatives(
                        _pts(pts[s : s + c]), res[s : s + c], np.array([c]), 0.01
                    )
                )
                for s, c in zip(starts, counts)
            ]
            + [np.zeros(0, dtype=bool)]
        )
        assert got.tolist() == want.tolist(), f"counts={counts.tolist()}"


def test_dedup_keeps_exactly_one_point_per_singleton_block():
    """Blocks of one bypass the clustering kernel; they must still be kept."""
    points = _pts([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    keep = _dedup(points, np.array([1, 1, 1]), 0.01)
    assert keep.tolist() == [True, True, True]


def test_dedup_mixes_singleton_and_clustered_blocks_in_order():
    """The bypass and the kernel write into one output; order must survive.

    Block 0 is a singleton, block 1 collapses to one image, block 2 is a
    singleton again. A scatter that appends the bypassed blocks after the
    clustered ones would pass every count-based assertion and still return the
    representatives in the wrong rows.
    """
    points = _pts([[9.0, 9.0], [0.0, 0.0], [0.0, 0.001], [5.0, 5.0]])
    keep = _dedup(points, np.array([1, 2, 1]), 0.01)
    assert keep.tolist() == [True, True, False, True]


def test_index_hits_are_the_query_hits_with_raw_weights(mesh, beta):
    grow = leaf_grow(mesh.origin_status, mesh.origin_deviation, mesh.leaf_origin)
    qidx, tri, w = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, beta, grow)
    idx, off, _ = mesh_query(mesh, beta)
    counts = np.diff(to_np(off))
    assert np.array_equal(to_np(tri), to_np(idx))
    assert np.array_equal(to_np(qidx), np.repeat(np.arange(beta.shape[0]), counts))
    expected = triangle_weights(mesh.vertices_source[mesh.leaves[tri]], beta[qidx])
    assert np.array_equal(to_np(w), to_np(expected))


def test_forward_raytrace_reads_a_scalar_source_and_empty_sources(mesh):
    x, y, counts = forward_raytrace(0.05, 0.02, _sie_like, mesh)
    assert counts.shape == (1,) and int(to_np(counts)[0]) >= 1
    assert x.shape == y.shape == (int(to_np(counts)[0]),)
    x, y, counts = forward_raytrace(f64([]), f64([]), _sie_like, mesh)
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
