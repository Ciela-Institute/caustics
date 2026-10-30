"""Holes around lens centers: merging, sampling, and the holes a mesh stores."""

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, SinglePlane
from caustics.lenses.func.adaptive import CenterHoles, LensMesh
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
    assert_same,
    broken_where,
    build,
    f64,
    numpy_lens,
    point_mass,
    sis,
    to_np,
    user_build,
)

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
    [[(0.3, -0.2)], ((0.3, -0.2),), np.array([[0.3, -0.2]]), f64([[0.3, -0.2]])],
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


def sis_in_shear_raytrace(c, b, kappa, gamma_1, gamma_2):
    """An SIS at ``c`` in a uniform convergence and shear about ``c``:
    the smooth pull other lenses add at an isothermal center."""

    def raytrace(x, y):
        dx, dy = x - c[0], y - c[1]
        r = backend.sqrt(dx * dx + dy * dy)
        return (
            x - b * dx / r - (kappa + gamma_1) * dx - gamma_2 * dy,
            y - b * dy / r - gamma_2 * dx - (kappa - gamma_1) * dy,
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


def sample(raytrace, centers, radius, min_img_sep, batch_size=None):
    """``sample_holes`` of ``raytrace`` through ``make_sampler``."""
    return sample_holes(
        make_sampler(raytrace, None, None, batch_size),
        f64(centers),
        f64(radius),
        min_img_sep,
        batch_size,
    )


def test_an_sis_hole_curve_is_its_analytic_circle_sampled_to_min_img_sep():
    c, b, r = (0.3, -0.2), 1.0, 0.01
    holes = sample(numpy_lens(*sis([c], b)).raytrace, [c], [r], r)
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
    "sis": numpy_lens(*sis([(0.3, -0.2)])).raytrace,
    "sis in shear": sis_in_shear_raytrace((0.3, -0.2), 1.0, 0.1, 0.2, -0.1),
}


@pytest.mark.parametrize("name", ISOTHERMAL)
def test_an_isothermal_center_is_a_pseudo_caustic(name):
    holes = sample(ISOTHERMAL[name], [(0.3, -0.2)], [0.01], 0.01)
    growth, err = to_np(holes.growth)[0], to_np(holes.growth_err)[0]
    assert to_np(holes.pseudo_caustic).tolist() == [True]
    assert abs(growth) <= err
    assert abs(growth) < 1e-5


@pytest.mark.parametrize("t", [1.0 - 1e-4, 1.0 + 1e-4])
def test_a_power_law_a_hair_off_isothermal_is_no_pseudo_caustic(t):
    """A slope ``t`` gives ``growth = 1 - t``, resolved well below 1e-4."""
    c = (0.3, -0.2)
    holes = sample(power_law_raytrace(c, 1.0, t), [c], [0.01], 0.01)
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
    holes = sample(lens.raytrace, centers, [0.005, 0.005], 0.005)
    assert to_np(holes.pseudo_caustic).tolist() == [True, True]


def test_a_point_mass_hole_curve_grows_as_the_hole_shrinks():
    c = (0.1, 0.2)
    holes = sample(numpy_lens(*point_mass(c, 0.1)).raytrace, [c], [0.01], 0.01)
    assert abs(to_np(holes.growth)[0] + 1.0) < 0.01
    assert to_np(holes.pseudo_caustic).tolist() == [False]


def test_a_regular_point_hole_curve_shrinks_with_the_hole():
    holes = sample(
        numpy_lens(affine, affine_jacobian).raytrace, [(0.4, -0.3)], [0.01], 0.01
    )
    assert abs(to_np(holes.growth)[0] - 1.0) < 1e-9
    assert to_np(holes.pseudo_caustic).tolist() == [False]


def test_a_lens_mapping_every_circle_to_one_point_has_no_growth():
    def constant(x, y):
        return 0.0 * x + 0.5, 0.0 * y - 0.1

    holes = sample(constant, [(0.4, -0.3)], [0.01], 0.01)
    assert np.isnan(to_np(holes.growth)[0])
    assert to_np(holes.pseudo_caustic).tolist() == [False]


def test_a_hole_curve_too_large_to_resolve_stops_at_the_cap_and_warns():
    """A 1" point mass maps a 0.005" circle to a loop of radius about 200":
    ``2**16`` samples cannot bring its chords down to 0.005"."""
    c = (0.0, 0.0)
    with pytest.warns(UserWarning, match=r"stopped at \d+ samples \(cap 65536\)"):
        holes = sample(numpy_lens(*point_mass(c, 1.0)).raytrace, [c], [0.005], 0.005)
    assert to_np(holes.offsets)[-1] <= HOLE_MAX_SAMPLES


def test_a_hole_frozen_below_the_cap_warns_with_its_own_sample_count():
    """A point mass off the hole's center makes the hole curve uneven.

    Refinement then bisects only some intervals, so a round can stop short
    of the cap because the next one would pass it. The warning states the
    count the hole kept, not the cap.
    """
    c = (0.0, 0.0)
    with pytest.warns(UserWarning, match=r"\(cap 65536\)") as got:
        holes = sample(
            numpy_lens(*point_mass((0.004, 0.0), 0.3)).raytrace, [c], [0.005], 0.005
        )
    n = int(to_np(holes.offsets)[-1])
    assert n < HOLE_MAX_SAMPLES
    assert any(f"stopped at {n} samples (cap 65536)" in str(w.message) for w in got)


def test_a_lens_not_finite_on_a_hole_circle_raises_naming_the_center():
    c = (0.4, -0.3)
    broken = numpy_lens(
        *broken_where(affine, affine_jacobian, lambda p: p[:, 0] > c[0])
    ).raytrace
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
    broken = numpy_lens(
        *broken_where(
            affine,
            affine_jacobian,
            lambda p: ((p - np.asarray(c)) ** 2).sum(axis=-1) < nan_within**2,
        )
    ).raytrace
    with pytest.raises(ValueError, match=named):
        sample(broken, [c], [0.01], 0.01)


def test_batching_the_hole_samples_changes_nothing():
    c, r = (0.3, -0.2), 0.01
    holes = sample(numpy_lens(*sis([c])).raytrace, [c], [r], r)
    lens_ = numpy_lens(*sis([c]))
    batched = sample(lens_.raytrace, [c], [r], r, batch_size=100)
    assert max(len(x) for x in lens_.calls.raytrace) <= 100
    assert_same(batched, holes)


def test_holes_sampled_together_match_holes_sampled_alone():
    """One call per round covers every hole; no hole's samples leak into another's."""
    lens = numpy_lens(*sis([(0.3, -0.2)])).raytrace
    pairs = [((0.3, -0.2), 0.01), ((1.5, 1.0), 0.02)]
    both = sample(lens, [c for c, _ in pairs], [r for _, r in pairs], 0.01)
    off = to_np(both.offsets)
    for h, (c, r) in enumerate(pairs):
        alone = sample(lens, [c], [r], 0.01)
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


SIS_C = (0.3001, -0.2003)  # off every lattice point of BUILD


BUILD = dict(fov=4.0, init_res=8, min_img_sep=0.02)


def test_a_build_stores_the_merged_and_sampled_holes():
    centers = [SIS_C, SIS_C, (1.5, 1.5)]
    mesh, lens_ = build(*sis([SIS_C]), **BUILD, centers=centers)
    want = sample_holes(
        make_sampler(lens_.raytrace, None, None),
        *merge_centers(centers, mesh.min_img_sep),
        mesh.min_img_sep,
        None,
    )
    assert mesh.holes.centers.shape[0] == 2
    assert to_np(mesh.holes.radius).tolist() == [mesh.min_img_sep] * 2
    assert_same(mesh.holes, want)


def test_centers_change_nothing_but_the_holes_even_outside_the_fov():
    plain, _ = user_build(*sis([SIS_C]), **BUILD)
    holed, _ = user_build(*sis([SIS_C]), **BUILD, centers=[SIS_C, (5.0, 5.0)])
    assert holed.holes.centers.shape[0] == 2
    for name in LensMesh._fields:
        if name != "holes":
            assert_same(getattr(holed, name), getattr(plain, name), name)


def test_holes_cost_raytraces_on_their_circles_only_and_no_jacobian():
    _, plain = user_build(*sis([SIS_C]), **BUILD)
    mesh, holed = user_build(*sis([SIS_C]), **BUILD, centers=[SIS_C])
    base = np.concatenate(plain.calls.raytrace)
    extra = np.concatenate(holed.calls.raytrace)
    n_samples = int(to_np(mesh.holes.offsets)[-1])
    assert len(extra) - len(base) == n_samples + 4 * HOLE_GROWTH_SAMPLES
    added = extra[
        ~np.isin(extra[:, 0] + 1j * extra[:, 1], base[:, 0] + 1j * base[:, 1])
    ]
    d = np.hypot(*(added - np.array(SIS_C)).T)
    radii = mesh.min_img_sep / 4.0 ** np.arange(4)
    assert np.isclose(d[:, None], radii, rtol=0, atol=1e-12).any(axis=1).all()
    assert sum(map(len, holed.calls.jacobian)) == sum(map(len, plain.calls.jacobian))


def test_holes_land_on_the_mesh_device(device):
    mesh, _ = user_build(*sis([SIS_C]), **BUILD, centers=[SIS_C], device=device)
    for field in CenterHoles._fields:
        assert mesh_backend.device(getattr(mesh.holes, field)) == mesh_backend.device(
            mesh.vertices_lens
        ), field
