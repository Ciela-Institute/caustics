import numpy as np
import pytest

import caustics
from caustics.backend_obj import backend
from caustics.lenses.utils import pixel_jacobian
from caustics.utils import (
    _contour_distance,
    _contours_agree,
    _extract_contours,
    _mask_contours,
    _contours_touch_edge,
    find_contours,
    meshgrid,
)


def _grid(fov, resolution):
    """Mirror of the grid find_contours builds, for exercising helpers directly."""
    npix = max(int(round(fov / resolution)) + 1, 2)
    return meshgrid(resolution, npix, dtype=backend.float64)


def test_extract_contours_unit_circle():
    X, Y = _grid(4.0, 0.05)
    contours = _extract_contours(lambda a, b: a**2 + b**2, X, Y, 1.0)

    assert len(contours) == 1
    c = contours[0]
    assert isinstance(c, np.ndarray)
    assert c.ndim == 2 and c.shape[1] == 2
    assert c.dtype == np.float64

    r = np.hypot(c[:, 0], c[:, 1])
    assert np.abs(r - 1.0).max() < 1e-3


def test_extract_contours_returns_empty_when_level_absent():
    X, Y = _grid(4.0, 0.1)
    assert _extract_contours(lambda a, b: a**2 + b**2, X, Y, -1.0) == []


def test_extract_contours_calls_f_once_with_full_meshgrid():
    calls = []

    def f(a, b):
        calls.append((backend.to_numpy(a).shape, backend.to_numpy(b).shape))
        return a**2 + b**2

    X, Y = _grid(4.0, 0.1)
    _extract_contours(f, X, Y, 1.0)
    assert calls == [((41, 41), (41, 41))]


def test_extract_contours_casts_float32_field_to_float64():
    X, Y = _grid(4.0, 0.05)

    def f32(a, b):
        return backend.to(a**2 + b**2, dtype=backend.float32)

    contours = _extract_contours(f32, X, Y, 1.0)
    assert contours[0].dtype == np.float64


def _circle(radius, centre=(0.0, 0.0), n=64):
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.stack(
        [centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)], axis=1
    )


def test_mask_contours_drops_contour_fully_inside_disc():
    small = _circle(0.1)
    large = _circle(1.0)
    kept = _mask_contours([large, small], [(0.0, 0.0)], 0.5)
    assert len(kept) == 1
    assert np.allclose(kept[0], large)


def test_mask_contours_keeps_contour_with_any_vertex_outside():
    # radius 1.0 circle centred at (0.1, 0.0) versus a radius 1.0 mask at the origin.
    # The circle's right vertex is at 1.1, so it is not fully contained
    kept = _mask_contours(
        contours=[_circle(radius=1.0, centre=(0.1, 0.0))],
        mask_positions=[(0.0, 0.0)],
        mask_radius=[1.0],
    )
    assert len(kept) == 1


def test_mask_contours_drops_real_contour_when_radius_too_large():
    # documented sharp edge of the rule: a genuine contour inside the disc is dropped
    assert _mask_contours([_circle(1.0)], [(0.0, 0.0)], 1.5) == []


def test_mask_contours_handles_multiple_positions_and_per_position_radii():
    a = _circle(0.1, centre=(2.0, 0.0))
    b = _circle(0.4, centre=(-2.0, 0.0))
    keep = _circle(1.0)
    kept = _mask_contours([keep, a, b], [(2.0, 0.0), (-2.0, 0.0)], [0.2, 0.5])
    assert len(kept) == 1
    assert np.allclose(kept[0], keep)


def test_mask_contours_is_a_noop_without_positions():
    contours = [_circle(1.0), _circle(0.1)]
    assert _mask_contours(contours, None, None) is contours
    assert len(_mask_contours(contours, [], 0.5)) == 2


BOUNDS = (-2.0, 2.0, -2.0, 2.0)


def test_contours_touch_edge_false_for_interior_contour():
    assert not _contours_touch_edge([_circle(1.0)], BOUNDS, 1e-8)


def test_contours_touch_edge_true_for_vertex_on_boundary():
    clipped = np.array([[0.0, -2.0], [0.5, 0.0], [0.0, 2.0]])
    assert _contours_touch_edge([clipped], BOUNDS, 1e-8)


def test_contours_touch_edge_checks_every_bound():
    for vertex in ([-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]):
        contour = np.array([[0.0, 0.0], vertex, [0.1, 0.1]])
        assert _contours_touch_edge([contour], BOUNDS, 1e-8)


def test_contours_touch_edge_true_if_any_contour_touches():
    clipped = np.array([[0.0, -2.0], [0.5, 0.0]])
    assert _contours_touch_edge([_circle(1.0), clipped], BOUNDS, 1e-8)


def test_contours_touch_edge_false_for_empty_list():
    assert not _contours_touch_edge([], BOUNDS, 1e-8)


def test_contours_touch_edge_tolerance_is_tight():
    # a contour one whole pixel inside the boundary must not be flagged
    near = np.array([[0.0, 0.0], [1.9, 0.0]])
    assert not _contours_touch_edge([near], BOUNDS, 1e-6 * 0.1)


def _point_to_segments_bruteforce(point, starts, ends):
    """Independent O(S) reference for the exact point-to-segment distance."""
    out = []
    for start, end in zip(starts, ends):
        span = end - start
        denom = float(span @ span)
        t = 0.0 if denom == 0 else float((point - start) @ span) / denom
        foot = start + min(max(t, 0.0), 1.0) * span
        out.append(float(np.hypot(*(point - foot))))
    return min(out)


def test_contour_distance_is_exact_against_bruteforce_projection():
    # The KD-tree only prunes which segments are projected onto; the answer must
    # match an independent brute-force projection over every segment exactly.
    a, b = _circle(1.0, n=137), _circle(1.013, n=291)
    brute = max(
        max(_point_to_segments_bruteforce(p, b[:-1], b[1:]) for p in a),
        max(_point_to_segments_bruteforce(p, a[:-1], a[1:]) for p in b),
    )
    assert _contour_distance(a, b) == pytest.approx(brute, rel=0, abs=1e-12)


def test_contour_distance_zero_for_identical_contours():
    c = _circle(1.0)
    assert _contour_distance(c, c) < 1e-12


def test_contour_distance_equals_offset_for_concentric_circles():
    inner, outer = _circle(1.0, n=2000), _circle(1.002, n=2000)
    assert _contour_distance(inner, outer) == pytest.approx(0.002, abs=5e-5)


def test_contour_distance_is_insensitive_to_vertex_density():
    # the whole point of measuring against segments: same curve, very different sampling
    coarse, fine = _circle(1.0, n=40), _circle(1.0, n=4000)

    coarse_vertex_spacing = np.linalg.norm(np.diff(coarse, axis=0), axis=1).max()
    assert coarse_vertex_spacing > 0.1  # sanity: the samplings really do differ

    d = _contour_distance(coarse, fine)
    assert d < 0.01  # far below the vertex spacing a point-to-point metric would report


def test_contour_distance_is_symmetric():
    a, b = _circle(1.0, n=200), _circle(1.05, n=97)
    assert _contour_distance(a, b) == pytest.approx(_contour_distance(b, a))


def test_contour_distance_measures_geometry_not_pixel_scale():
    """The metric must converge in distance, not merely trail the resolution.

    A metric that trails the pixel scale holds dH/h constant across refinements;
    one that measures real displacement lets it fall, because a smooth contour's
    geometry converges at second order while h only halves. This is precisely
    what a vertex-to-vertex Hausdorff distance fails -- there dH/h stays pinned
    near 1.0 -- and it is why this module projects onto segments instead.
    """
    ratios = []
    h = 0.05
    previous = _extract_contours(lambda a, b: a**2 + b**2, *_grid(4.0, h), 1.0)[0]
    for _ in range(3):
        h /= 2
        current = _extract_contours(lambda a, b: a**2 + b**2, *_grid(4.0, h), 1.0)[0]
        ratios.append(_contour_distance(previous, current) / h)
        previous = current

    # Measured ~0.0247, 0.0121, 0.0061: each roughly half its predecessor.
    for coarser, finer in zip(ratios, ratios[1:]):
        assert finer < 0.65 * coarser, f"dH/h did not fall: {ratios}"
    # And it must land far below 1.0, where a pixel-scale-trailing metric sits.
    assert max(ratios) < 0.1


def test_contour_distance_cost_is_independent_of_tolerance():
    # No tolerance reaches the metric any more, so an extreme geometry_tolerance
    # cannot drive allocation. This is what the removed densification cap was
    # patching over.
    a, b = _circle(1.0, n=500), _circle(1.01, n=500)
    assert _contour_distance(a, b) == pytest.approx(0.01, abs=1e-4)
    agreed, worst = _contours_agree([a], [b], 1e-12)
    assert not agreed
    assert np.isfinite(worst)


def test_contours_agree_for_identical_sets():
    contours = [_circle(1.0), _circle(0.5, centre=(3.0, 0.0))]
    agreed, worst = _contours_agree(contours, list(contours), 1e-3)
    assert agreed
    assert worst < 1e-9


def test_contours_agree_is_order_independent():
    a, b = _circle(1.0), _circle(0.5, centre=(3.0, 0.0))
    agreed, _ = _contours_agree([a, b], [b, a], 1e-3)
    assert agreed


def test_contours_agree_false_when_counts_differ():
    agreed, worst = _contours_agree([_circle(1.0)], [_circle(1.0), _circle(0.1)], 1e-3)
    assert not agreed
    assert worst == np.inf


def test_contours_agree_false_when_a_contour_moves_beyond_tolerance():
    tol = 1e-3
    agreed, worst = _contours_agree(
        [_circle(1.0, n=2000)], [_circle(1.0 + 2 * tol, n=2000)], tol
    )
    assert not agreed
    assert worst == pytest.approx(2 * tol, abs=1e-4)


def test_contours_agree_catches_one_bad_contour_among_stable_ones():
    tol = 1e-3
    stable = _circle(0.5, centre=(3.0, 0.0), n=2000)
    previous = [_circle(1.0, n=2000), stable]
    current = [_circle(1.0 + 5 * tol, n=2000), stable]
    agreed, _ = _contours_agree(previous, current, tol)
    assert not agreed


def test_contours_agree_rejects_two_empty_sets():
    # Refinement only ever reaches an empty set by losing contours it already
    # had, so "agreeing" here would report a silent empty success.
    agreed, worst = _contours_agree([], [], 1e-3)
    assert not agreed
    assert worst == float("inf")


def _circle_field(a, b):
    return a**2 + b**2


def test_find_contours_single_circle():
    contours = find_contours(_circle_field, 1.0, fov=4.0, resolution=0.05)
    assert len(contours) == 1
    r = np.hypot(contours[0][:, 0], contours[0][:, 1])
    assert np.abs(r - 1.0).max() < 1e-3


def test_find_contours_return_type():
    contours = find_contours(_circle_field, 1.0, fov=4.0, resolution=0.05)
    assert isinstance(contours, list)
    for c in contours:
        assert isinstance(c, np.ndarray)
        assert c.ndim == 2 and c.shape[1] == 2
        assert c.dtype == np.float64


def test_find_contours_expands_fov_until_contour_is_enclosed():
    # radius-4 circle: fov=1,2,4 find nothing, fov=8 touches the boundary exactly,
    # fov=16 encloses it
    contours = find_contours(_circle_field, 16.0, fov=1.0, resolution=0.1)
    assert len(contours) == 1
    r = np.hypot(contours[0][:, 0], contours[0][:, 1])
    assert np.abs(r - 4.0).max() < 1e-2


def test_find_contours_raises_when_no_contour_exists():
    with pytest.raises(RuntimeError, match="no contours"):
        find_contours(
            _circle_field, -1.0, fov=4.0, resolution=0.1, max_fov_expansions=3
        )


def test_find_contours_raises_for_unbounded_contour():
    with pytest.raises(RuntimeError, match="edge"):
        find_contours(
            lambda a, b: a, 0.0, fov=4.0, resolution=0.1, max_fov_expansions=3
        )


def _sliver_field(a, b):
    # A unit circle plus a thin sliver centred at x=1.85 that falls entirely
    # between grid points at resolution=0.1 -- and so is invisible to Phase
    # 1's edge check at that resolution -- but resolves once refinement
    # reaches resolution<=0.05.
    return (a**2 + b**2 - 1.0) * ((a - 1.85) ** 2 - 0.02**2)


def test_find_contours_raises_when_refinement_reveals_edge_touching_contour():
    # Phase 1 accepts fov=4.0 having only ever seen the unit circle at
    # resolution=0.1, since the sliver is not resolved there at all. Without a
    # Phase 2 edge check, refinement would converge once the sliver resolves
    # and silently return contours clipped at the y=+/-2.0 domain boundary
    # instead of raising.
    with pytest.raises(RuntimeError, match="grid edge"):
        find_contours(
            _sliver_field, 0.0, fov=4.0, resolution=0.1, geometry_tolerance=1e-3
        )


def _bounded_sliver_field(a, b):
    # `_sliver_field` above is not a fair vehicle for an "increasing fov
    # fixes it" check: its second factor, (a - 1.85)**2 - 0.02**2, does not
    # depend on b at all, so its zero set is two mathematically infinite
    # vertical lines -- no finite fov can ever enclose them, and indeed
    # find_contours(_sliver_field, ..., fov=8.0, ...) still raises. This
    # variant closes the sliver into a bounded, thin ellipse (semi-axes 0.02
    # in x, 2.3 in y) that is genuinely enclosable, while preserving the
    # property that exercises the fix: at resolution=0.1 the ellipse is
    # invisible (its ends land inside a single grid cell, same as the
    # sliver), and only resolves once refinement reaches resolution<=0.05.
    return (a**2 + b**2 - 1.0) * (((a - 1.85) / 0.02) ** 2 + (b / 2.3) ** 2 - 1.0)


def test_find_contours_edge_touching_error_is_actionable_with_larger_fov():
    # At fov=4.0 the same shape as above: revealed only by refinement, and
    # clipped by the domain edge (the ellipse's y-extent of 2.3 exceeds the
    # fov=4 half-span of 2.0), so this raises for the same reason.
    with pytest.raises(RuntimeError, match="grid edge"):
        find_contours(
            _bounded_sliver_field, 0.0, fov=4.0, resolution=0.1, geometry_tolerance=1e-2
        )

    # A large enough fov encloses the ellipse entirely (half-span 4.0 > 2.3),
    # so refinement converges with every contour interior -- proving the
    # RuntimeError above names a real, fixable problem rather than being a
    # dead end.
    contours = find_contours(
        _bounded_sliver_field, 0.0, fov=8.0, resolution=0.1, geometry_tolerance=1e-2
    )
    assert len(contours) == 2
    for c in contours:
        assert np.abs(c[:, 0]).max() < 3.99
        assert np.abs(c[:, 1]).max() < 3.99


def test_find_contours_accepts_empty_mask_positions():
    # `_mask_contours` already treats an empty list as a no-op; validation
    # must not demand a mask_radius that will never be used.
    contours = find_contours(
        _circle_field, 1.0, fov=4.0, resolution=0.05, mask_positions=[]
    )
    assert len(contours) == 1


def test_find_contours_applies_masks_before_checks():
    # a genuine circle plus a tiny artifact-like ring near the origin
    def two_scales(a, b):
        r2 = a**2 + b**2
        return (r2 - 1.0) * (r2 - 0.01)

    unmasked = find_contours(two_scales, 0.0, fov=4.0, resolution=0.02)
    assert len(unmasked) == 2

    masked = find_contours(
        two_scales,
        0.0,
        fov=4.0,
        resolution=0.02,
        mask_positions=[(0.0, 0.0)],
        mask_radius=0.5,
    )
    assert len(masked) == 1
    r = np.hypot(masked[0][:, 0], masked[0][:, 1])
    assert np.abs(r - 1.0).max() < 1e-2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fov": 0.0},
        {"fov": -1.0},
        {"resolution": 0.0},
        {"resolution": -0.1},
        {"fov_expansion_factor": 1.0},
        {"fov_expansion_factor": 0.5},
        {"max_fov_expansions": -1},
        {"max_resolution_halvings": -1},
        {"max_resolution_halvings": 0},
        {"geometry_tolerance": 0.0},
        {"mask_positions": [(0.0, 0.0)]},
        {"mask_positions": [(0.0, 0.0)], "mask_radius": 0.0},
        {"mask_positions": [(0.0, 0.0)], "mask_radius": 0.1, "resolution": 0.1},
        {
            "mask_positions": [(0.0, 0.0), (1.0, 1.0)],
            "mask_radius": [0.5, 0.1],
            "resolution": 0.1,
        },
        {"mask_positions": [(0.0, 0.0, 1.0)], "mask_radius": 0.5},
        {"mask_positions": [(0.0, 0.0), (1.0, 1.0)], "mask_radius": [0.1, 0.2, 0.3]},
    ],
)
def test_find_contours_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        find_contours(_circle_field, 1.0, **kwargs)


def test_find_contours_refines_until_geometry_is_stable():
    # tolerance 1e-3 converges after two halvings from resolution 0.1 (see spec table);
    # the returned curve is far more accurate than the coarse Phase 1 grid's 2.5e-3
    contours = find_contours(
        _circle_field, 1.0, fov=4.0, resolution=0.1, geometry_tolerance=1e-3
    )
    assert len(contours) == 1
    r = np.hypot(contours[0][:, 0], contours[0][:, 1])
    assert np.abs(r - 1.0).max() < 5e-4


def test_find_contours_multiple_disjoint_contours():
    def two_circles(a, b):
        left = (a + 1.5) ** 2 + b**2
        right = (a - 1.5) ** 2 + b**2
        return backend.where(left < right, left, right)

    contours = find_contours(two_circles, 0.25, fov=6.0, resolution=0.1)
    assert len(contours) == 2
    for c in contours:
        centre = -1.5 if c[:, 0].mean() < 0 else 1.5
        r = np.hypot(c[:, 0] - centre, c[:, 1])
        assert np.abs(r - 0.5).max() < 1e-2


def test_find_contours_raises_when_refinement_does_not_converge():
    with pytest.raises(RuntimeError, match="refin"):
        find_contours(
            _circle_field,
            1.0,
            fov=4.0,
            resolution=0.1,
            geometry_tolerance=1e-12,
            max_resolution_halvings=3,
        )


def test_find_contours_refinement_error_reports_distance():
    with pytest.raises(RuntimeError) as excinfo:
        find_contours(
            _circle_field,
            1.0,
            fov=4.0,
            resolution=0.1,
            geometry_tolerance=1e-12,
            max_resolution_halvings=3,
        )
    message = str(excinfo.value)
    assert "contour counts 1 then 1" in message
    assert "distance" in message.lower()


def test_find_contours_raises_when_refinement_loses_all_contours():
    # Grids nest under halving, so a real field cannot lose a sign change;
    # masking can. This stub reproduces the end state either way: a set that
    # was non-empty in Phase 1 and is empty twice running in refinement. That
    # must raise rather than return [].
    def vanishing(a, b):
        if backend.to_numpy(a).shape[0] > 41:
            return a**2 + b**2 + 2.0
        return a**2 + b**2

    with pytest.raises(RuntimeError, match="refin"):
        find_contours(
            vanishing, 1.0, fov=4.0, resolution=0.1, max_resolution_halvings=3
        )


def test_find_contours_rejects_mask_radius_below_pixel_diagonal():
    # A singularity landing on a grid point puts the contour's corner vertices
    # on the diagonal neighbours, at resolution*sqrt(2). A smaller radius never
    # catches the artifact it exists to remove.
    floor = 0.1 * np.sqrt(2)

    with pytest.raises(ValueError, match="mask_radius"):
        find_contours(
            _circle_field,
            1.0,
            fov=4.0,
            resolution=0.1,
            mask_positions=[(0.0, 0.0)],
            mask_radius=floor,
        )

    contours = find_contours(
        _circle_field,
        1.0,
        fov=4.0,
        resolution=0.1,
        mask_positions=[(0.0, 0.0)],
        mask_radius=1.01 * floor,
    )
    assert len(contours) == 1


def test_find_contours_mask_radius_floor_uses_the_initial_resolution():
    # Refinement only ever shrinks the pixel scale, so the coarsest grid sets
    # the largest artifact and therefore the binding floor.
    with pytest.raises(ValueError, match="mask_radius"):
        find_contours(
            _circle_field,
            1.0,
            fov=4.0,
            resolution=0.4,
            mask_positions=[(0.0, 0.0)],
            mask_radius=0.3,
        )


def test_find_contours_zero_halvings_rejected():
    # max_resolution_halvings=0 would skip the refinement loop entirely and
    # return an unverified Phase 1 contour set, silently weakening the
    # function's "verified stable across at least one refinement" contract.
    with pytest.raises(ValueError, match="max_resolution_halvings"):
        find_contours(
            _circle_field, 1.0, fov=4.0, resolution=0.05, max_resolution_halvings=0
        )


def test_sis_critical_curve_is_a_circle_of_radius_rein():
    """For an SIS, det A = 1 - Rein / r, so the critical curve is r = Rein."""
    rein = 1.0
    cosmology = caustics.FlatLambdaCDM(name="cosmo")
    lens = caustics.SIS(
        cosmology=cosmology, z_l=0.5, z_s=1.0, x0=0.0, y0=0.0, Rein=rein, name="sis"
    )

    def signed_det(X, Y):
        def det_at(px, py):
            jac = pixel_jacobian(lens.raytrace, px, py)
            return jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]

        flat = backend.vmap(det_at, in_dims=(0, 0))(X.reshape(-1), Y.reshape(-1))
        return backend.view(flat, X.shape)

    contours = find_contours(
        signed_det,
        0.0,
        fov=4.0,
        resolution=0.1,
        geometry_tolerance=1e-3,
        mask_positions=[(0.0, 0.0)],
        mask_radius=0.5,
    )

    assert len(contours) == 1
    r = np.hypot(contours[0][:, 0], contours[0][:, 1])
    assert np.abs(r - rein).max() < 1e-3


def test_inverse_magnification_cannot_locate_critical_curves():
    """1 / magnification is abs(det A); it never crosses zero, so nothing is found."""
    cosmology = caustics.FlatLambdaCDM(name="cosmo2")
    lens = caustics.SIS(
        cosmology=cosmology, z_l=0.5, z_s=1.0, x0=0.0, y0=0.0, Rein=1.0, name="sis2"
    )

    X, Y = _grid(4.0, 0.05)
    z = backend.to_numpy(1.0 / lens.magnification(X, Y))
    assert (z < 0).sum() == 0

    assert (
        _extract_contours(lambda a, b: 1.0 / lens.magnification(a, b), X, Y, 0.0) == []
    )
