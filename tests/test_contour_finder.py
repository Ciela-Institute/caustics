import numpy as np

from caustics.backend_obj import backend
from caustics.lenses.utils import (
    _extract_contours,
    _mask_contours,
    _contours_touch_edge,
)
from caustics.utils import meshgrid


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
    # radius 1.0 curve against a radius 0.99 mask: every vertex is outside
    kept = _mask_contours([_circle(1.0)], [(0.0, 0.0)], 0.99)
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
