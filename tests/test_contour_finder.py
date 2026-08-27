import numpy as np

from caustics.backend_obj import backend
from caustics.lenses.utils import _extract_contours
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
