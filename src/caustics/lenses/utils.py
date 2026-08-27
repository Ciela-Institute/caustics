from typing import Callable, Optional, Tuple

import numpy as np
from contourpy import contour_generator
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from ..backend_obj import backend, ArrayLike
from ..utils import meshgrid

__all__ = ("pixel_jacobian", "pixel_magnification", "magnification", "find_contours")


def pixel_jacobian(
    raytrace, x, y
) -> Tuple[Tuple[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]:
    """Computes the Jacobian matrix of the partial derivatives of the
    image position with respect to the source position
    (:math:`\\partial \beta / \\partial \theta`).  This is done at a
    single point on the lensing plane.

    Parameters
    -----------
    raytrace: function
        A function that maps the lensing plane coordinates to the source plane coordinates.
    x: ArrayLike
        The x-coordinate on the lensing plane.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinate on the lensing plane.

        *Unit: arcsec*

    Returns
    --------
    The Jacobian matrix of the image position with respect
    to the source position at the given point.

        *Unit: unitless*

    """
    jac = backend.jacfwd(raytrace, (0, 1))(x, y)  # type: ignore
    return jac


def pixel_magnification(raytrace, x, y) -> ArrayLike:
    """
    Computes the magnification at a single point on the lensing plane.
    The magnification is derived from the determinant
    of the Jacobian matrix of the image position with respect to the source position.

    Parameters
    ----------
    raytrace: function
        A function that maps the lensing plane coordinates to the source plane coordinates.

    x: ArrayLike
        The x-coordinate on the lensing plane.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinate on the lensing plane.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        The magnification at the given point on the lensing plane.

        *Unit: unitless*

    """
    jac = pixel_jacobian(raytrace, x, y)
    return 1 / backend.abs(jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0])  # fmt: skip


def magnification(raytrace, x, y) -> ArrayLike:
    """
    Computes the magnification over a grid on the lensing plane.
    This is done by calling `pixel_magnification`
    for each point on the grid.

    Parameters
    ----------
    raytrace: function
        A function that maps the lensing plane coordinates to the source plane coordinates.

    x: ArrayLike
        The x-coordinates on the lensing plane.

        *Unit: arcsec*

    y: ArrayLike
        The y-coordinates on the lensing plane.

        *Unit: arcsec*

    Returns
    --------
    ArrayLike
        A tensor representing the magnification at each point on the grid.

        *Unit: unitless*

    """
    return backend.view(
        backend.vmap(pixel_magnification, in_dims=(None, 0, 0))(
            raytrace, x.reshape(-1), y.reshape(-1)
        ),
        x.shape,
    )


def _extract_contours(f, X, Y, target_value: float) -> list:
    """Evaluate `f` on the grid and extract contour lines at `target_value`."""
    z = np.asarray(backend.to_numpy(f(X, Y)), dtype=np.float64)
    generator = contour_generator(
        x=np.asarray(backend.to_numpy(X), dtype=np.float64),
        y=np.asarray(backend.to_numpy(Y), dtype=np.float64),
        z=z,
        name="serial",
        line_type="Separate",
        quad_as_tri=True,
    )
    return generator.lines(float(target_value))


def _mask_contours(contours: list, mask_positions, mask_radius) -> list:
    """Drop contours lying entirely within `mask_radius` of a masked position."""
    if mask_positions is None or len(mask_positions) == 0:
        return contours

    positions = np.atleast_2d(np.asarray(mask_positions, dtype=np.float64))
    radii = np.broadcast_to(
        np.asarray(mask_radius, dtype=np.float64), (positions.shape[0],)
    )

    kept = []
    for contour in contours:
        inside = False
        for (px, py), radius in zip(positions, radii):
            if np.all(np.hypot(contour[:, 0] - px, contour[:, 1] - py) <= radius):
                inside = True
                break
        if not inside:
            kept.append(contour)
    return kept


def _contours_touch_edge(
    contours: list, bounds: Tuple[float, float, float, float], atol: float
) -> bool:
    """True if any contour vertex lies within `atol` of the grid boundary."""
    x_min, x_max, y_min, y_max = bounds
    for contour in contours:
        cx, cy = contour[:, 0], contour[:, 1]
        if (
            np.any(np.abs(cx - x_min) <= atol)
            or np.any(np.abs(cx - x_max) <= atol)
            or np.any(np.abs(cy - y_min) <= atol)
            or np.any(np.abs(cy - y_max) <= atol)
        ):
            return True
    return False


# Total inserted points is capped so a pathologically small `spacing` (e.g. an
# extreme `geometry_tolerance`) cannot blow up memory. Rescaling to the coarsest
# spacing that still fits the cap only ever makes `_densify_contour` sparser, and
# `_contour_distance`'s guarantee is one-sided (densified points lie on the
# polyline, so the reported distance can only overstate a difference, never
# understate one) -- so a coarser cap-limited spacing can still never falsely
# declare convergence.
_MAX_DENSIFIED_POINTS = 1_000_000


def _densify_contour(contour: np.ndarray, spacing: float) -> np.ndarray:
    """Insert points along each segment so consecutive spacing never exceeds `spacing`."""
    if len(contour) < 2:
        return contour

    starts, ends = contour[:-1], contour[1:]
    lengths = np.linalg.norm(ends - starts, axis=1)
    effective_spacing = spacing
    counts = np.ceil(lengths / effective_spacing)
    if counts.sum() > _MAX_DENSIFIED_POINTS:
        effective_spacing = lengths.sum() / _MAX_DENSIFIED_POINTS
        counts = np.ceil(lengths / effective_spacing)
    counts = np.maximum(counts.astype(int), 1)

    pieces = []
    for start, end, count in zip(starts, ends, counts):
        fractions = np.linspace(0.0, 1.0, count + 1)[:-1, None]
        pieces.append(start + fractions * (end - start))
    pieces.append(contour[-1][None, :])
    return np.concatenate(pieces, axis=0)


def _contour_distance(a: np.ndarray, b: np.ndarray, spacing: float) -> float:
    """Symmetric point-to-polyline Hausdorff distance between two contours."""
    a_to_b = cKDTree(_densify_contour(b, spacing)).query(a, k=1)[0].max()
    b_to_a = cKDTree(_densify_contour(a, spacing)).query(b, k=1)[0].max()
    return float(max(a_to_b, b_to_a))


def _contours_agree(
    previous: list, current: list, tolerance: float
) -> Tuple[bool, float]:
    """Compare two contour sets by count and optimally paired polyline distance."""
    if len(previous) != len(current):
        return False, float("inf")
    if len(previous) == 0:
        return True, 0.0

    spacing = tolerance / 10
    cost = np.empty((len(previous), len(current)), dtype=np.float64)
    for i, prev in enumerate(previous):
        for j, cur in enumerate(current):
            cost[i, j] = _contour_distance(prev, cur, spacing)

    rows, cols = linear_sum_assignment(cost)
    paired = cost[rows, cols]
    return bool(np.all(paired < tolerance)), float(paired.max())


def _validate_find_contours_args(
    fov,
    fov_expansion_factor,
    max_fov_expansions,
    resolution,
    max_resolution_halvings,
    geometry_tolerance,
    mask_positions,
    mask_radius,
):
    if fov <= 0:
        raise ValueError(f"fov must be positive (received {fov})")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive (received {resolution})")
    if fov_expansion_factor <= 1:
        raise ValueError(
            f"fov_expansion_factor must exceed 1 (received {fov_expansion_factor})"
        )
    if max_fov_expansions < 0:
        raise ValueError(
            f"max_fov_expansions must be non-negative (received {max_fov_expansions})"
        )
    if max_resolution_halvings < 1:
        raise ValueError(
            f"max_resolution_halvings must be at least 1 (received {max_resolution_halvings})"
        )
    if geometry_tolerance <= 0:
        raise ValueError(
            f"geometry_tolerance must be positive (received {geometry_tolerance})"
        )

    if mask_positions is None:
        return

    positions = np.atleast_2d(np.asarray(mask_positions, dtype=np.float64))
    if positions.size == 0:
        return
    if positions.shape[1] != 2:
        raise ValueError(
            f"mask_positions must have shape (M, 2) (received {positions.shape})"
        )
    if mask_radius is None:
        raise ValueError("mask_radius is required when mask_positions is given")

    radii = np.atleast_1d(np.asarray(mask_radius, dtype=np.float64))
    if radii.size not in (1, positions.shape[0]):
        raise ValueError(
            f"mask_radius must be a scalar or have one value per masked position "
            f"({positions.shape[0]}); received {radii.size}"
        )
    if np.any(radii <= 0):
        raise ValueError(f"mask_radius must be positive (received {mask_radius})")


def find_contours(
    f: Callable,
    target_value: float,
    fov: float = 5.0,
    fov_expansion_factor: float = 2.0,
    max_fov_expansions: int = 5,
    resolution: float = 0.1,
    max_resolution_halvings: int = 8,
    geometry_tolerance: float = 1e-3,
    mask_positions: Optional[ArrayLike] = None,
    mask_radius: Optional[ArrayLike] = None,
    device=None,
) -> list[np.ndarray]:
    """
    Find contours of a scalar function at a target value, adapting both the field
    of view and the pixel scale until the result is stable.

    The field of view is grown by ``fov_expansion_factor`` until the target
    contours are present and fully enclosed by the grid. The pixel scale is then
    halved repeatedly, at fixed field of view, until two successive contour sets
    have the same number of contours and every optimally paired contour moves by
    less than ``geometry_tolerance``.

    Distances between contour sets are point-to-polyline, not point-to-point, so
    ``geometry_tolerance`` measures how far the curve actually moved rather than
    how densely it happens to be sampled.

    Parameters
    ----------
    f: Callable
        The scalar field to contour. Called once per grid as ``f(X, Y)`` with the
        full 2D meshgrid, and must return an array of the same shape.

    target_value: float
        The contour level to extract.

    fov: float
        The initial field of view, a square side length centred on the origin.

        *Unit: arcsec*

    fov_expansion_factor: float
        The factor by which the field of view grows when contours are missing or
        clipped. Must exceed 1.

        *Unit: unitless*

    max_fov_expansions: int
        The maximum number of expansions. The initial grid is not counted.

    resolution: float
        The initial pixel scale.

        *Unit: arcsec*

    max_resolution_halvings: int
        The maximum number of times the pixel scale is halved.

    geometry_tolerance: float
        The convergence threshold on contour displacement between successive
        refinements.

        *Unit: arcsec*

    mask_positions: Optional[ArrayLike]
        An ``(M, 2)`` array of positions around which contours are discarded. Use
        this for singular components of a lens model: when a singularity lands
        exactly on a grid point, the field flips sign at that single pixel and an
        unphysical one-pixel contour appears, which never converges under
        refinement. A contour is discarded only when *every* one of its vertices
        lies within ``mask_radius`` of a masked position, so a genuine contour
        passing nearby survives. Note that ``mask_radius`` must therefore stay
        below the extent of the smallest genuine contour.

        *Unit: arcsec*

    mask_radius: Optional[ArrayLike]
        The mask disc radius, either a scalar or one value per masked position.
        Required when ``mask_positions`` is given.

        *Unit: arcsec*

    device: optional
        The device on which to build the coordinate grid. Defaults to the backend
        default.

    Returns
    -------
    list of ArrayLike
        One ``(N, 2)`` float64 numpy array of ``(x, y)`` vertices per contour,
        taken from the finest grid. The order carries no meaning.

        *Unit: arcsec*

    Raises
    ------
    ValueError
        If any argument is out of range, or ``mask_positions`` is given without a
        positive ``mask_radius``.

    RuntimeError
        If the field of view cannot be grown enough to enclose the contours, if
        the contour geometry has not stabilised within
        ``max_resolution_halvings``, or if refinement converges on a contour set
        that touches the grid edge (a feature revealed only by refinement that
        extends beyond ``fov``).

    Notes
    -----
    An empty contour set is always treated as "the field of view is too small",
    because that cause cannot be distinguished from "the feature is smaller than
    the pixel scale". Choosing an initial ``resolution`` fine enough to detect
    the feature at all is the caller's responsibility.

    Contour extraction runs through numpy, so gradients do not propagate through
    this function.

    Cost scales as the square of the pixel count, and contours with cusps
    converge at first order rather than second, so they need roughly twice as
    many halvings per digit of accuracy as smooth contours. The refinement grid
    is bounded only by ``resolution`` and ``max_resolution_halvings``: the final
    grid is ``npix ~ fov * 2 ** max_resolution_halvings / resolution`` on a side,
    so choose those two together with the available memory in mind.

    For extreme ``geometry_tolerance``, the densification used internally to
    measure contour displacement is capped, and the metric's resolution floor
    becomes ``perimeter / 2e6`` rather than ``geometry_tolerance / 20``.
    """
    _validate_find_contours_args(
        fov,
        fov_expansion_factor,
        max_fov_expansions,
        resolution,
        max_resolution_halvings,
        geometry_tolerance,
        mask_positions,
        mask_radius,
    )

    contours = None
    touched_edge = False
    for _ in range(max_fov_expansions + 1):
        npix = max(int(round(fov / resolution)) + 1, 2)
        X, Y = meshgrid(resolution, npix, device=device, dtype=backend.float64)
        candidate = _mask_contours(
            _extract_contours(f, X, Y, target_value), mask_positions, mask_radius
        )
        half_span = resolution * (npix - 1) / 2
        bounds = (-half_span, half_span, -half_span, half_span)
        touched_edge = _contours_touch_edge(candidate, bounds, 1e-6 * resolution)
        if len(candidate) > 0 and not touched_edge:
            contours = candidate
            break
        fov *= fov_expansion_factor

    if contours is None:
        actual_span = resolution * (npix - 1)
        reason = (
            "contours touching the grid edge"
            if touched_edge
            else f"no contours at target_value={target_value}"
        )
        raise RuntimeError(
            f"find_contours failed to enclose the contours: after "
            f"{max_fov_expansions} expansion(s) the grid still produced {reason} "
            f"(final grid span={actual_span:.6g})."
        )

    previous = contours
    worst = float("inf")
    for _ in range(max_resolution_halvings):
        resolution /= 2
        npix = max(int(round(fov / resolution)) + 1, 2)
        X, Y = meshgrid(resolution, npix, device=device, dtype=backend.float64)
        current = _mask_contours(
            _extract_contours(f, X, Y, target_value), mask_positions, mask_radius
        )
        agreed, worst = _contours_agree(previous, current, geometry_tolerance)
        if agreed:
            half_span = resolution * (npix - 1) / 2
            if _contours_touch_edge(
                current,
                (-half_span, half_span, -half_span, half_span),
                1e-6 * resolution,
            ):
                raise RuntimeError(
                    f"find_contours refinement converged on a contour set touching the "
                    f"grid edge at resolution={resolution:.6g} (fov={fov:.6g}); a feature "
                    f"revealed only by refinement extends beyond the field of view. "
                    f"Increase fov."
                )
            return current
        previous = current

    raise RuntimeError(
        f"find_contours refinement failed to converge: after "
        f"{max_resolution_halvings} resolution halving(s) the contour set still "
        f"changed (contour counts {len(contours)} then {len(previous)}, largest "
        f"paired distance {worst:.6g} against "
        f"geometry_tolerance={geometry_tolerance:.6g})."
    )
