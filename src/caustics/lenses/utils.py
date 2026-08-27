from typing import Tuple

import numpy as np
from contourpy import contour_generator

from ..backend_obj import backend, ArrayLike

__all__ = ("pixel_jacobian", "pixel_magnification", "magnification")


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
