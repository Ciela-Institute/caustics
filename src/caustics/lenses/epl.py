# mypy: disable-error-code="operator"
from typing import Optional, Union

import torch
from torch import Tensor

from ..cosmology import Cosmology
from ..utils import derotate, translate_rotate
from .base import ThinLens
from ..parametrized import unpack
from ..packed import Packed

__all__ = ("EPL",)


class EPL(ThinLens):
    """
    Elliptical power law (EPL, aka singular power-law ellipsoid) profile.

    This class represents a thin gravitational lens model
    with an elliptical power law profile.
    The lensing equations are solved iteratively
    using an approach based on Tessore et al. 2015.

    Attributes
    ----------
    n_iter: int
        Number of iterations for the iterative solver.
    s: float
        Softening length for the elliptical power-law profile.


        *Unit: arcsec*


    Parameters
    ----------
    z_l: Optional[Union[Tensor, float]]
        This is the redshift of the lens.
        In the context of gravitational lensing,
        the lens is the galaxy or other mass distribution
        that is bending the light from a more distant source.

        *Unit: unitless*

    x0 and y0: Optional[Union[Tensor, float]]
        These are the coordinates of the lens center in the lens plane.
        The lens plane is the plane perpendicular to the line of sight
        in which the deflection of light by the lens is considered.

        *Unit: arcsec*

    q: Optional[Union[Tensor, float]]
        This is the axis ratio of the lens, i.e., the ratio
        of the minor axis to the major axis of the elliptical lens.

        *Unit: unitless*

    phi: Optional[Union[Tensor, float]]
        This is the orientation of the lens on the sky,
        typically given as an angle measured counter-clockwise
        from some reference direction.

        *Unit: radians*

    b: Optional[Union[Tensor, float]]
        This is the scale length of the lens,
        which sets the overall scale of the lensing effect.
        In some contexts, this is referred to as the Einstein radius.

        *Unit: arcsec*

    t: Optional[Union[Tensor, float]]
        This is the power-law slope parameter of the lens model.
        In the context of the EPL model,
        t is equivalent to the gamma parameter minus one,
        where gamma is the power-law index
        of the radial mass distribution of the lens.

        *Unit: unitless*

    """

    _null_params = {
        "x0": 0.0,
        "y0": 0.0,
        "q": 0.5,
        "phi": 0.0,
        "b": 1.0,
        "t": 1.0,
    }

    def __init__(
        self,
        cosmology: Cosmology,
        z_l: Optional[Union[Tensor, float]] = None,
        x0: Optional[Union[Tensor, float]] = None,
        y0: Optional[Union[Tensor, float]] = None,
        q: Optional[Union[Tensor, float]] = None,
        phi: Optional[Union[Tensor, float]] = None,
        b: Optional[Union[Tensor, float]] = None,
        t: Optional[Union[Tensor, float]] = None,
        s: float = 0.0,
        n_iter: int = 18,
        name: Optional[str] = None,
    ):
        """
        Initialize an EPL lens model.

        Parameters
        -----------
        name: string
            Name of the lens model.
        cosmology: Cosmology
            Cosmology object that provides cosmological distance calculations.
        z_l: Optional[Tensor]
            Redshift of the lens.
            If not provided, it is considered as a free parameter.

            *Unit: unitless*

        x0: Optional[Tensor]
            X coordinate of the lens center.
            If not provided, it is considered as a free parameter.

            *Unit: arcsec*

        y0: Optional[Tensor]
            Y coordinate of the lens center.
            If not provided, it is considered as a free parameter.

            *Unit: arcsec*

        q: Optional[Tensor]
            Axis ratio of the lens.
            If not provided, it is considered as a free parameter.

            *Unit: unitless*

        phi: Optional[Tensor]
            Position angle of the lens.
            If not provided, it is considered as a free parameter.

            *Unit: radians*

        b: Optional[Tensor]
            Scale length of the lens.
            If not provided, it is considered as a free parameter.

            *Unit: arcsec*

        t: Optional[Tensor]
            Power law slope (`gamma-1`) of the lens.
            If not provided, it is considered as a free parameter.

            *Unit: unitless*

        s: float
            Softening length for the elliptical power-law profile.

            *Unit: arcsec*

        n_iter: int
            Number of iterations for the iterative solver.
        """
        super().__init__(cosmology, z_l, name=name)

        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("b", b)
        self.add_param("t", t)
        self.s = s

        self.n_iter = n_iter

    @unpack
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the reduced deflection angles of the lens.

        Parameters
        ----------
        x: Tensor
            X coordinates in the lens plane.

            *Unit: arcsec*

        y: Tensor
            Y coordinates in the lens plane.

            *Unit: arcsec*

        z_s: Tensor
            Source redshifts.

            *Unit: unitless*

        params: (Packed, optional)
            Dynamic parameter container for the lens model.

        Returns
        --------
        x_component: Tensor
            Deflection Angle

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle

            *Unit: arcsec*

        """
        x, y = translate_rotate(x, y, x0, y0, phi)

        # follow Tessore et al 2015 (eq. 5)
        z = q * x + y * 1j
        r = torch.abs(z)

        # Tessore et al 2015 (eq. 23)
        r_omega = self._r_omega(z, t, q)
        alpha_c = 2.0 / (1.0 + q) * (b / r) ** t * r_omega  # fmt: skip

        alpha_real = torch.nan_to_num(alpha_c.real, posinf=10**10, neginf=-(10**10))
        alpha_imag = torch.nan_to_num(alpha_c.imag, posinf=10**10, neginf=-(10**10))
        return derotate(alpha_real, alpha_imag, phi)

    def _r_omega(self, z, t, q):
        """
        Iteratively computes `R * omega(phi)` (eq. 23 in Tessore et al 2015).

        Parameters
        ----------
        z: Tensor
            `R * e^(i * phi)`, position vector in the lens plane.

            *Unit: meters*

        t: Tensor
            Power law slow (`gamma-1`).

            *Unit: unitless*

        q: Tensor
            Axis ratio.

            *Unit: unitless*

        Returns
        --------
        Tensor
            The value of `R * omega(phi)`.

            *Unit: arcsec*

        """
        # constants
        f = (1.0 - q) / (1.0 + q)
        phi = z / torch.conj(z)

        # first term in series
        omega_i = z
        part_sum = omega_i

        for i in range(1, self.n_iter):
            factor = (2.0 * i - (2.0 - t)) / (2.0 * i + (2.0 - t))  # fmt: skip
            omega_i = -f * factor * phi * omega_i  # fmt: skip
            part_sum = part_sum + omega_i  # fmt: skip

        return part_sum

    @unpack
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Compute the lensing potential of the lens.

        Parameters
        ----------
        x: Tensor
            X coordinates in the lens plane.

            *Unit: arcsec*

        y: Tensor
            Y coordinates in the lens plane.

            *Unit: arcsec*

        z_s: Tensor
            Source redshifts.

            *Unit: unitless*

        params: (Packed, optional)
            Dynamic parameter container for the lens model.

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        ax, ay = self.reduced_deflection_angle(x, y, z_s, params)
        ax, ay = derotate(ax, ay, -phi)
        x, y = translate_rotate(x, y, x0, y0, phi)
        return (x * ax + y * ay) / (2 - t)  # fmt: skip

    @unpack
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the convergence of the lens, which describes the local density of the lens.

        Parameters
        ----------
        x: Tensor
            X coordinates in the lens plane.

            *Unit: arcsec*

        y: Tensor
            Y coordinates in the lens plane.

            *Unit: arcsec*

        z_s: Tensor
            Source redshifts.

            *Unit: unitless*

        params: (Packed, optional)
            Dynamic parameter container for the lens model.

        Returns
        -------
        Tensor
            The convergence of the lens.

            *Unit: unitless*

        """
        x, y = translate_rotate(x, y, x0, y0, phi)
        psi = (q**2 * (x**2 + self.s**2) + y**2).sqrt()  # fmt: skip
        return (2 - t) / 2 * (b / psi) ** t  # fmt: skip
