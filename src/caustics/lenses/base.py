# mypy: disable-error-code="call-overload"
from abc import abstractmethod
from typing import Optional, Union, Annotated
import warnings

from caskade import Module, Param, forward

from ..backend_obj import backend, ArrayLike
from ..cosmology import Cosmology
from .utils import magnification
from . import func

__all__ = ("ThinLens", "ThickLens")

CosmologyType = Annotated[
    Cosmology,
    "Cosmology object that encapsulates cosmological parameters and distances",
]
NameType = Annotated[Optional[str], "Name of the lens model"]
ZType = Annotated[
    Optional[Union[ArrayLike, float]],
    "The redshift of an object in the lens system",
    True,
]


class Lens(Module):
    """
    Base class for all lenses
    """

    def __init__(
        self, cosmology: CosmologyType, name: NameType = None, z_s: ZType = None
    ):
        """
        Initializes a new instance of the Lens class.

        Parameters
        ----------
        name: string
            The name of the lens model.

        cosmology: Cosmology
            An instance of a Cosmology class that describes the cosmological
            parameters of the model.

        z_s: ZType
            Redshift of the source. Needed for various lensing calculations so
            z_s is held by the lens object rather than the source object.
        """
        super().__init__(name)
        self.cosmology = cosmology
        self.z_s = Param("z_s", z_s, units="unitless", valid=(0, None))

    @forward
    def jacobian_lens_equation(
        self,
        x: ArrayLike,
        y: ArrayLike,
        method="autograd",
        pixelscale=None,
        **kwargs,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the lensing equation at specified points.
        This equates to a (2,2) matrix at each (x,y) point.

        method: autograd or fft
        """

        if method == "autograd":
            return self._jacobian_lens_equation_autograd(x, y, **kwargs)
        elif method == "finitediff":
            if pixelscale is None:
                raise ValueError(
                    "Finite differences lensing jacobian requires regular grid "
                    "and known pixelscale. "
                    "Please include the pixelscale argument"
                )
            return self._jacobian_lens_equation_finitediff(x, y, pixelscale, **kwargs)
        else:
            raise ValueError("method should be one of: autograd, finitediff")

    @forward
    def shear(
        self,
        x: ArrayLike,
        y: ArrayLike,
        method="autograd",
        pixelscale: Optional[ArrayLike] = None,
    ):
        """
        General shear calculation for a lens model using the jacobian of the
        lens equation. Individual lenses may implement more efficient methods.
        """
        A = self.jacobian_lens_equation(x, y, method=method, pixelscale=pixelscale)
        I = backend.eye(2, device=A.device, dtype=A.dtype).reshape(  # noqa E741
            *[1] * len(A.shape[:-2]), 2, 2
        )
        negPsi = (
            0.5
            * backend.unsqueeze(backend.unsqueeze(A[..., 0, 0] + A[..., 1, 1], -1), -1)
            * I
            - A
        )
        return 0.5 * (negPsi[..., 0, 0] - negPsi[..., 1, 1]), negPsi[..., 0, 1]

    @forward
    def magnification(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> ArrayLike:
        """
        Compute the gravitational magnification at the given coordinates.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            Gravitational magnification at the given coordinates.

            *Unit: unitless*

        """
        return magnification(self.raytrace, x, y)

    @forward
    def forward_raytrace(
        self,
        bx: ArrayLike,
        by: ArrayLike,
        epsilon: float = 1e-3,
        x0: Optional[ArrayLike] = None,
        y0: Optional[ArrayLike] = None,
        fov: float = 5.0,
        divisions: int = 100,
        max_depth: int = 25,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Perform a forward ray-tracing operation which maps from the source plane
        to the image plane.

        Parameters
        ----------
        bx: ArrayLike
            ArrayLike of x coordinate in the source plane.

            *Unit: arcsec*

        by: ArrayLike
            ArrayLike of y coordinate in the source plane.

            *Unit: arcsec*

        epsilon: ArrayLike
            maximum distance between two images (arcsec) before they are
            considered the same image.

            *Unit: arcsec*

        fov: float
            the field of view in which the initial random samples are taken.

            *Unit: arcsec*

        divisions: int
            the number of divisions of the fov on each axis when constructing
            the grid to perform in the triangle search.

        Returns
        -------
        x_component: ArrayLike
            x-coordinate ArrayLike of the ray-traced light rays

            *Unit: arcsec*

        y_component: ArrayLike
            y-coordinate ArrayLike of the ray-traced light rays

            *Unit: arcsec*
        """
        if x0 is None:
            x0 = backend.zeros((), device=bx.device, dtype=bx.dtype)
        if y0 is None:
            y0 = backend.zeros((), device=by.device, dtype=by.dtype)

        return func.forward_raytrace(
            backend.stack((bx, by)),
            self.raytrace,
            x0,
            y0,
            fov,
            divisions,
            epsilon,
            max_depth,
        )


class ThickLens(Lens):
    """
    Base class for modeling gravitational lenses that cannot be
    treated using the thin lens approximation.
    It is an abstract class and should be subclassed
    for different types of lens models.

    Attributes
    ----------
    cosmology: Cosmology
        An instance of a Cosmology class that describes
        the cosmological parameters of the model.
    """

    @forward
    def reduced_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        ThickLens objects do not have a reduced deflection angle
        since the distance D_ls is undefined

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: unitless*

        """
        warnings.warn(
            "ThickLens objects do not have a reduced deflection angle "
            "since they have no unique lens redshift. "
            "The distance D_{ls} is undefined in the equation "
            "$\\alpha_{reduced} = \\frac{D_{ls}}{D_s}\\alpha_{physical}$."
            "See `effective_reduced_deflection_angle`. "
            "Now using effective_reduced_deflection_angle, "
            "please switch functions to remove this warning"
        )
        return self.effective_reduced_deflection_angle(x, y, **kwargs)

    @forward
    def effective_reduced_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> tuple[ArrayLike, ArrayLike]:
        """ThickLens objects do not have a reduced deflection angle since the
        distance D_ls is undefined. Instead we define an effective
        reduced deflection angle by simply assuming the relation
        $\alpha = \theta - \beta$ holds, where $\alpha$ is the
        effective reduced deflection angle, $\theta$ are the observed
        angular coordinates, and $\beta$ are the angular coordinates
        to the source plane.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        """
        bx, by = self.raytrace(x, y, **kwargs)
        return x - bx, y - by

    @forward
    def physical_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *args,
        **kwargs,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Physical deflection angles are computed with respect to a lensing
        plane. ThickLens objects have no unique definition of a lens
        plane and so cannot compute a physical_deflection_angle

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        x_component: ArrayLike
            Deflection Angle in x direction.

            *Unit: arcsec*

        y_component: ArrayLike
            Deflection Angle in y direction.

            *Unit: arcsec*

        """
        raise NotImplementedError(
            "Physical deflection angles are computed with respect to a lensing plane. "
            "ThickLens objects have no unique definition of a lens plane "
            "and so cannot compute a physical_deflection_angle"
        )

    @abstractmethod
    @forward
    def raytrace(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *args,
        **kwargs,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Performs ray tracing by computing the angular position on the
        source plance associated with a given input observed angular
        coordinate x,y.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        x: ArrayLike
            x coordinate ArrayLike of the ray-traced light rays

            *Unit: arcsec*

        y: ArrayLike
            y coordinate ArrayLike of the ray-traced light rays

            *Unit: arcsec*

        """
        ...

    @abstractmethod
    @forward
    def surface_density(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *args,
        **kwargs,
    ) -> ArrayLike:
        """
        Computes the projected mass density at given coordinates.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The projected mass density at the given coordinates
            in units of solar masses per square Mpc.

            *Unit: Msun/Mpc^2*

        """
        ...

    @abstractmethod
    @forward
    def time_delay(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *args,
        **kwargs,
    ) -> ArrayLike:
        """
        Computes the gravitational time delay at given coordinates.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            The gravitational time delay at the given coordinates.

            *Unit: seconds*

        """
        ...

    @forward
    def _jacobian_effective_deflection_angle_finitediff(
        self,
        x: ArrayLike,
        y: ArrayLike,
        pixelscale: ArrayLike,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the effective reduced deflection angle vector field.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Compute deflection angles
        ax, ay = self.effective_reduced_deflection_angle(x, y)

        # Build Jacobian
        J = backend.zeros((*ax.shape, 2, 2), device=ax.device, dtype=ax.dtype)
        grad_ax = backend.gradient(ax, spacing=pixelscale)
        grad_ay = backend.gradient(ay, spacing=pixelscale)
        J = backend.fill_at_indices(J, (Ellipsis, 0, 1), grad_ax[0])
        J = backend.fill_at_indices(J, (Ellipsis, 0, 0), grad_ax[1])
        J = backend.fill_at_indices(J, (Ellipsis, 1, 1), grad_ay[0])
        J = backend.fill_at_indices(J, (Ellipsis, 1, 0), grad_ay[1])
        return J

    @forward
    def _jacobian_effective_deflection_angle_autograd(
        self,
        x: ArrayLike,
        y: ArrayLike,
        chunk_size: Optional[int] = None,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the effective reduced deflection angle vector field.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        J = backend.vmap(
            backend.jacfwd(
                self.effective_reduced_deflection_angle,
                argnums=(0, 1),
                randomness="different",
            ),
            chunk_size=chunk_size,
        )(backend.flatten(x), backend.flatten(y))
        J = backend.stack([backend.stack(Jrow, dim=-1) for Jrow in J], dim=-2)
        return J.reshape(*x.shape, 2, 2)

    @forward
    def jacobian_effective_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        method="autograd",
        pixelscale=None,
        **kwargs,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the effective reduced deflection angle vector field.
        This equates to a (2,2) matrix at each (x,y) point.

        method: autograd or fft
        """

        if method == "autograd":
            return self._jacobian_effective_deflection_angle_autograd(x, y, **kwargs)
        elif method == "finitediff":
            if pixelscale is None:
                raise ValueError(
                    "Finite differences lensing jacobian requires "
                    "regular grid and known pixelscale. "
                    "Please include the pixelscale argument"
                )
            return self._jacobian_effective_deflection_angle_finitediff(
                x, y, pixelscale, **kwargs
            )
        else:
            raise ValueError("method should be one of: autograd, finitediff")

    @forward
    def _jacobian_lens_equation_finitediff(
        self,
        x: ArrayLike,
        y: ArrayLike,
        pixelscale: ArrayLike,
        **kwargs,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the lensing equation at specified points.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Build Jacobian
        J = self._jacobian_effective_deflection_angle_finitediff(
            x, y, pixelscale, **kwargs
        )
        return backend.to(backend.eye(2), device=J.device) - J

    @forward
    def _jacobian_lens_equation_autograd(
        self,
        x: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the lensing equation at specified points.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Build Jacobian
        J = self._jacobian_effective_deflection_angle_autograd(x, y, **kwargs)
        return backend.to(backend.eye(2), device=J.device) - backend.detach(J)

    @forward
    def effective_convergence_div(
        self,
        x: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> ArrayLike:
        """
        Using the divergence of the effective reduced delfection angle
        we can compute the divergence component of the effective convergence field.
        This field produces a single plane convergence field
        which reproduces as much of the deflection field
        as possible for a single plane.

        See: https://arxiv.org/pdf/2006.07383.pdf
        see also the `effective_convergence_curl` method.
        """
        J = self.jacobian_effective_deflection_angle(x, y, **kwargs)
        return 0.5 * (J[..., 0, 0] + J[..., 1, 1])

    @forward
    def effective_convergence_curl(
        self,
        x: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> ArrayLike:
        """
        Use the curl of the effective reduced deflection angle vector field
        to compute an effective convergence which derives specifically
        from the curl of the deflection field.
        This field is purely a result of multiplane lensing
        and cannot occur in single plane lensing.

        See: https://arxiv.org/pdf/2006.07383.pdf
        """
        J = self.jacobian_effective_deflection_angle(x, y, **kwargs)
        return 0.5 * (J[..., 1, 0] - J[..., 0, 1])


class ThinLens(Lens):
    """Base class for thin gravitational lenses.

    This class provides an interface for thin gravitational lenses,
    i.e., lenses that can be modeled using the thin lens
    approximation.  The class provides methods to compute several
    lensing quantities such as the deflection angle, convergence,
    potential, surface mass density, and gravitational time delay.

    Attributes
    ----------
    name: string
        Name of the lens model.

    cosmology: Cosmology
        Cosmology object that encapsulates cosmological parameters and distances.

    z_l: (Optional[ArrayLike], optional)
        Redshift of the lens. Defaults to None.

        *Unit: unitless*

    """

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZType = None,
        z_s: ZType = None,
        name: NameType = None,
    ):
        super().__init__(cosmology=cosmology, name=name, z_s=z_s)
        self.z_l = Param("z_l", z_l, units="unitless", valid=(0, None))

    @forward
    def reduced_deflection_angle(
        self, x: ArrayLike, y: ArrayLike, chunk_size: Optional[int] = None
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Computes the reduced deflection angle of the lens at given coordinates [arcsec].

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        chunk_size: int
            Chunk size for the autograd computation.

            *Unit: number*

        Returns
        --------
        x_component: ArrayLike
            Deflection Angle in the x-direction.

            *Unit: arcsec*

        y_component: ArrayLike
            Deflection Angle in the y-direction.

            *Unit: arcsec*

        """
        ax, ay = backend.vmap(
            backend.grad(self.potential, (0, 1)), chunk_size=chunk_size
        )(backend.flatten(x), backend.flatten(y))
        return ax.reshape(x.shape), ay.reshape(y.shape)

    @forward
    def physical_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z_s: Annotated[ArrayLike, "Param"],
        z_l: Annotated[ArrayLike, "Param"],
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Computes the physical deflection angle immediately after passing through this lens's plane.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        x_component: ArrayLike
            Deflection Angle in x-direction.

            *Unit: arcsec*

        y_component: ArrayLike
            Deflection Angle in y-direction.

            *Unit: arcsec*

        """
        d_s = self.cosmology.angular_diameter_distance(z_s)
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)
        deflection_angle_x, deflection_angle_y = self.reduced_deflection_angle(x, y)
        return func.physical_from_reduced_deflection_angle(
            deflection_angle_x, deflection_angle_y, d_s, d_ls
        )

    @abstractmethod
    @forward
    def convergence(
        self,
        x: ArrayLike,
        y: ArrayLike,
        chunk_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> ArrayLike:
        """
        Computes the convergence of the lens at given coordinates.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        chunk_size: int
            Chunk size for the autograd computation.

            *Unit: number*

        Returns
        -------
        ArrayLike
            Dimensionless convergence, normalized by the critical surface density at the lens plane

            *Unit: unitless*

        """
        Psi_H = backend.vmap(
            backend.hessian(self.potential, (0, 1)),
            chunk_size=chunk_size,
        )(backend.flatten(x), backend.flatten(y))
        Psi_H = backend.stack([backend.stack(Hrow, dim=-1) for Hrow in Psi_H], dim=-2)
        Psi_H = Psi_H.reshape(*x.shape, 2, 2)
        return 0.5 * (Psi_H[..., 0, 0] + Psi_H[..., 1, 1]).reshape(x.shape)

    @abstractmethod
    @forward
    def potential(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *args,
        **kwargs,
    ) -> ArrayLike:
        """
        Computes the gravitational lensing potential at given coordinates.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            Gravitational lensing potential at the given coordinates in arcsec^2.

            *Unit: arsec^2*

        """
        ...

    @forward
    def surface_density(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z_s: Annotated[ArrayLike, "Param"],
        z_l: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Computes the surface mass density of the lens at given coordinates.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        ArrayLike
            Surface mass density at the given coordinates in solar masses per Mpc^2.

            *Unit: Msun/Mpc^2*

        """
        critical_surface_density = self.cosmology.critical_surface_density(z_l, z_s)
        return self.convergence(x, y) * critical_surface_density  # fmt: skip

    @forward
    def raytrace(
        self,
        x: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Perform a ray-tracing operation by subtracting
        the deflection angles from the input coordinates.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        Returns
        -------
        x_component: ArrayLike
            Deflection Angle in x direction.

            *Unit: arcsec*

        y_component: ArrayLike
            Deflection Angle in y direction.

            *Unit: arcsec*

        """
        ax, ay = self.reduced_deflection_angle(x, y, **kwargs)
        return x - ax, y - ay

    @forward
    def _arcsec2_to_days(
        self, z_s: Annotated[ArrayLike, "Param"], z_l: Annotated[ArrayLike, "Param"]
    ):
        """
        This method is used by :func:`caustics.lenses.ThinLens.time_delay` to
        convert arcsec^2 to days in the context of gravitational time delays.
        """
        d_l = self.cosmology.angular_diameter_distance(z_l)
        d_s = self.cosmology.angular_diameter_distance(z_s)
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)
        return func.time_delay_arcsec2_to_days(d_l, d_s, d_ls, z_l)

    @forward
    def time_delay(
        self,
        x: ArrayLike,
        y: ArrayLike,
        shapiro_time_delay: bool = True,
        geometric_time_delay: bool = True,
    ) -> ArrayLike:
        """
        Computes the gravitational time delay for light passing through the lens at given coordinates.

        This time delay is induced by the photons traveling through a gravitational potential well (Shapiro time delay) plus the effect of the increased path length that the photons must traverse (geometric time delay).
        The main equation involved here is the following:

        .. math::

            \\Delta t = \\frac{1 + z_l}{c} \\frac{D_s}{D_l D_{ls}} \\left[ \\frac{1}{2}|\\vec{\\alpha}(\\vec{\\theta})|^2 - \\psi(\\vec{\\theta}) \\right]

        where :math:`\\vec{\\alpha}(\\vec{\\theta})` is the deflection angle,
        :math:`\\psi(\\vec{\\theta})` is the lensing potential,
        :math:`D_l` is the comoving distance to the lens,
        :math:`D_s` is the comoving distance to the source,
        and :math:`D_{ls}` is the comoving distance between the lens and the source. In the above equation, the first term is the geometric time delay and the second term is the gravitational time delay.

        Parameters
        ----------
        x: ArrayLike
            ArrayLike of x coordinates in the lens plane.

            *Unit: arcsec*

        y: ArrayLike
            ArrayLike of y coordinates in the lens plane.

            *Unit: arcsec*

        shapiro_time_delay: bool
            Whether to include the Shapiro time delay component.

        geometric_time_delay: bool
            Whether to include the geometric time delay component.

        Returns
        -------
        ArrayLike
            Time delay at the given coordinates.

            *Unit: days*

        References
        ----------
        1. Irwin I. Shapiro (1964). "Fourth Test of General Relativity". Physical Review Letters. 13 (26): 789-791
        2. Refsdal, S. (1964). "On the possibility of determining Hubble's parameter and the masses of galaxies from the gravitational lens effect". Monthly Notices of the Royal Astronomical Society. 128 (4): 307-310.
        """
        TD = backend.zeros_like(x)

        if shapiro_time_delay:
            potential = self.potential(x, y)
            TD = TD - potential
        if geometric_time_delay:
            ax, ay = self.reduced_deflection_angle(x, y)
            fp = 0.5 * (ax**2 + ay**2)
            TD = TD + fp

        factor = self._arcsec2_to_days()

        return factor * TD

    @forward
    def _jacobian_deflection_angle_finitediff(
        self,
        x: ArrayLike,
        y: ArrayLike,
        pixelscale: ArrayLike,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the deflection angle vector.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Compute deflection angles
        ax, ay = self.reduced_deflection_angle(x, y)

        # Build Jacobian
        J = backend.zeros((*ax.shape, 2, 2), device=ax.device, dtype=ax.dtype)
        grad_ax = backend.gradient(ax, spacing=pixelscale)
        grad_ay = backend.gradient(ay, spacing=pixelscale)
        J = backend.fill_at_indices(J, (Ellipsis, 0, 1), grad_ax[0])
        J = backend.fill_at_indices(J, (Ellipsis, 0, 0), grad_ax[1])
        J = backend.fill_at_indices(J, (Ellipsis, 1, 1), grad_ay[0])
        J = backend.fill_at_indices(J, (Ellipsis, 1, 0), grad_ay[1])
        return J

    @forward
    def _jacobian_deflection_angle_autograd(
        self,
        x: ArrayLike,
        y: ArrayLike,
        chunk_size: Optional[int] = None,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the deflection angle vector.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Compute deflection angle gradients
        J = backend.vmap(
            backend.jacfwd(
                self.reduced_deflection_angle, argnums=(0, 1), randomness="different"
            ),
            chunk_size=chunk_size,
        )(backend.flatten(x), backend.flatten(y))
        J = backend.stack([backend.stack(Jrow, dim=-1) for Jrow in J], dim=-2)
        return J.reshape(*x.shape, 2, 2)

    @forward
    def jacobian_deflection_angle(
        self,
        x: ArrayLike,
        y: ArrayLike,
        method="autograd",
        pixelscale=None,
        chunk_size: Optional[int] = None,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the deflection angle vector.
        This equates to a (2,2) matrix at each (x,y) point.

        method: autograd or fft
        """

        if method == "autograd":
            return self._jacobian_deflection_angle_autograd(x, y, chunk_size)
        elif method == "finitediff":
            if pixelscale is None:
                raise ValueError(
                    "Finite differences lensing jacobian requires regular grid "
                    "and known pixelscale. Please include the pixelscale argument"
                )
            return self._jacobian_deflection_angle_finitediff(x, y, pixelscale)
        else:
            raise ValueError("method should be one of: autograd, finitediff")

    @forward
    def _jacobian_lens_equation_finitediff(
        self,
        x: ArrayLike,
        y: ArrayLike,
        pixelscale: ArrayLike,
        **kwargs,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the lensing equation at specified points.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Build Jacobian
        J = self._jacobian_deflection_angle_finitediff(x, y, pixelscale, **kwargs)
        return backend.to(backend.eye(2), device=J.device) - J

    @forward
    def _jacobian_lens_equation_autograd(
        self,
        x: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> tuple[tuple[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]:
        """
        Return the jacobian of the lensing equation at specified points.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Build Jacobian
        J = self._jacobian_deflection_angle_autograd(x, y, **kwargs)
        return backend.to(backend.eye(2), device=J.device) - backend.detach(J)
