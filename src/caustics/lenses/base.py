# mypy: disable-error-code="call-overload"
from abc import abstractmethod
from typing import Optional, Union, Annotated
import warnings

from caskade import Module, Param, forward

from ..backend_obj import backend, ArrayLike
from ..cosmology import Cosmology
from ..jit import jit_forward
from .utils import magnification
from . import func
from .func.adaptive.images import _forward_raytrace_at_user

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
        I = backend.eye(
            2, device=backend.device(A), dtype=A.dtype
        ).reshape(  # noqa E741
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
    def build_closed_lens_mesh(
        self,
        fov: float,
        init_res: int,
        min_img_sep: float,
        *,
        mu_min: Optional[float] = None,
        mu_max: Optional[float] = None,
        max_depth: int = 25,
        growth: float = 1.5,
        max_iters: int = 10,
        fov_x0: float = 0.0,
        fov_y0: float = 0.0,
        centers: Optional[ArrayLike] = None,
        device=None,
        batch_size: Optional[int] = None,
    ) -> tuple[
        func.LensMesh,
        func.CriticalCurvesAndCaustics,
        Optional[func.MagnificationRegion],
    ]:
        """
        Build a lens mesh whose fov cuts no critical curve and, given ``mu_min``, the region ``mu_min <= mu_tot <= mu_max``.

        Without ``mu_min`` this is
        :func:`~caustics.lenses.func.build_closed_lens_mesh`, the region is
        None and ``mu_max`` is unused. With it, this is
        :func:`~caustics.lenses.func.build_closed_magnification_region`, which
        grows the fov further until the region is bounded. The lens map is
        ``self.raytrace`` and its Jacobian ``self.jacobian_lens_equation``, at
        the parameter values of this call.

        The center of every component, ``self.component_centers()``, is given
        a hole, and so is each of ``centers``. A hole drops no image. Where the
        lens map is smooth, as at a shear's, a mass sheet's or a cored
        profile's center, its hole curve is a speck and never a
        pseudo-caustic, so the hole changes only a critical curve crossing its
        disk, which is re-joined round it. A component centered near or
        outside ``fov`` widens it.

        Under jax, the map and its Jacobian run through :func:`~caustics.jit_forward`,
        compiled once per lens object and padded batch size and reused at
        other parameter values; a ``ThickLens`` runs eagerly.

        Parameters
        ----------
        fov: float
            Side of the square lens-plane domain to start from.

            *Unit: arcsec*

        init_res: int
            Level-0 cells per axis. Must resolve the smallest structure of the
            lens.

        min_img_sep: float
            Lens-plane tolerance: the mesh refines to half of it, and images
            closer than half of it are one image.

            *Unit: arcsec*

        mu_min: Optional[float]
            Lowest total magnification in the region. None builds no region.

        mu_max: Optional[float]
            Highest total magnification in the region; unbounded when None.

        max_depth: int
            Hard cap on the refinement level.

        growth: float
            Factor each extension asks to multiply the fov by.

        max_iters: int
            Most extensions to run in each growing loop.

        fov_x0: float
            x coordinate of the center of the starting fov.

            *Unit: arcsec*

        fov_y0: float
            y coordinate of the center of the starting fov.

            *Unit: arcsec*

        centers: Optional[ArrayLike]
            ``(S, 2)`` positions where the lens map may jump besides the
            component centers, each given a hole too: the image-plane position
            of a singular center in a later plane of a multiplane lens.

            *Unit: arcsec*

        device: Optional
            Device the lens lives on; the mesh is stored there.

        batch_size: Optional[int]
            Most points per lens call.

        Returns
        -------
        mesh: LensMesh
        curves: CriticalCurvesAndCaustics
            The mesh's critical curves, caustics and pseudo-caustics.
        region: Optional[MagnificationRegion]
            None without ``mu_min``.
        """
        holes = backend.to(self.component_centers(), dtype=backend.float64)
        if centers is not None:
            centers = backend.as_array(
                centers, dtype=backend.float64, device=backend.device(holes)
            )
            holes = backend.concatenate((holes, centers), dim=0)
        # Not x0, y0: @forward would fill those from the lens's own x0, y0.
        build = dict(
            max_depth=max_depth,
            growth=growth,
            max_iters=max_iters,
            x0=fov_x0,
            y0=fov_y0,
            centers=holes,
            device=device,
            batch_size=batch_size,
        )
        if mu_min is None:
            mesh, curves = func.build_closed_lens_mesh(
                jit_forward(self.raytrace),
                jit_forward(self.jacobian_lens_equation),
                fov,
                init_res,
                min_img_sep,
                **build,
            )
            return mesh, curves, None
        return func.build_closed_magnification_region(
            jit_forward(self.raytrace),
            jit_forward(self.jacobian_lens_equation),
            fov,
            init_res,
            min_img_sep,
            mu_min,
            mu_max,
            **build,
        )

    @forward
    def forward_raytrace(
        self,
        bx: ArrayLike,
        by: ArrayLike,
        *,
        mesh: Optional[func.LensMesh] = None,
        fov: Optional[float] = None,
        init_res: Optional[int] = None,
        min_img_sep: Optional[float] = None,
        max_depth: int = 25,
        growth: float = 1.5,
        max_iters: int = 10,
        fov_x0: float = 0.0,
        fov_y0: float = 0.0,
        centers: Optional[ArrayLike] = None,
        device=None,
        batch_size: Optional[int] = None,
        residual_tol: float = 1e-6,
        lm_kwargs: Optional[dict] = None,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Lens-plane positions of every image of each source-plane point.

        Without ``mesh``, :meth:`build_closed_lens_mesh` builds one first from
        ``fov``, ``init_res``, ``min_img_sep`` and the build arguments after
        them. A ``mesh`` passed in is used as is and the build arguments are
        unused; it must come from this lens at the same parameter values.
        Building it once and passing it to many calls saves the build. The
        images are found by :func:`~caustics.lenses.func.forward_raytrace`.
        Under jax, the solve is compiled once per lens object and padded
        size, and reused at other parameter values.

        Parameters
        ----------
        bx: ArrayLike
            Source-plane x coordinates, any shape, flattened to ``(B,)``.

            *Unit: arcsec*

        by: ArrayLike
            Source-plane y coordinates, the shape of ``bx``.

            *Unit: arcsec*

        mesh: Optional[LensMesh]
            A lens mesh of this lens; built when None.

        fov, init_res, min_img_sep, max_depth, growth, max_iters, fov_x0, fov_y0, centers, device:
            As for :meth:`build_closed_lens_mesh`; used only to build a mesh.

        batch_size: Optional[int]
            Most points per lens call in the build, and most sources per chunk
            in the solve.

        residual_tol: float
            Largest ``|raytrace(x) - beta|`` of a kept image.

            *Unit: arcsec*

        lm_kwargs: Optional[dict]
            Extra keyword arguments for :func:`~caustics.utils.batch_lm`.

        Returns
        -------
        x: ArrayLike
            ``(K,)`` float64 image x coordinates, source by source: the
            ``counts[b]`` images of source ``b`` follow those of sources
            ``0 .. b - 1``.

            *Unit: arcsec*

        y: ArrayLike
            ``(K,)`` float64 image y coordinates, in the order of ``x``.

            *Unit: arcsec*

        counts: ArrayLike
            ``(B,)`` int64 images per source.

        Raises
        ------
        ValueError
            If any source lies outside the image of the mesh's fov boundary,
            where some of its images can lie outside the fov. The closed build
            grows the fov only until it cuts no critical curve, so a far
            source needs a larger ``fov``, or a ``mesh`` that covers it.
        """
        if mesh is None:
            mesh, _, _ = self.build_closed_lens_mesh(
                fov,
                init_res,
                min_img_sep,
                max_depth=max_depth,
                growth=growth,
                max_iters=max_iters,
                fov_x0=fov_x0,
                fov_y0=fov_y0,
                centers=centers,
                device=device,
                batch_size=batch_size,
            )
        raytrace = jit_forward(self.raytrace)
        lm_map = (raytrace.on_points, (raytrace.values(),)) if raytrace.jitted else None
        return _forward_raytrace_at_user(
            bx,
            by,
            raytrace,
            mesh,
            lm_map,
            batch_size=batch_size,
            residual_tol=residual_tol,
            lm_kwargs=lm_kwargs,
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

    # `jit_forward` runs these eagerly: `Multiplane` sorts its planes by
    # redshift in Python, which traced values cannot do.
    _jit_forward_eager = True

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
        J = backend.zeros((*ax.shape, 2, 2), device=backend.device(ax), dtype=ax.dtype)
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
        return backend.to(backend.eye(2), device=backend.device(J)) - J

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
        return backend.to(backend.eye(2), device=backend.device(J)) - backend.detach(J)

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
    def component_centers(
        self,
        x0: Annotated[ArrayLike, "Param"],
        y0: Annotated[ArrayLike, "Param"],
    ) -> ArrayLike:
        """
        Lens-plane center of each component of the lens: of a single profile, its own.

        :meth:`build_closed_lens_mesh` cuts a hole around each.

        Returns
        -------
        ArrayLike
            ``(1, 2)``: ``(x0, y0)``.

            *Unit: arcsec*
        """
        return backend.stack((x0, y0)).reshape(1, 2)

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
        J = backend.zeros((*ax.shape, 2, 2), device=backend.device(ax), dtype=ax.dtype)
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
        return backend.to(backend.eye(2), device=backend.device(J)) - J

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
        return backend.to(backend.eye(2), device=backend.device(J)) - backend.detach(J)
