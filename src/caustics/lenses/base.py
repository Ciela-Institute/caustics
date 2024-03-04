# mypy: disable-error-code="call-overload"
from abc import abstractmethod
from typing import Optional, Union, Annotated, List
from functools import partial
import warnings

import torch
from torch import Tensor

from ..constants import arcsec_to_rad, c_Mpc_s
from ..cosmology import Cosmology
from ..parametrized import Parametrized, unpack
from .utils import get_magnification
from ..utils import batch_lm
from ..packed import Packed

__all__ = ("ThinLens", "ThickLens")

CosmologyType = Annotated[
    Cosmology,
    "Cosmology object that encapsulates cosmological parameters and distances",
]
NameType = Annotated[Optional[str], "Name of the lens model"]
ZLType = Annotated[Optional[Union[Tensor, float]], "The redshift of the lens", True]
LensesType = Annotated[List["ThinLens"], "A list of ThinLens objects"]


class Lens(Parametrized):
    """
    Base class for all lenses
    """

    def __init__(self, cosmology: CosmologyType, name: NameType = None):
        """
        Initializes a new instance of the Lens class.

        Parameters
        ----------
        name: string
            The name of the lens model.
        cosmology: Cosmology
            An instance of a Cosmology class that describes
            the cosmological parametersof the model.
        """
        super().__init__(name)
        self.cosmology = cosmology

    @unpack
    def jacobian_lens_equation(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        method="autograd",
        pixelscale=None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the lensing equation at specified points.
        This equates to a (2,2) matrix at each (x,y) point.

        method: autograd or fft
        """

        if method == "autograd":
            return self._jacobian_lens_equation_autograd(x, y, z_s, params, **kwargs)
        elif method == "finitediff":
            if pixelscale is None:
                raise ValueError(
                    "Finite differences lensing jacobian requires regular grid "
                    "and known pixelscale. "
                    "Please include the pixelscale argument"
                )
            return self._jacobian_lens_equation_finitediff(
                x, y, z_s, pixelscale, params, **kwargs
            )
        else:
            raise ValueError("method should be one of: autograd, finitediff")

    @unpack
    def magnification(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the gravitational magnification at the given coordinates.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        -------
        Tensor
            Gravitational magnification at the given coordinates.
        """
        return get_magnification(partial(self.raytrace, params=params), x, y, z_s)

    @unpack
    def forward_raytrace(
        self,
        bx: Tensor,
        by: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        epsilon=1e-2,
        n_init=100,
        fov=5.0,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform a forward ray-tracing operation which maps from the source plane to the image plane.

        Parameters
        ----------
        bx: Tensor
            Tensor of x coordinate in the source plane (scalar).
        by: Tensor
            Tensor of y coordinate in the source plane (scalar).
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.
        epsilon: Tensor
            maximum distance between two images (arcsec) before they are considered the same image.
        n_init: int
            number of random initialization points used to try and find image plane points.
        fov: float
            the field of view in which the initial random samples are taken.

        Returns
        -------
        tuple[Tensor, Tensor]
            Ray-traced coordinates in the x and y directions.
        """

        bxy = torch.stack((bx, by)).repeat(n_init, 1)  # has shape (n_init, Dout:2)

        # TODO make FOV more general so that it doesn't have to be centered on zero,zero
        if fov is None:
            raise ValueError("fov must be given to generate initial guesses")

        # Random starting points in image plane
        guesses = (torch.as_tensor(fov) * (torch.rand(n_init, 2) - 0.5)).to(
            device=bxy.device
        )  # Has shape (n_init, Din:2)

        # Optimize guesses in image plane
        x, l, c = batch_lm(  # noqa: E741 Unused `l` variable
            guesses,
            bxy,
            lambda *a, **k: torch.stack(
                self.raytrace(a[0][..., 0], a[0][..., 1], *a[1:], **k), dim=-1
            ),
            f_args=(z_s, params),
        )

        # Clip points that didn't converge
        x = x[c < 1e-2 * epsilon**2]

        # Cluster results into n-images
        res = []
        while len(x) > 0:
            res.append(x[0])
            d = torch.linalg.norm(x - x[0], dim=-1)
            x = x[d > epsilon]

        res = torch.stack(res, dim=0)
        return res[..., 0], res[..., 1]


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

    @unpack
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        ThickLens objects do not have a reduced deflection angle
        since the distance D_ls is undefined

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: Packed, optional
            Dynamic parameter container for the lens model. Defaults to None.

        Raises
        ------
        NotImplementedError
        """
        warnings.warn(
            "ThickLens objects do not have a reduced deflection angle "
            "since they have no unique lens redshift. "
            "The distance D_{ls} is undefined in the equation "
            "$\alpha_{reduced} = \frac{D_{ls}}{D_s}\alpha_{physical}$."
            "See `effective_reduced_deflection_angle`. "
            "Now using effective_reduced_deflection_angle, "
            "please switch functions to remove this warning"
        )
        return self.effective_reduced_deflection_angle(x, y, z_s, params, **kwargs)

    @unpack
    def effective_reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """ThickLens objects do not have a reduced deflection angle since the
        distance D_ls is undefined. Instead we define an effective
        reduced deflection angle by simply assuming the relation
        $\alpha = \theta - \beta$ holds, where $\alpha$ is the
        effective reduced deflection angle, $\theta$ are the observed
        angular coordinates, and $\beta$ are the angular coordinates
        to the source plane.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        """
        bx, by = self.raytrace(x, y, z_s, params, **kwargs)
        return x - bx, y - by

    @unpack
    def physical_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Physical deflection angles are computed with respect to a lensing
        plane. ThickLens objects have no unique definition of a lens
        plane and so cannot compute a physical_deflection_angle

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        -------
        tuple[Tensor, Tensor]
            Tuple of Tensors representing the x and y components
            of the deflection angle, respectively.

        """
        raise NotImplementedError(
            "Physical deflection angles are computed with respect to a lensing plane. "
            "ThickLens objects have no unique definition of a lens plane "
            "and so cannot compute a physical_deflection_angle"
        )

    @abstractmethod
    @unpack
    def raytrace(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Performs ray tracing by computing the angular position on the
        source plance associated with a given input observed angular
        coordinate x,y.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        -------
        x: Tensor
            x coordinate Tensor of the ray-traced light rays
        y: Tensor
            y coordinate Tensor of the ray-traced light rays

        """
        ...

    @abstractmethod
    @unpack
    def surface_density(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Computes the projected mass density at given coordinates.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        -------
        Tensor
            The projected mass density at the given coordinates
            in units of solar masses per square Megaparsec.
        """
        ...

    @abstractmethod
    @unpack
    def time_delay(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Computes the gravitational time delay at given coordinates.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor ofsource redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        -------
        Tensor
            The gravitational time delay at the given coordinates.
        """
        ...

    @unpack
    def _jacobian_effective_deflection_angle_finitediff(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        pixelscale: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the effective reduced deflection angle vector field.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Compute deflection angles
        ax, ay = self.effective_reduced_deflection_angle(x, y, z_s, params)

        # Build Jacobian
        J = torch.zeros((*ax.shape, 2, 2), device=ax.device, dtype=ax.dtype)
        J[..., 0, 1], J[..., 0, 0] = torch.gradient(ax, spacing=pixelscale)
        J[..., 1, 1], J[..., 1, 0] = torch.gradient(ay, spacing=pixelscale)
        return J

    @unpack
    def _jacobian_effective_deflection_angle_autograd(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the effective reduced deflection angle vector field.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Ensure the x,y coordinates track gradients
        x = x.detach().requires_grad_()
        y = y.detach().requires_grad_()

        # Compute deflection angles
        ax, ay = self.effective_reduced_deflection_angle(x, y, z_s, params)

        # Build Jacobian
        J = torch.zeros((*ax.shape, 2, 2), device=ax.device, dtype=ax.dtype)
        (J[..., 0, 0],) = torch.autograd.grad(
            ax, x, grad_outputs=torch.ones_like(ax), create_graph=True
        )
        (J[..., 0, 1],) = torch.autograd.grad(
            ax, y, grad_outputs=torch.ones_like(ax), create_graph=True
        )
        (J[..., 1, 0],) = torch.autograd.grad(
            ay, x, grad_outputs=torch.ones_like(ay), create_graph=True
        )
        (J[..., 1, 1],) = torch.autograd.grad(
            ay, y, grad_outputs=torch.ones_like(ay), create_graph=True
        )
        return J.detach()

    @unpack
    def jacobian_effective_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        method="autograd",
        pixelscale=None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the effective reduced deflection angle vector field.
        This equates to a (2,2) matrix at each (x,y) point.

        method: autograd or fft
        """

        if method == "autograd":
            return self._jacobian_effective_deflection_angle_autograd(x, y, z_s, params)
        elif method == "finitediff":
            if pixelscale is None:
                raise ValueError(
                    "Finite differences lensing jacobian requires "
                    "regular grid and known pixelscale. "
                    "Please include the pixelscale argument"
                )
            return self._jacobian_effective_deflection_angle_finitediff(
                x, y, z_s, pixelscale, params
            )
        else:
            raise ValueError("method should be one of: autograd, finitediff")

    @unpack
    def _jacobian_lens_equation_finitediff(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        pixelscale: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the lensing equation at specified points.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Build Jacobian
        J = self._jacobian_effective_deflection_angle_finitediff(
            x, y, z_s, pixelscale, params, **kwargs
        )
        return torch.eye(2).to(J.device) - J

    @unpack
    def _jacobian_lens_equation_autograd(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the lensing equation at specified points.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Build Jacobian
        J = self._jacobian_effective_deflection_angle_autograd(
            x, y, z_s, params, **kwargs
        )
        return torch.eye(2).to(J.device) - J.detach()

    @unpack
    def effective_convergence_div(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Using the divergence of the effective reduced delfection angle
        we can compute the divergence component of the effective convergence field.
        This field produces a single plane convergence field
        which reproduces as much of the deflection field
        as possible for a single plane.

        See: https://arxiv.org/pdf/2006.07383.pdf
        see also the `effective_convergence_curl` method.
        """
        J = self.jacobian_effective_deflection_angle(x, y, z_s, params, **kwargs)
        return 0.5 * (J[..., 0, 0] + J[..., 1, 1])

    @unpack
    def effective_convergence_curl(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Use the curl of the effective reduced deflection angle vector field
        to compute an effective convergence which derives specifically
        from the curl of the deflection field.
        This field is purely a result of multiplane lensing
        and cannot occur in single plane lensing.

        See: https://arxiv.org/pdf/2006.07383.pdf
        """
        J = self.jacobian_effective_deflection_angle(x, y, z_s, params, **kwargs)
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
    z_l: (Optional[Tensor], optional)
        Redshift of the lens. Defaults to None.

    """

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZLType = None,
        name: NameType = None,
    ):
        super().__init__(cosmology=cosmology, name=name)
        self.add_param("z_l", z_l)

    @unpack
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Computes the reduced deflection angle of the lens at given coordinates [arcsec].

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        --------
        tuple[Tensor, Tensor]
            Reduced deflection angle in x and y directions.
        """
        d_s = self.cosmology.angular_diameter_distance(z_s, params)
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s, params)
        deflection_angle_x, deflection_angle_y = self.physical_deflection_angle(
            x, y, z_s, params
        )
        return (
            (d_ls / d_s) * deflection_angle_x,
            (d_ls / d_s) * deflection_angle_y,
        )

    @unpack
    def physical_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Computes the physical deflection angle immediately after passing through this lens's plane.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        -------
        tuple[Tensor, Tensor]
            Physical deflection angle in x and y directions in arcseconds.
        """
        d_s = self.cosmology.angular_diameter_distance(z_s, params)
        d_ls = self.cosmology.angular_diameter_distance_z1z2(z_l, z_s, params)
        deflection_angle_x, deflection_angle_y = self.reduced_deflection_angle(
            x, y, z_s, params
        )
        return (
            (d_s / d_ls) * deflection_angle_x,
            (d_s / d_ls) * deflection_angle_y,
        )

    @abstractmethod
    @unpack
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Computes the convergence of the lens at given coordinates.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        -------
        Tensor
            Convergence at the given coordinates.
        """
        ...

    @abstractmethod
    @unpack
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Computes the gravitational lensing potential at given coordinates.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        -------
        Tensor
            Gravitational lensing potential at the given coordinates in arcsec^2.
        """
        ...

    @unpack
    def surface_density(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Computes the surface mass density of the lens at given coordinates.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        -------
        Tensor
            Surface mass density at the given coordinates in solar masses per Mpc^2.
        """
        critical_surface_density = self.cosmology.critical_surface_density(
            z_l, z_s, params
        )
        return self.convergence(x, y, z_s, params) * critical_surface_density  # fmt: skip

    @unpack
    def raytrace(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform a ray-tracing operation by subtracting
        the deflection angles from the input coordinates.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.

        Returns
        -------
        tuple[Tensor, Tensor]
            Ray-traced coordinates in the x and y directions.
        """
        ax, ay = self.reduced_deflection_angle(x, y, z_s, params, **kwargs)
        return x - ax, y - ay

    @staticmethod
    def _arcsec2_to_time(z_l, z_s, cosmology, params):
        """
        This method is used by :func:`caustics.lenses.ThinLens.time_delay` to convert arcsec^2 to seconds in the context of gravitational time delays.
        """
        d_l = cosmology.angular_diameter_distance(z_l, params)
        d_s = cosmology.angular_diameter_distance(z_s, params)
        d_ls = cosmology.angular_diameter_distance_z1z2(z_l, z_s, params)
        return (1 + z_l) / c_Mpc_s * d_s * d_l / d_ls * arcsec_to_rad**2

    @unpack
    def time_delay(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        z_l: Optional[Tensor] = None,
        shapiro_time_delay: bool = True,
        geometric_time_delay: bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Computes the gravitational time delay for light passing through the lens at given coordinates.

        This time delay is induced by the photons travelling through a gravitational potential well (Shapiro time delay) plus the effect of the increased path length that the photons must traverse (geometric time delay).
        The main equation involved here is the following::

            \Delta t = \frac{1 + z_l}{c} \frac{D_s}{D_l D_{ls}} \left[ \frac{1}{2}|\vec{\alpha}(\vec{\theta})|^2 - \psi(\vec{\theta}) \right]

        where :math:`\vec{\alpha}(\vec{\theta})` is the deflection angle,
        :math:`\psi(\vec{\theta})` is the lensing potential,
        :math:`D_l` is the comoving distance to the lens,
        :math:`D_s` is the comoving distance to the source,
        and :math:`D_{ls}` is the comoving distance between the lens and the source. In the above equation, the first term is the geometric time delay and the second term is the gravitational time delay.

        Parameters
        ----------
        x: Tensor
            Tensor of x coordinates in the lens plane.
        y: Tensor
            Tensor of y coordinates in the lens plane.
        z_s: Tensor
            Tensor of source redshifts.
        z_l: Tensor
            Redshift of the lens.
        params: (Packed, optional)
            Dynamic parameter container for the lens model. Defaults to None.
        shapiro_time_delay: bool
            Whether to include the Shapiro time delay component.
        geometric_time_delay: bool
            Whether to include the geometric time delay component.

        Returns
        -------
        Tensor
            Time delay at the given coordinates.

        References
        ----------
        1. Irwin I. Shapiro (1964). "Fourth Test of General Relativity". Physical Review Letters. 13 (26): 789-791
        2. Refsdal, S. (1964). "On the possibility of determining Hubble's parameter and the masses of galaxies from the gravitational lens effect". Monthly Notices of the Royal Astronomical Society. 128 (4): 307-310.
        """
        TD = torch.zeros_like(x)

        if shapiro_time_delay:
            potential = self.potential(x, y, z_s, params)
            TD -= potential
        if geometric_time_delay:
            ax, ay = self.physical_deflection_angle(x, y, z_s, params)
            fp = 0.5 * (ax**2 + ay**2)
            TD += fp

        factor = self._arcsec2_to_time(z_l, z_s, self.cosmology, params)

        return factor * TD

    @unpack
    def _jacobian_deflection_angle_finitediff(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        pixelscale: Tensor,
        *args,
        params: Optional[Packed] = None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the deflection angle vector.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Compute deflection angles
        ax, ay = self.reduced_deflection_angle(x, y, z_s, params)

        # Build Jacobian
        J = torch.zeros((*ax.shape, 2, 2), device=ax.device, dtype=ax.dtype)
        J[..., 0, 1], J[..., 0, 0] = torch.gradient(ax, spacing=pixelscale)
        J[..., 1, 1], J[..., 1, 0] = torch.gradient(ay, spacing=pixelscale)
        return J

    @unpack
    def _jacobian_deflection_angle_autograd(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the deflection angle vector.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Ensure the x,y coordinates track gradients
        x = x.detach().requires_grad_()
        y = y.detach().requires_grad_()

        # Compute deflection angles
        ax, ay = self.reduced_deflection_angle(x, y, z_s, params)

        # Build Jacobian
        J = torch.zeros((*ax.shape, 2, 2), device=ax.device, dtype=ax.dtype)
        (J[..., 0, 0],) = torch.autograd.grad(
            ax, x, grad_outputs=torch.ones_like(ax), create_graph=True
        )
        (J[..., 0, 1],) = torch.autograd.grad(
            ax, y, grad_outputs=torch.ones_like(ax), create_graph=True
        )
        (J[..., 1, 0],) = torch.autograd.grad(
            ay, x, grad_outputs=torch.ones_like(ay), create_graph=True
        )
        (J[..., 1, 1],) = torch.autograd.grad(
            ay, y, grad_outputs=torch.ones_like(ay), create_graph=True
        )
        return J.detach()

    @unpack
    def jacobian_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        method="autograd",
        pixelscale=None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the deflection angle vector.
        This equates to a (2,2) matrix at each (x,y) point.

        method: autograd or fft
        """

        if method == "autograd":
            return self._jacobian_deflection_angle_autograd(x, y, z_s, params)
        elif method == "finitediff":
            if pixelscale is None:
                raise ValueError(
                    "Finite differences lensing jacobian requires regular grid "
                    "and known pixelscale. Please include the pixelscale argument"
                )
            return self._jacobian_deflection_angle_finitediff(
                x, y, z_s, pixelscale, params
            )
        else:
            raise ValueError("method should be one of: autograd, finitediff")

    @unpack
    def _jacobian_lens_equation_finitediff(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        pixelscale: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the lensing equation at specified points.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Build Jacobian
        J = self._jacobian_deflection_angle_finitediff(
            x, y, z_s, pixelscale, params, **kwargs
        )
        return torch.eye(2).to(J.device) - J

    @unpack
    def _jacobian_lens_equation_autograd(
        self,
        x: Tensor,
        y: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Return the jacobian of the lensing equation at specified points.
        This equates to a (2,2) matrix at each (x,y) point.
        """
        # Build Jacobian
        J = self._jacobian_deflection_angle_autograd(x, y, z_s, params, **kwargs)
        return torch.eye(2).to(J.device) - J.detach()
