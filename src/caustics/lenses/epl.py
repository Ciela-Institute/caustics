# mypy: disable-error-code="operator,dict-item"
from typing import Optional, Union, Annotated

from torch import Tensor, pi
from caskade import forward, Param

from .base import ThinLens, CosmologyType, NameType, ZType
from . import func
from ..angle_mixin import Angle_Mixin

__all__ = ("EPL",)


class EPL(Angle_Mixin, ThinLens):
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
    chunk_size: int
        Number of iterations to do in parallel for the iterative solver.
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

    Rein: Optional[Union[Tensor, float]]
        The Einstein radius of the lens, exact at q=1.0.

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
        "Rein": 1.0,
        "t": 1.0,
    }

    def __init__(
        self,
        cosmology: CosmologyType,
        z_l: ZType = None,
        z_s: ZType = None,
        x0: Annotated[
            Optional[Union[Tensor, float]], "X coordinate of the lens center", True
        ] = None,
        y0: Annotated[
            Optional[Union[Tensor, float]], "Y coordinate of the lens center", True
        ] = None,
        q: Annotated[
            Optional[Union[Tensor, float]], "Axis ratio of the lens", True
        ] = None,
        phi: Annotated[
            Optional[Union[Tensor, float]], "Position angle of the lens", True
        ] = None,
        Rein: Annotated[
            Optional[Union[Tensor, float]], "Einstein radius of the lens", True
        ] = None,
        t: Annotated[
            Optional[Union[Tensor, float]],
            "Power law slope (`gamma-1`) of the lens",
            True,
        ] = None,
        angle_system: str = "q_phi",
        e1: Optional[Union[Tensor, float]] = None,
        e2: Optional[Union[Tensor, float]] = None,
        c1: Optional[Union[Tensor, float]] = None,
        c2: Optional[Union[Tensor, float]] = None,
        s: Annotated[
            float, "Softening length for the elliptical power-law profile"
        ] = 0.0,
        n_iter: Annotated[int, "Number of iterations for the iterative solver"] = 18,
        chunk_size: Annotated[
            Optional[int], "Number of chunks for the iterative solver"
        ] = None,
        name: NameType = None,
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

        Rein: Optional[Tensor]
            Einstein radius of the lens.

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

        chunk_size: Optional[int]
            Number of iterations to do in parallel for the iterative solver.
            If not provided, it is set to n_iter which is fastest, but uses more memory.
        """
        super().__init__(cosmology, z_l, name=name, z_s=z_s)

        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.q = Param("q", q, units="unitless", valid=(0, 1))
        self.phi = Param("phi", phi, units="radians", valid=(0, pi), cyclic=True)
        self.Rein = Param("Rein", Rein, units="arcsec", valid=(0, None))
        self.t = Param("t", t, units="unitless", valid=(0, 2))
        self.angle_system = angle_system
        if self.angle_system == "e1_e2":
            self.e1 = e1
            self.e2 = e2
        elif self.angle_system == "c1_c2":
            self.c1 = c1
            self.c2 = c2
        self.s = s

        self.n_iter = n_iter
        self.chunk_size = chunk_size

    @forward
    def reduced_deflection_angle(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        q: Annotated[Tensor, "Param"],
        phi: Annotated[Tensor, "Param"],
        Rein: Annotated[Tensor, "Param"],
        t: Annotated[Tensor, "Param"],
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

        Returns
        --------
        x_component: Tensor
            Deflection Angle

            *Unit: arcsec*

        y_component: Tensor
            Deflection Angle

            *Unit: arcsec*

        """
        return func.reduced_deflection_angle_epl(
            x0, y0, q, phi, Rein, t, x, y, self.n_iter, self.chunk_size
        )

    @forward
    def potential(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        q: Annotated[Tensor, "Param"],
        phi: Annotated[Tensor, "Param"],
        Rein: Annotated[Tensor, "Param"],
        t: Annotated[Tensor, "Param"],
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

        Returns
        -------
        Tensor
            The lensing potential.

            *Unit: arcsec^2*

        """
        return func.potential_epl(
            x0, y0, q, phi, Rein, t, x, y, self.n_iter, self.chunk_size
        )

    @forward
    def convergence(
        self,
        x: Tensor,
        y: Tensor,
        x0: Annotated[Tensor, "Param"],
        y0: Annotated[Tensor, "Param"],
        q: Annotated[Tensor, "Param"],
        phi: Annotated[Tensor, "Param"],
        Rein: Annotated[Tensor, "Param"],
        t: Annotated[Tensor, "Param"],
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

        Returns
        -------
        Tensor
            The convergence of the lens.

            *Unit: unitless*

        """
        return func.convergence_epl(x0, y0, q, phi, Rein, t, x, y, self.s)
