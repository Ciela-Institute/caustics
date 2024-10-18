from abc import abstractmethod
from typing import Optional, Annotated

from torch import Tensor
from caskade import Module, forward

__all__ = ("Source",)

NameType = Annotated[Optional[str], "Name of the source"]


class Source(Module):
    """
    This is an abstract base class used to represent a source
    in a strong gravitational lensing system.
    It provides the basic structure and required methods
    that any derived source class should implement.
    The Source class inherits from the Parametrized class,
    implying that it contains parameters that can
    be optimized or manipulated.

    The class introduces one abstract method, `brightness`,
    that must be implemented in any concrete subclass.
    This method calculates the brightness of
    the source at given coordinates.
    """

    @abstractmethod
    @forward
    def brightness(self, x: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Abstract method that calculates the brightness of the source at the given coordinates.
        This method is expected to be implemented in any class that derives from Source.

        Parameters
        ----------
        x: Tensor
            The x-coordinate(s) at which to calculate
            the source brightness.
            This could be a single value or a tensor of values.

            *Unit: arcsec*

        y: Tensor
            The y-coordinate(s) at which to calculate
            the source brightness.
            This could be a single value or a tensor of values.

            *Unit: arcsec*

        Returns
        -------
        Tensor
            The brightness of the source at the given coordinate(s).
            The exact form of the output will depend on
            the specific implementation in the derived class.

        Notes
        -----
        This method must be overridden in any class
        that inherits from `Source`.
        """
        ...
