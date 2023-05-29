from abc import abstractmethod
from typing import Any, Optional

from torch import Tensor

from ..parametrized import Parametrized

__all__ = ("Source",)


class Source(Parametrized):
    """
    This is an abstract base class used to represent a source in a strong gravitational lensing system. 
    It provides the basic structure and required methods that any derived source class should implement. 
    The Source class inherits from the Parametrized class, implying that it contains parameters that can 
    be optimized or manipulated.
    
    The class introduces one abstract method, `brightness`, that must be implemented in any concrete 
    subclass. This method calculates the brightness of the source at given coordinates.
    """
    @abstractmethod
    def brightness(
        self, thx: Tensor, thy: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        """
        Abstract method that calculates the brightness of the source at the given coordinates. 
        This method is expected to be implemented in any class that derives from Source.
        
        Args:
            thx (Tensor): The x-coordinate(s) at which to calculate the source brightness. 
                This could be a single value or a tensor of values.
                
            thy (Tensor): The y-coordinate(s) at which to calculate the source brightness. 
                This could be a single value or a tensor of values.
                
            x (Optional[dict[str, Any]]): Additional parameters that might be required to calculate 
                the brightness. The exact contents will depend on the specific implementation in derived classes. 

        Returns:
            Tensor: The brightness of the source at the given coordinate(s). The exact form of the output 
            will depend on the specific implementation in the derived class.
            
        Note: 
            This method must be overridden in any class that inherits from `Source`.
        """
        ...
