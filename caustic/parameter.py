from __future__ import annotations
from typing import Optional, Union, Callable

import torch
from torch import Tensor
import numpy as np

__all__ = ("Parameter",)


class Parameter:
    """
    Represents a static, dynamic or symbolic parameter for the Parametrized class.

    A static parameter has a fixed value, while a dynamic parameter must be provided each time 
    in the forward call of a simulator (or parametrized module).
    
    A symbolic parameter has a value infered dynmacially from other parameters during a forward pass of the
    simulator. It is akin to a symbolic link in linux file systems.

    Attributes:
        value (Optional[Tensor]): The value of the parameter.
        shape (tuple[int, ...]): The shape of the parameter.
    """

    def __init__(
        self, 
        name: str,
        parent: "Parametrized",
        value: Optional[Union[Tensor, int, float, np.ndarray, Callable, Parameter]] = None, 
        shape: Optional[tuple[int, ...]] = ()
    ):
        if value is None: # Dynamic parameter
            if shape is None:
                raise ValueError("If value is None, a shape must be provided")
            if not isinstance(shape, tuple):
                raise TypeError("The shape of a parameter must be a tuple")
        elif isinstance(value, (Callable)): # Symbolic parameter
            if shape is None:
                raise ValueError("If value is a function, a shape must be provided")
        if isinstance(value, (int, float, np.ndarray)):
            value = torch.as_tensor(value)
        if isinstance(value, (Tensor, Parameter)):
            shape = value.shape
        self.name = name
        self.parent = parent
        self._shape = shape # only time this is defined and should be immutable
        self.value = value
   
    @staticmethod
    def _none_value(x: "Packed" = None):
        return None
    
    def _static_value(self, x: "Packed" = None) -> Tensor:
        return self.__static_value

    def _make_soft_link(self, parameter: Parameter):
        """
        Function used in value setter to create a symbolic link 
        from the given parameter. In linux systems, this would be a soft link
        pointing to another parameter in the DAG.
        """
        def soft_link(x: "Packed" = None):
            if x is None:
                return parameter
            else:
                breakpoint()
                return x[parameter.parent.name][parameter.name]
        return soft_link 

    def symbolic_link(self, x: "Packed" = None):
        return self._symbolic_link(x)

    @property
    def static(self) -> bool:
        return not self.dynamic
    
    @property
    def symbolic(self) -> bool:
        return (self.symbolic_link() is not None) and (not self.dynamic)

    @property
    def dynamic(self) -> bool:
        return self.value() is None
    
    @property
    def value(self) -> Callable:
        """
        value returns a function to be called either with no arguments or 
        with the Packed parameters of the simulator. 
        The function type generalize the usecase of this property to 
        all 3 types of parameters (static, dynamic and symbolic).
        """
        return self._value
    
    @value.setter
    def value(self, value: Union[type(None), Tensor, int, float, np.ndarray, Callable, Parameter]):
        if isinstance(value, Tensor):
            if value.shape != self.shape:
                raise ValueError(f"Cannot set Parameter value with a different shape. Received {value.shape}, expected {self.shape}")
        # Now update the internal state of the parameter to reflect the type of value
        if value is None: # Dynamic parameter
            self.__static_value = None
            self._symbolic_link = self._none_value
            self._value = self._none_value
            self._dtype = None
        elif isinstance(value, Callable): # Symbolic parameter with value infered from a function (e.g. a Simulator)
            self.__static_value = None
            self._symbolic_link = value
            self._value = self._symbolic_link
            self._dtype = None
        elif isinstance(value, Parameter): # Symbolic parameter linked directly to another parameter
            self.__static_value = None
            self._symbolic_link = self._make_soft_link(value)
            self._value = self._symbolic_link
            self._dtype = None
        elif isinstance(value, (Tensor, int, float, np.ndarray)): # Static parameter
            value = torch.as_tensor(value)
            self.__static_value = value
            self._symbolic_link = self._none_value
            self._value = self._static_value
            self._dtype = value.dtype
        
    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
    
    def set_static(self):
        self.value = None

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Moves and/or casts the values of the parameter.

        Args:
            device (Optional[torch.device], optional): The device to move the values to. Defaults to None.
            dtype (Optional[torch.dtype], optional): The desired data type. Defaults to None.
        """
        if self.static and not self.symbolic:
            self.value = self.__static_value.to(device=device, dtype=dtype)
        return self

    def __repr__(self) -> str:
        if self.dynamic:
            return f"{self.name} (dynamic, shape={self.shape})"
        if self.symbolic:
            return f"{self.name} (symbolic link={self.symbolic_link()}, shape={self.shape})"
        if self.static:
            dtype = str(self.dtype).split(".")[-1]
            return f"{self.name} (static value={self.value()}, dtype={dtype})"
    
    def __str__(self) -> str:
        return self.name
