from __future__ import annotations  # Enables postponed evaluation for type hint of Parametrized
from typing import Optional, Union, Callable

import torch
from torch import Tensor
from caustic.namespace_dict import NamespaceDict

__all__ = ("Parameter",)


class Parameter:
    """
    Represents a static or dynamic parameter used for strong gravitational lensing simulations in the caustic codebase.

    A static parameter has a fixed value, while a dynamic parameter must be passed in each time it's required.

    Attributes:
        value (Optional[Tensor]): The value of the parameter.
        shape (tuple[int, ...]): The shape of the parameter.
    """

    def __init__(
        self, 
        name: str,
        parent: "Parametrized",
        value: Optional[Union[Tensor, float, Callable, "Parameter"]] = None, 
        shape: Optional[tuple[int, ...]] = ()
    ):
        self.name = name
        self.parent = parent
        self.symbolic_link = None
        if value is None: # Dynamic parameter
            if shape is None:
                raise ValueError("If value is None, a shape must be provided")
            if not isinstance(shape, tuple):
                raise TypeError("The shape of a parameter must be a tuple")
        elif isinstance(value, (Callable)): # Symbolic parameter
            if shape is None:
                raise ValueError("If value is a function, a shape must be provided")
        elif isinstance(value, (Tensor, float)): # Static parameter
            if shape is not None and shape != value.shape:
                raise ValueError(
                    f"value's shape {value.shape} does not match provided shape {shape}"
                )
        self._value_setter(value, shape, torch.float32)
   
    def _value_setter(self, value, shape, dtype):
        if value is None: # Dynamic parameter
            self._value = self._none_value
            self._static_value = None
            self._symbolic_value = self._none_value
            self.symbolic_link = None
            self._shape = shape
            self._dtype = None
        elif isinstance(value, Callable): # Symbolic parameter with value infered from a function (e.g. a Simulator)
            self._value = self._symbolic_value
            self._static_value = None
            self._symbolic_value = value
            fn_name = getattr(value, name, None) # symbolic_link use
            self.symbolic_link = fn_name if fn_name is not None else str(value)
            self._shape = shape
            self._dtype = None
        elif isinstance(value, Parameter): # Symbolic parameter linked directly to another parameter
            self._value = self._symbolic_value
            self._static_value = None
            self._symbolic_value = self._symbolic_parameter_value
            self.symbolic_link = value
            self._shape = shape
            self._dtype = None
        elif isinstance(value, (Tensor, float)): # Static parameter
            value = torch.as_tensor(value).to(dtype)
            self._value = self._static_value
            self._static_value = value
            self._symbolic_value = self._none_value
            self.symbolic_link = None
            self._shape = value.shape
            self._dtype = dtype
    
    @staticmethod
    def _none_value(x: "Packed" = None):
        return None
    
    def _static_value(self, x: "Packed" = None):
        return self._static_value

    def _symbolic_value(self, x: "Packed" = None):
        if x is None:
            return self._symbolic_value
        else:
            return self._symbolic_value(x)
    
    def _symbolic_parameter_value(self, x: "Packed" = None):
        if x is None:
            breakpoint()
            return self.symbolic_link 
        else:
            ln = self.symbolic_link
            return x[ln.name][ln.name]

    @property
    def static(self) -> bool:
        return not self.dynamic
    
    @property
    def symbolic(self) -> bool:
        return self.symbolic_link is not None

    @property
    def dynamic(self) -> bool:
        return self.value() is None
    
    @property
    def value(self):
        return self._value # _value is a function to be called
    
    @value.setter
    def value(self, value: Union[type(None), Tensor, float, Callable]):
        if isinstance(value, Tensor):
            dtype = value.dtype
            if value.shape != self.shape:
                raise ValueError(f"Cannot set Parameter value with a different shape. Received {value.shape}, expected {self.shape}")
        else:
            dtype = torch.float32
        self._value_setter(value, self.shape, dtype)
        
    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
    
    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """
        Moves and/or casts the values of the parameter.

        Args:
            device (Optional[torch.device], optional): The device to move the values to. Defaults to None.
            dtype (Optional[torch.dtype], optional): The desired data type. Defaults to None.
        """
        if self.static:
            self.value = self._value.to(device=device, dtype=dtype)

    def __repr__(self) -> str:
        if self.static:
            dtype = str(self.dtype)[-1] if self.dtype is not None else "None"
            return f"Param(value={self.value()}, dtype={dtype})"
        else:
            return f"Param(shape={self.shape})"
