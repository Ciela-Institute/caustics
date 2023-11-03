from typing import Optional, Union

import torch
from torch import Tensor
from .namespace_dict import NamespaceDict

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
        self, value: Optional[Union[Tensor, float]] = None, shape: Optional[tuple[int, ...]] = ()
    ):
        # Must assign one of value or shape
        if value is None:
            if shape is None:
                raise ValueError("If value is None, a shape must be provided")
            if not isinstance(shape, tuple):
                raise TypeError("The shape of a parameter must be a tuple")
            self._shape = shape
        else:
            value = torch.as_tensor(value)
            self._shape = value.shape
        self._value = value
        self._dtype = None if value is None else value.dtype

    @property
    def static(self) -> bool:
        return not self.dynamic

    @property
    def dynamic(self) -> bool:
        return self._value is None

    @property
    def value(self) -> Optional[Tensor]:
        return self._value
    
    @value.setter
    def value(self, value: Union[None, Tensor, float]):
        if value is not None:
            value = torch.as_tensor(value)
            if value.shape != self.shape:
                raise ValueError(f"Cannot set Parameter value with a different shape. Received {value.shape}, expected {self.shape}")
        self._value = value
        self._dtype = None if value is None else value.dtype
    
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
        if self.static:
            self.value = self._value.to(device=device, dtype=dtype)
        return self

    def __repr__(self) -> str:
        if self.static:
            return f"Param(value={self.value}, dtype={str(self.dtype)})"
        else:
            return f"Param(shape={self.shape})"
