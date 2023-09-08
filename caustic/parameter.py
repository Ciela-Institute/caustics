from typing import Optional

import torch
from torch import Tensor

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
        self, value: Optional[Tensor] = None, shape: Optional[tuple[int, ...]] = ()
    ):
        """
        Initializes an instance of the Parameter class.

        Args:
            value (Optional[Tensor], optional): The value of the parameter. Defaults to None.
            shape (Optional[tuple[int, ...]], optional): The shape of the parameter. Defaults to an empty tuple.

        Raises:
            ValueError: If both value and shape are None, or if shape is provided and doesn't match the shape of the value.
        """
        # Must assign one of value or shape
        self._value = value
        if value is None:
            if shape is None:
                raise ValueError("if value is None, a shape must be provided")
            self._shape = shape
        else:
            if shape is not None and shape != value.shape:
                raise ValueError(
                    f"value's shape {value.shape} does not match provided shape {shape}"
                )
            self._value = value
            self._shape = value.shape

    @property
    def static(self) -> bool:
        """
        Checks if the parameter is static.

        Returns:
            bool: True if the parameter is static, False otherwise.
        """
        return not self.dynamic

    @property
    def dynamic(self) -> bool:
        """
        Checks if the parameter is dynamic.

        Returns:
            bool: True if the parameter is dynamic, False otherwise.
        """
        return self._value is None

    @property
    def value(self) -> Optional[Tensor]:
        """
        Returns the value of the parameter.

        Returns:
            Optional[Tensor]: The value of the parameter, or None if the parameter is dynamic.
        """
        return self._value

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the parameter.

        Returns:
            tuple[int, ...]: The shape of the parameter.
        """
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
        if self._value is not None:
            self._value = self._value.to(device=device, dtype=dtype)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Parameter object.

        Returns:
            str: A string representation of the Parameter object.
        """
        if self.static:
            return f"Param(value={self.value})"
        else:
            return f"Param(shape={self.shape})"
