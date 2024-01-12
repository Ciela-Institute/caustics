from datetime import datetime as dt
from collections import OrderedDict
from typing import Any, Dict

from torch import Tensor
from .._version import __version__
from ..namespace_dict import NamespaceDict, NestedNamespaceDict

from safetensors.torch import save

IMMUTABLE_ERR = TypeError("'StateDict' cannot be modified after creation.")
STATIC_PARAMS = "static"


class StateDict(OrderedDict):
    """A dictionary object that is immutable after creation.
    This is used to store the parameters of a simulator at a given
    point in time.

    Methods
    -------
    to_params()
        Convert the state dict to a dictionary of parameters.
    """

    __slots__ = ("_metadata", "_created")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._metadata = {}
        self._metadata["software_version"] = __version__
        self._metadata["created_time"] = dt.utcnow().isoformat()
        self._created = True

    def __delitem__(self, _) -> None:
        raise IMMUTABLE_ERR

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, "_created"):
            raise IMMUTABLE_ERR
        super().__setitem__(key, value)

    @classmethod
    def from_params(cls, params: "NestedNamespaceDict | NamespaceDict"):
        """Class method to create a StateDict
        from a dictionary of parameters

        Parameters
        ----------
        params : NamespaceDict
            A dictionary of parameters,
            can either be the full parameters
            that are "static" and "dynamic",
            or "static" only.

        Returns
        -------
        StateDict
            A state dictionary object
        """
        if isinstance(params, NestedNamespaceDict) and STATIC_PARAMS in params:
            params: NamespaceDict = params[STATIC_PARAMS].flatten()
        tensors_dict: Dict[str, Tensor] = {k: v.value for k, v in params.items()}
        return cls(tensors_dict)

    def to_params(self) -> NamespaceDict:
        """
        Convert the state dict to a dictionary of parameters.

        Returns
        -------
        NamespaceDict
            A dictionary of 'static' parameters.
        """
        from ..parameter import Parameter

        params = NamespaceDict()
        for k, v in self.items():
            params[k] = Parameter(v)
        return params

    def _to_safetensors(self) -> bytes:
        return save(self, metadata=self._metadata)
