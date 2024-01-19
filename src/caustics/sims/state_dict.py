from datetime import datetime as dt
from collections import OrderedDict
from typing import Any, Dict, Optional

from torch import Tensor
import torch
from .._version import __version__
from ..namespace_dict import NamespaceDict, NestedNamespaceDict

from safetensors.torch import save

IMMUTABLE_ERR = TypeError("'StateDict' cannot be modified after creation.")
PARAM_KEYS = ["dynamic", "static"]


def _sanitize(tensors_dict: Dict[str, Optional[Tensor]]) -> Dict[str, Tensor]:
    """
    Sanitize the input dictionary of tensors by
    replacing Nones with tensors of size 0.

    Parameters
    ----------
    tensors_dict : dict
        A dictionary of tensors, including None.

    Returns
    -------
    dict
        A dictionary of tensors, with empty tensors
        replaced by tensors of size 0.
    """
    return {
        k: v if isinstance(v, Tensor) else torch.ones(0)
        for k, v in tensors_dict.items()
    }


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

    def __repr__(self) -> str:
        state_dict_list = [
            (k, v) if v.nelement() > 0 else (k, None) for k, v in self.items()
        ]
        class_name = self.__class__.__name__
        if not state_dict_list:
            return "%s()" % (class_name,)
        return "%s(%r)" % (class_name, state_dict_list)

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
        if not isinstance(params, (NamespaceDict, NestedNamespaceDict)):
            raise TypeError("params must be a NamespaceDict or NestedNamespaceDict")

        if isinstance(params, NestedNamespaceDict):
            # In this case, params is the full parameters
            # with both "static" and "dynamic" keys
            if sorted(params.keys()) != PARAM_KEYS:
                raise ValueError(f"params must have keys {PARAM_KEYS}")

            # Extract the "static" and "dynamic" parameters
            param_dicts = list(params.values())

            # Extract the "static" and "dynamic" parameters
            # to a single merged dictionary
            final_dict = NestedNamespaceDict()
            for pdict in param_dicts:
                for k, v in pdict.items():
                    if k not in final_dict:
                        final_dict[k] = v
                    else:
                        final_dict[k] = {**final_dict[k], **v}

            # Flatten the dictionary to a single level
            params: NamespaceDict = final_dict.flatten()

        tensors_dict: Dict[str, Tensor] = _sanitize(
            {k: v.value for k, v in params.items()}
        )
        return cls(tensors_dict)

    def to_params(self) -> NestedNamespaceDict:
        """
        Convert the state dict to
        a nested dictionary of parameters.

        Returns
        -------
        NestedNamespaceDict
            A nested dictionary of parameters.
        """
        from ..parameter import Parameter

        params = NamespaceDict()
        for k, v in self.items():
            if v.nelement() == 0:
                # Set to None if the tensor is empty
                v = None
            params[k] = Parameter(v)
        return NestedNamespaceDict(params)

    def _to_safetensors(self) -> bytes:
        return save(self, metadata=self._metadata)
