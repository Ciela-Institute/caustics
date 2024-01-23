from datetime import datetime as dt
from collections import OrderedDict
from typing import Any, Dict, Optional
from pathlib import Path
import os

from torch import Tensor
import torch
from .._version import __version__
from ..namespace_dict import NamespaceDict, NestedNamespaceDict
from .. import io

from safetensors.torch import save, load_file

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


class ImmutableODict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._created = True

    def __delitem__(self, _) -> None:
        raise IMMUTABLE_ERR

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, "_created"):
            raise IMMUTABLE_ERR
        super().__setitem__(key, value)

    def __setattr__(self, name, value) -> None:
        if hasattr(self, "_created"):
            raise IMMUTABLE_ERR
        return super().__setattr__(name, value)


class StateDict(ImmutableODict):
    """A dictionary object that is immutable after creation.
    This is used to store the parameters of a simulator at a given
    point in time.

    Methods
    -------
    to_params()
        Convert the state dict to a dictionary of parameters.
    """

    __slots__ = ("_metadata", "_created", "_created_time")

    def __init__(self, metadata=None, *args, **kwargs):
        # Get created time
        self._created_time = dt.utcnow()
        # Create metadata
        _meta = {
            "software_version": __version__,
            "created_time": self._created_time.isoformat(),
        }
        if metadata:
            _meta.update(metadata)

        # Set metadata
        self._metadata = ImmutableODict(_meta)

        # Now create the object, this will set _created
        # to True, and prevent any further modification
        super().__init__(*args, **kwargs)

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

        tensors_dict: Dict[str, Tensor] = {k: v.value for k, v in params.items()}
        return cls(**tensors_dict)

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

    def save(self, file_path: Optional[str] = None) -> str:
        """
        Saves the state dictionary to an optional
        ``file_path`` as safetensors format.
        If ``file_path`` is not given,
        this will default to a file in
        the current working directory.

        *Note: The path specified must
        have a '.st' extension.*

        Parameters
        ----------
        file_path : str, optional
            The file path to save the
            state dictionary to, by default None

        Returns
        -------
        str
            The final path of the saved file
        """
        if not file_path:
            file_path = Path(os.path.curdir) / self.__st_file
        elif isinstance(file_path, str):
            file_path = Path(file_path)

        ext = ".st"
        if file_path.suffix != ext:
            raise ValueError(f"File must have '{ext}' extension")

        return io.to_file(file_path, self._to_safetensors())

    @classmethod
    def load(cls, file_path: str) -> "StateDict":
        """
        Loads the state dictionary from a
        specified ``file_path``.

        Parameters
        ----------
        file_path : str
            The file path to load the
            state dictionary from.

        Returns
        -------
        StateDict
            The loaded state dictionary
        """
        # TODO: Need to rethink this for remote paths

        # Load just the metadata
        metadata = io.get_safetensors_metadata(file_path)

        # Load the full data to cpu first
        st_dict = load_file(file_path)
        st_dict = {k: v if v.nelement() > 0 else None for k, v in st_dict.items()}
        return cls(metadata=metadata, **st_dict)

    @property
    def __st_file(self) -> str:
        file_format = "%Y%m%dT%H%M%S_caustics.st"
        return self._created_time.strftime(file_format)

    def _to_safetensors(self) -> bytes:
        return save(_sanitize(self), metadata=self._metadata)
