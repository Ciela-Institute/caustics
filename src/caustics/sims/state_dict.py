from datetime import datetime as dt
from collections import OrderedDict
from typing import Any, Dict, Optional
from pathlib import Path

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


def _merge_and_flatten(params: "NamespaceDict | NestedNamespaceDict") -> NamespaceDict:
    """
    Extract the parameters from a nested dictionary
    of parameters and merge them into a single
    dictionary of parameters.

    Parameters
    ----------
    params : NamespaceDict | NestedNamespaceDict
        The nested dictionary of parameters
        that includes both "static" and "dynamic".

    Returns
    -------
    NamespaceDict
        The merged dictionary of parameters.

    Raises
    ------
    TypeError
        If the input ``params`` is not a
        ``NamespaceDict`` or ``NestedNamespaceDict``.
    ValueError
        If the input ``params`` is a ``NestedNamespaceDict``
        but does not have the keys ``"static"`` and ``"dynamic"``.
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

        # Merge the "static" and "dynamic" dictionaries
        # to a single merged dictionary
        final_dict = NestedNamespaceDict()
        for pdict in param_dicts:
            for k, v in pdict.items():
                if k not in final_dict:
                    final_dict[k] = v
                else:
                    final_dict[k] = {**final_dict[k], **v}

        # Flatten the dictionary to a single level
        params = final_dict.flatten()
    return params


def _get_param_values(flat_params: "NamespaceDict") -> Dict[str, Optional[Tensor]]:
    """
    Get the values of the parameters from a
    flattened dictionary of parameters.

    Parameters
    ----------
    flat_params : NamespaceDict
        A flattened dictionary of parameters.

    Returns
    -------
    Dict[str, Optional[Tensor]]
        A dictionary of parameter values,
        these values can be a tensor or None.
    """
    return {k: v.value for k, v in flat_params.items()}


def _extract_tensors_dict(
    params: "NamespaceDict | NestedNamespaceDict",
) -> Dict[str, Optional[Tensor]]:
    """
    Extract the tensors from a nested dictionary
    of parameters and merge them into a single
    dictionary of parameters. Then return a
    dictionary of tensors by getting the parameter
    tensor values.

    Parameters
    ----------
    params : NestedNamespaceDict
        The nested dictionary of parameters
        that includes both "static" and "dynamic"
    export_params : bool, optional
        Whether to return the merged parameters as well,
        not just the dictionary of tensors,
        by default False.

    Returns
    -------
    dict
        A dictionary of tensors
    """
    all_params = _merge_and_flatten(params)
    return _get_param_values(all_params)


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
        tensors_dict = _extract_tensors_dict(params)
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

    def save(self, file_path: "str | Path | None" = None) -> str:
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
        input_path: Path

        if not file_path:
            input_path = Path.cwd() / self.__st_file
        elif isinstance(file_path, str):
            input_path = Path(file_path)
        else:
            input_path = file_path

        ext = ".st"
        if input_path.suffix != ext:
            raise ValueError(f"File must have '{ext}' extension")

        return io.to_file(input_path, self._to_safetensors())

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
