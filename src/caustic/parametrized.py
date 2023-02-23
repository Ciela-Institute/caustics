from collections import OrderedDict
from math import prod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from .param import Param

__all__ = ("Parametrized",)


class Parametrized:
    """
    Represents a class with Param and Parametrized attributes.

    TODO
        - Improve error checking by giving names of missing args.
        - Improve typing if possible
    """

    def __init__(self, name: str):
        self.name = name

        self._params: OrderedDict[str, Param] = OrderedDict()
        self._dynamic_size = 0
        self._n_dynamic = 0
        self._n_static = 0

        self._children: OrderedDict[str, Parametrized] = OrderedDict()

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        for param in self._params.values():
            param.to(device, dtype)

        for child in self._children.values():
            child.to(device, dtype)

    def add_param(
        self, name: str, value: Optional[Tensor] = None, shape: Tuple[int, ...] = ()
    ):
        """
        Stores parameter in _params and records its size.
        """
        self._params[name] = Param(value, shape)
        if value is None:
            size = prod(shape)
            self._dynamic_size += size
            self._n_dynamic += 1
        else:
            self._n_static += 1

    def __setattr__(self, key, val):
        """
        When val is Parametrized, stores it in _children and adds it as an attribute.
        """
        if isinstance(val, Param):
            raise ValueError(
                "cannot add Params directly as attributes: use add_param instead"
            )

        if isinstance(val, Parametrized):
            # Add children of val as children of self to keep track of all nested
            # Parametrizeds at top level
            self._children[val.name] = val
            for name, parametrized in val._children.items():
                self._children[name] = parametrized
        # elif isinstance(val, ParametrizedList):
        #     for parametrized in val:
        #         self._children[name]

        super().__setattr__(key, val)

    @property
    def n_dynamic(self) -> int:
        """
        Number of dynamic arguments in this object.
        """
        return self._n_dynamic

    @property
    def n_static(self) -> int:
        """
        Number of static arguments in this object.
        """
        return self._n_static

    @property
    def dynamic_size(self) -> int:
        """
        Total number of dynamic values in this object.
        """
        return self._dynamic_size

    def x_to_dict(
        self,
        x: Union[
            Dict[str, Union[List[Tensor], Tensor, Dict[str, Tensor]]],
            List[Tensor],
            Tensor,
        ],
    ) -> Dict[str, Any]:
        """
        Converts a list or tensor into a dict that can subsequently be unpacked
        into arguments to this object and its children.
        """
        if isinstance(x, Dict):
            # Assume we don't need to repack
            return x
        elif isinstance(x, list):
            n_passed = len(x)
            n_expected = (
                sum([child.n_dynamic for child in self._children.values()])
                + self.n_dynamic
            )
            if n_passed != n_expected:
                raise ValueError(
                    f"{n_passed} dynamic args were passed, but {n_expected} are "
                    "required"
                )

            cur_offset = self.n_dynamic
            x_repacked = {self.name: x[:cur_offset]}
            for name, child in self._children.items():
                x_repacked[name] = x[cur_offset : cur_offset + child.n_dynamic]
                cur_offset += child.n_dynamic

            return x_repacked
        elif isinstance(x, Tensor):
            n_passed = x.shape[-1]
            n_expected = (
                sum([child.dynamic_size for child in self._children.values()])
                + self.dynamic_size
            )
            if n_passed != n_expected:
                raise ValueError(
                    f"{n_passed} flattened dynamic args were passed, but {n_expected}"
                    " are required"
                )

            cur_offset = self.dynamic_size
            x_repacked = {self.name: x[..., :cur_offset]}
            for name, child in self._children.items():
                x_repacked[name] = x[..., cur_offset : cur_offset + child.dynamic_size]
                cur_offset += child.dynamic_size

            return x_repacked
        else:
            raise ValueError("can only repack a list or 1D tensor")

    def unpack(
        self, x: Dict[str, Union[List[Tensor], Tensor, Dict[str, Tensor]]]
    ) -> List[Tensor]:
        """
        Unpacks a dict of kwargs, list of args or flattened vector of args to retrieve
        this object's static and dynamic parameters.
        """
        my_x = x[self.name]
        if isinstance(my_x, Dict):
            # TODO: validate to avoid collisons with static params
            return [
                value if value is not None else my_x[name]
                for name, value in self._params.items()
            ]
        elif isinstance(my_x, List):
            vals = []
            i = 0
            for param in self._params.values():
                if not param.dynamic:
                    vals.append(param.value)
                else:
                    vals.append(my_x[i])
                    i += 1

            return vals
        elif isinstance(my_x, Tensor):
            vals = []
            i = 0
            for param in self._params.values():
                if not param.dynamic:
                    vals.append(param.value)
                else:
                    size = prod(param.shape)
                    vals.append(my_x[..., i : i + size].reshape(param.shape))
                    i += size

            return vals
        else:
            raise ValueError(
                f"invalid argument type: must be a dict containing key {self.name} "
                "and value containing args as list or flattened tensor, or kwargs"
            )

    def __getattribute__(self, key):
        """
        Enables accessing static params as attributes.
        """
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            try:
                param = self._params[key]
                if not param.dynamic:
                    return param
            except:
                raise e

    @property
    def params(self) -> Tuple[OrderedDict[str, Param], OrderedDict[str, Param]]:
        """
        Gets all of the static and dynamic Params for this object *and its children*.
        """
        static = OrderedDict()
        dynamic = OrderedDict()
        for name, param in self._params.items():
            if param.dynamic:
                dynamic[name] = param
            else:
                static[name] = param

        # Include childrens' parameters
        for name, child in self._children.items():
            child_static, child_dynamic = child.params
            static.update(child_static)
            dynamic.update(child_dynamic)

        return static, dynamic

    @property
    def static_params(self) -> OrderedDict[str, Param]:
        return self.params[0]

    @property
    def dynamic_params(self) -> OrderedDict[str, Param]:
        return self.params[1]

    def __str__(self) -> str:
        """
        Lists this object's name and the parameters for it and its children in
        the order they must be packed.
        """
        static, dynamic = self.params
        static = list(static.keys())
        dynamic = list(dynamic.keys())
        return (
            f"{self.__class__.__name__}(\n"
            f"    name='{self.name}',\n"
            f"    static params={static},\n"
            f"    dynamic params={dynamic}"
            "\n)"
        )


class ParametrizedList(list[Parametrized]):
    ...
