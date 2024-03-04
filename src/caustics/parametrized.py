# mypy: disable-error-code="var-annotated,index,type-arg"
from collections import OrderedDict
from math import prod
from typing import Optional, Union
from dataclasses import dataclass
import functools
import inspect

import torch
import re
import keyword
from torch import Tensor
import graphviz

from .packed import Packed
from .namespace_dict import NamespaceDict, NestedNamespaceDict
from .parameter import Parameter

__all__ = ("Parametrized", "unpack")


@dataclass
class ClassParam:
    description: str
    isParam: bool = False
    unit: Optional[str] = None


def check_valid_name(name):
    if keyword.iskeyword(name) or not bool(re.match("^[a-zA-Z_][a-zA-Z0-9_]*$", name)):
        raise NameError(
            f"The string {name} contains illegal characters (like space or '-'). "
            "Please use snake case or another valid python variable naming style."
        )


class Parametrized:
    """
    Represents a class with Param and Parametrized attributes,
    typically used to construct parts of a simulator
    that have parameters which need to be tracked during MCMC sampling.

    This class can contain Params, Parametrized,
    tensor buffers or normal attributes as its attributes.
    It provides functionalities to manage these attributes,
    ensuring that an attribute of one type isn't rebound
    to be of a different type.

    TODO
    - Attributes can be Params, Parametrized, tensor buffers or just normal attributes.
    - Need to make sure an attribute of one of those types isn't rebound to be of a different type.

    Attributes
    ----------
    name: str
        The name of the Parametrized object. Default to class name.
    parents: NestedNamespaceDict
        Nested dictionary of parent Parametrized objects (higher level, more abstract modules).
    params: OrderedDict[str, Parameter]
        Dictionary of parameters.
    childs: NestedNamespaceDict
        Nested dictionary of childs Parametrized objects (lower level, more specialized modules).
    dynamic_size: int
        Size of dynamic parameters.
    n_dynamic: int
        Number of dynamic parameters.
    n_static: int
        Number of static parameters.
    """

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = self._default_name()
        check_valid_name(name)
        if not isinstance(name, str):
            raise ValueError(f"name must be a string (received {name})")
        self._name = name
        self._parents: OrderedDict[str, Parametrized] = NamespaceDict()
        self._params: OrderedDict[str, Parameter] = NamespaceDict()
        self._childs: OrderedDict[str, Parametrized] = NamespaceDict()
        self._module_key_map = {}

    def _default_name(self):
        return re.search("([A-Z])\w+", str(self.__class__)).group()

    def __getattribute__(self, key):
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            # Check if key refers to a parametrized module name (different from its attribute key)
            _map = super().__getattribute__(
                "_module_key_map"
            )  # use super to avoid recursion error
            if key in _map.keys():
                return super().__getattribute__(_map[key])
            else:
                raise e

    def __setattr__(self, key, value):
        try:
            if key in self._params.keys():
                # Redefine parameter value instead of making a new attribute
                self._params[key].value = value
            elif isinstance(value, Parameter):
                # Create new parameter and attach it as an attribute
                self.add_param(key, value.value, value.shape)
            elif isinstance(value, Parametrized):
                # Update map from attribute key to module name
                # for __getattribute__ method
                self._module_key_map[value.name] = key
                self.add_parametrized(value, set_attr=False)
                # set attr only to user defined key,
                # not module name (self.{module.name} is still accessible,
                # see __getattribute__ method)
                super().__setattr__(key, value)
            else:
                super().__setattr__(key, value)
        except AttributeError:  # _params or another attribute in here do not exist yet
            super().__setattr__(key, value)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        check_valid_name(new_name)
        old_name = self.name
        for parent in self._parents.values():
            del parent._childs[old_name]
            parent._childs[new_name] = self
        for child in self._childs.values():
            del child._parents[old_name]
            child._parents[new_name] = self
        self._name = new_name

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """
        Moves static Params for this component and its childs
        to the specified device and casts them to the specified data type.
        """
        for name, p in self._params.items():
            self._params[name] = p.to(device, dtype)
        for child in self._childs.values():
            child.to(device, dtype)
        return self

    @staticmethod
    def _generate_unique_name(name, module_names):
        i = 1
        while f"{name}_{i}" in module_names:
            i += 1
        return f"{name}_{i}"

    def add_parametrized(self, p: "Parametrized", set_attr=True):
        """
        Add a child to this module, and create edges for the DAG
        """
        # If self.name is already in the module parents, we need to update self.name
        if self.name in p._parents.keys():
            new_name = self._generate_unique_name(self.name, p._parents.keys())
            self.name = new_name  # from name.setter, this updates the DAG edges as well
        p._parents[self.name] = self
        # If the module name is already in self._childs, we need to update module name
        if p.name is self._childs.keys():
            new_name = self._generate_unique_name(p.name, self._childs.keys())
            p.name = new_name
        self._childs[p.name] = p
        if set_attr:
            super().__setattr__(p.name, p)

    def add_param(
        self,
        name: str,
        value: Optional[Union[Tensor, float]] = None,
        shape: Optional[tuple[int, ...]] = (),
    ):
        """
        Stores a parameter in the _params dictionary and records its size.

        Parameters
        ----------
        name: str
            The name of the parameter.
        value: (Optional[Tensor], optional)
            The value of the parameter. Defaults to None.
        shape: (Optional[tuple[int, ...]], optional)
            The shape of the parameter. Defaults to an empty tuple.
        """
        self._params[name] = Parameter(value, shape)
        # __setattr__ inside add_param to catch all uses of this method
        super().__setattr__(name, self._params[name])

    @property
    def n_dynamic(self) -> int:
        return len(self.module_params.dynamic)

    @property
    def n_static(self) -> int:
        return len(self.module_params.static)

    @property
    def dynamic_size(self) -> int:
        return sum(prod(dyn.shape) for dyn in self.module_params.dynamic.values())

    def pack(
        self,
        x: Union[
            list[Tensor],
            dict[str, Union[list[Tensor], Tensor, dict[str, Tensor]]],
            Tensor,
        ] = Packed(),
    ) -> Packed:
        """
        Converts a list or tensor into a dict that can subsequently be unpacked
        into arguments to this component and its childs.
        Also, add a batch dimension to each Tensor
        without such a dimension.

        Parameters
        ----------
        x : list of tensor, dict of tensor, or tensor
            The input to be packed.

        Returns
        -------
        Packed
            The packed input, and whether or not the input was batched.

        Raises
        ------
        ValueError
            If the input is not a list, dictionary, or tensor.
        ValueError
            If the input is a dictionary and some keys are missing.
        ValueError
            If the number of dynamic arguments does not match the expected number.
        ValueError
            If the input is a tensor and the shape does not match the expected shape.
        """
        if isinstance(x, (dict, Packed)):
            missing_names = [
                name for name in self.params.dynamic.keys() if name not in x
            ]
            if len(missing_names) > 0:
                raise ValueError(f"missing x keys for {missing_names}")

            # TODO: check structure!
            return Packed(x)
        elif isinstance(x, (list, tuple)):
            n_passed = len(x)
            n_dynamic_params = len(self.params.dynamic.flatten())
            n_dynamic_modules = len(self.dynamic_modules)
            x_repacked = {}
            if n_passed == n_dynamic_params:
                cur_offset = 0
                for name, module in self.dynamic_modules.items():
                    x_repacked[name] = x[cur_offset : cur_offset + module.n_dynamic]
                    cur_offset += module.n_dynamic
            elif n_passed == n_dynamic_modules:
                for i, name in enumerate(self.dynamic_modules.keys()):
                    x_repacked[name] = x[i]
            else:
                raise ValueError(
                    f"{n_passed} dynamic args were passed, but {n_dynamic_params} parameters or "
                    f"{n_dynamic_modules} Tensor (1 per dynamic module) are required"
                )
            return Packed(x_repacked)

        elif isinstance(x, Tensor):
            n_passed = x.shape[-1]
            n_expected = sum(
                [module.dynamic_size for module in self.dynamic_modules.values()]
            )
            if n_passed != n_expected:
                # TODO: give component and arg names
                raise ValueError(
                    f"{n_passed} flattened dynamic args were passed, but {n_expected} "
                    f"are required"
                )

            cur_offset = 0
            x_repacked = {}
            for name, module in self.dynamic_modules.items():
                x_repacked[name] = x[..., cur_offset : cur_offset + module.dynamic_size]
                cur_offset += module.dynamic_size
            return Packed(x_repacked)

        else:
            raise ValueError("Data structure not supported")

    def unpack(
        self, x: Optional[dict[str, Union[list[Tensor], dict[str, Tensor], Tensor]]]
    ) -> list[Tensor]:
        """
        Unpacks a dict of kwargs, list of args or flattened vector of args to retrieve
        this object's static and dynamic parameters.

        Parameters
        ----------
        x:  (Optional[dict[str, Union[list[Tensor], dict[str, Tensor], Tensor]]])
            The packed object to be unpacked.

        Returns
        -------
        list[Tensor]
            Unpacked static and dynamic parameters of the object. Note that
            parameters will have an added batch dimension from the pack method.

        Raises
        ------
        ValueError
            If the input is not a dict, list, tuple or tensor.
        ValueError
            If the argument type is invalid. It must be a dict containing key {self.name}
            and value containing args as list or flattened tensor, or kwargs.
        """
        # Check if module has dynamic parameters
        if self.module_params.dynamic:
            dynamic_x = x[self.name]
        else:  # all parameters are static and module is not present in x
            dynamic_x = []
            if isinstance(x, dict):
                if self.name in x.keys() and x.get(self.name, {}):
                    print(
                        f"Module {self.name} is static, "
                        f"the parameters {' '.join(x[self.name].keys())} "
                        "passed dynamically will be ignored."
                    )
        unpacked_x = []
        offset = 0
        for name, param in self._params.items():
            if param.dynamic:
                if isinstance(dynamic_x, dict):
                    param_value = dynamic_x[name]
                elif isinstance(dynamic_x, (list, tuple)):
                    param_value = dynamic_x[offset]
                    offset += 1
                elif isinstance(dynamic_x, Tensor):
                    size = prod(param.shape)
                    param_value = dynamic_x[..., offset : offset + size].reshape(
                        param.shape
                    )
                    offset += size
                else:
                    raise ValueError(
                        f"Invalid data type found when unpacking parameters for {self.name}."
                        "Expected argument of unpack to be a list/tuple/dict of Tensor, "
                        "or simply a flattened tensor"
                        f"but found {type(dynamic_x)}."
                    )
            else:  # param is static
                param_value = param.value
            if not isinstance(param_value, Tensor):
                raise ValueError(
                    f"Invalid data type found when unpacking parameters for {self.name}."
                    f"Argument of unpack must contain Tensor, but found {type(param_value)}"
                )
            unpacked_x.append(param_value)
        return unpacked_x

    @property
    def module_params(self) -> NestedNamespaceDict:
        static = NestedNamespaceDict()
        dynamic = NestedNamespaceDict()
        for name, param in self._params.items():
            if param.static:
                static[name] = param
            else:
                dynamic[name] = param
        return NestedNamespaceDict([("static", static), ("dynamic", dynamic)])

    @property
    def params(self) -> NestedNamespaceDict:
        # todo make this an ordinary dict and reorder at the end.
        static = NestedNamespaceDict()
        dynamic = NestedNamespaceDict()

        def _get_params(module):
            mp = module.module_params
            if mp.static:
                static[module.name] = mp.static
            if mp.dynamic:
                dynamic[module.name] = mp.dynamic
            for child in module._childs.values():
                _get_params(child)

        _get_params(self)
        # TODO reorder
        return NestedNamespaceDict([("static", static), ("dynamic", dynamic)])

    @property
    def dynamic_modules(self) -> NamespaceDict[str, "Parametrized"]:
        # Only catch modules with dynamic parameters
        modules = (
            NamespaceDict()
        )  # todo make this an ordinary dict and reorder at the end.

        def _get_childs(module):
            # Start from root, and move down the DAG
            if module.module_params.dynamic:
                modules[module.name] = module
            if module._childs != {}:
                for child in module._childs.values():
                    _get_childs(child)

        _get_childs(self)
        # TODO reorder
        return modules

    def __repr__(self) -> str:
        # TODO: change
        return str(self)

    def __str__(self) -> str:
        static = self.module_params.static
        dynamic = self.module_params.dynamic
        static_str = ", ".join(list(static.keys()))
        dynamic_str = ", ".join(list(dynamic.keys()))
        desc_dynamic_strs = []
        if self.n_dynamic > 0:
            desc_dynamic_strs.append(f"('{self.name}': {list(dynamic.keys())})")

        for n, d in self._childs.items():
            if d.n_dynamic > 0:
                desc_dynamic_strs.append(
                    f"('{n}': {list(d.module_params.dynamic.keys())})"
                )

        desc_dynamic_str = ", ".join(desc_dynamic_strs)

        return (
            f"{self.__class__.__name__}(\n"
            f"    name='{self.name}',\n"
            f"    static=[{static_str}],\n"
            f"    dynamic=[{dynamic_str}],\n"
            f"    x keys=[{desc_dynamic_str}]\n"
            f")"
        )

    def get_graph(
        self, show_dynamic_params: bool = False, show_static_params: bool = False
    ) -> "graphviz.Digraph":  # type: ignore
        """
        Returns a graph representation of the object and its parameters.

        Parameters
        ----------
        show_dynamic_params: (bool, optional)
            If true, the dynamic parameters are shown in the graph. Defaults to False.
        show_static_params: (bool, optional)
            If true, the static parameters are shown in the graph. Defaults to False.

        Returns
        -------
        graphviz.Digraph
            The graph representation of the object.
        """

        def add_component(p: Parametrized, dot):
            dot.attr("node", style="solid", color="black", shape="ellipse")
            dot.node(p.name, f"{p.__class__.__name__}('{p.name}')")

        def add_params(p: Parametrized, dot):
            static = p.module_params.static.keys()
            dynamic = p.module_params.dynamic.keys()

            dot.attr("node", style="solid", color="black", shape="box")
            for n in dynamic:
                if show_dynamic_params:
                    dot.node(f"{p.name}/{n}", n)
                    dot.edge(p.name, f"{p.name}/{n}")

            dot.attr("node", style="filled", color="lightgrey", shape="box")
            for n in static:
                if show_static_params:
                    dot.node(f"{p.name}/{n}", n)
                    dot.edge(p.name, f"{p.name}/{n}")

        dot = graphviz.Digraph(strict=True)
        add_component(self, dot)
        add_params(self, dot)

        for child in self._childs.values():
            add_component(child, dot)

            for parent in child._parents.values():
                if parent.name not in self._childs and parent.name != self.name:
                    continue
                add_component(parent, dot)
                dot.edge(parent.name, child.name)
                add_params(child, dot)

        return dot


def count_args_before_varargs(function):
    """
    Counts the number of arguments before the *args argument in a function.
    """
    signature = inspect.signature(function)
    count = 0

    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            break
        if param.name == "self":
            continue
        count += 1

    return count


def unpack(method):
    """
    Decorator that unpacks the "params" argument of a method.
    There are a number of ways to interact with this method.
    Let's consider a hypothetical lens with a function ``func`` that takes a position ``x`` and ``y`` and two parameters ``a`` and ``b`` and a key word argument ``c``.
    The following are all valid ways to call ``func``:: python

        lens.func(x, y, a=a, b=b, c=c) # a and b are Tensors
        lens.func(x, y, [a, b], c=c) # a and b are Tensors, or even [a, b] is a tensor
        lens.func(x, y, params=[a, b], c=c) # a and b are Tensors, or even [a, b] is a tensor

        # If the ``a`` parameter has been set at a static value like this:
        lens.a = a # a is a Tensor
        # then the following is also valid:
        lens.func(x, y, b=b, c=c) # b is a Tensor
        lens.func(x, y, [b], c=c) # a and b are Tensors
        lens.func(x, y, params=[b], c=c) # a and b are Tensors

        # If lens.func calls another method from a different parametrized object (say cosmo) and that method takes a dynamic parameter ``d``, then the following is also valid:
        lens.func(x, y, a=a, b=b, cosmo_d=d, c=c) # a, b and d are Tensors
        lens.func(x, y, [a, b, d], c=c) # a, b and d are Tensors
        lens.func(x, y, params=[a, b, d], c=c) # a, b and d are Tensors

    In all cases any valid way to construct the ``Packed`` object also works (i.e. by tensor, dict, or tuple).
    This gives a great deal of flexibility in how the parameters are passed to the method.
    However, the following is not a valid way to pass the parameters:: python

        lens.func(x, y, a, b, c=c)

    This is because the ``a`` and ``b`` parameters are not named and so cannot be recognized by the unpack method.


    """

    nargs = count_args_before_varargs(method)

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        # Extract "params" regardless of how it is/they are passed
        # ---------------------------------------------------------
        if len(args) > nargs:
            # Params is given as last argument
            x = self.pack(args[-1])
            args = args[:-1]
        elif "params" in kwargs:
            # Params is given as a keyword argument
            x = self.pack(kwargs.pop("params"))
        elif self.params.dynamic:
            # Params are given individually and are collected into a packed object
            all_keys = self.params.dynamic
            keys = list(all_keys.pop(self.name).keys())
            if all_keys:
                keys += list(k.replace(".", "_") for k in all_keys.flatten().keys())
            try:
                x = self.pack([kwargs.pop(name) for name in keys])
            except KeyError as e:
                raise KeyError(
                    f"Missing parameter {e} in method '{method.__name__}' of module '{self.name}'"
                )
        else:
            # No dynamic parameters, so no packing is needed
            x = Packed()

        # Fill kwargs with module params
        # ---------------------------------------------------------
        for name, value in zip(self._params.keys(), self.unpack(x)):
            kwargs[name] = value

        return method(self, *args, params=x, **kwargs)

    return wrapped
