from collections import OrderedDict, defaultdict
from math import prod
from operator import itemgetter
from typing import Optional, Union

import torch
import re
from torch import Tensor

from .packed import Packed
from .namespace_dict import NamespaceDict, NestedNamespaceDict
from .parameter import Parameter

__all__ = ("Parametrized",)

class Parametrized:
    """
    Represents a class with Param and Parametrized attributes, typically used to construct parts of a simulator
    that have parameters which need to be tracked during MCMC sampling.

    This class can contain Params, Parametrized, tensor buffers or normal attributes as its attributes.
    It provides functionalities to manage these attributes, ensuring that an attribute of one type isn't rebound
    to be of a different type.

    TODO
    - Attributes can be Params, Parametrized, tensor buffers or just normal attributes.
    - Need to make sure an attribute of one of those types isn't rebound to be of a different type.

    Attributes:
        name (str): The name of the Parametrized object. Default to class name.
        parents (NestedNamespaceDict): Nested dictionary of parent Parametrized objects (higher level, more abstract modules).
        params (OrderedDict[str, Parameter]): Dictionary of parameters.
        childs NestedNamespaceDict: Nested dictionary of childs Parametrized objects (lower level, more specialized modules).
        dynamic_size (int): Size of dynamic parameters.
        n_dynamic (int): Number of dynamic parameters.
        n_static (int): Number of static parameters.
    """

    def __init__(self, name: str = None):
        if name is None:
            name = re.search("([A-Z])\w+", str(self.__class__)).group()
        if not isinstance(name, str):
            raise ValueError(f"name must be a string (received {name})")
        self._name = name
        self._parents: OrderedDict[str, Parametrized] = NamespaceDict()
        self._params: OrderedDict[str, Parameter] = NamespaceDict()
        self._childs: OrderedDict[str, Parametrized] = NamespaceDict()
        self._dynamic_size = 0
        self._n_dynamic = 0
        self._n_static = 0

    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, new_name: str):
        old_name = self.name
        for parent in self._parents.values():
            parent._childs[new_name] = self
            del parent._childs[old_name]
        for child in self._childs.values():
            child._parents[new_name] = self
            del child._parents[old_name]
        self._name = new_name

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Moves static Params for this component and its childs to the specified device and casts them to the specified data type.
        """
        # TODO I think we should make users specify a separate method so we dont override this behavior
        for name in self._params.keys():
            param = self._params[name]
            if isinstance(param, torch.Tensor):
                self._params[name] = param.to(device, dtype)
        for child in self._childs.values():
            child.to(device, dtype)
        return self

    @staticmethod
    def _generate_unique_name(name, module_names):
        i = 1
        while f"{name}_{i}" in module_names:
            i += 1
        return f"{name}_{i}"
                
    def add_parametrized(self, p: "Parametrized"):
        """
        Add a child to this module, and create edges for the DAG
        """
        # If self.name is already in the module parents, we need to update self.name
        if self.name in p._parents.keys():
            new_name = self._generate_unique_name(self.name, p._parents.keys())
            self.name = new_name # from name.setter, this updates the DAG edges as well
        p._parents[self.name] = self
        # If the module name is already in self._childs, we need to update module name
        if p.name is self._childs.keys():
            new_name = self._generate_unique_name(p.name, self._childs.keys())
            p.name = new_name
        self._childs[p.name] = p

    def add_param(
        self,
        name: str,
        value: Optional[Union[Tensor, float]] = None,
        shape: Optional[tuple[int, ...]] = (),
    ):
        """
        Stores a parameter in the _params dictionary and records its size.

        Args:
            name (str): The name of the parameter.
            value (Optional[Tensor], optional): The value of the parameter. Defaults to None.
            shape (Optional[tuple[int, ...]], optional): The shape of the parameter. Defaults to an empty tuple.
        """
        self._params[name] = Parameter(value, shape)
        if value is None:
            assert isinstance(shape, tuple)  # quiet pyright error
            size = prod(shape)
            self._dynamic_size += size
            self._n_dynamic += 1
        else:
            self._n_static += 1

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            raise ValueError("cannot add Params directly as attributes: use add_param instead")
        elif isinstance(val, Parametrized):
            self.add_parametrized(val)
            super().__setattr__(key, val)
        else:
            super().__setattr__(key, val)

    @property
    def n_dynamic(self) -> int:
        return self._n_dynamic

    @property
    def n_static(self) -> int:
        return self._n_static

    @property
    def dynamic_size(self) -> int:
        return self._dynamic_size

    def pack(
        self,
        x: Union[
            list[Tensor],
            dict[str, Union[list[Tensor], Tensor, dict[str, Tensor]]],
            Tensor,
        ],
    ) -> Packed:
        """
        Converts a list or tensor into a dict that can subsequently be unpacked
        into arguments to this component and its childs.

        Args:
            x (Union[list[Tensor], dict[str, Union[list[Tensor], Tensor, dict[str, Tensor]]], Tensor):
                The input to be packed. Can be a list of tensors, a dictionary of tensors, or a single tensor.

        Returns:
            Packed: The packed input.

        Raises:
            ValueError: If the input is not a list, dictionary, or tensor.
            ValueError: If the input is a dictionary and some keys are missing.
            ValueError: If the number of dynamic arguments does not match the expected number.
            ValueError: If the input is a tensor and the shape does not match the expected shape.
        """
        if isinstance(x, (dict, Packed)):
            missing_names = [name for name in self.params.dynamic.keys() if name not in x]
            if len(missing_names) > 0:
                raise ValueError(f"missing x keys for {missing_names}")

            # TODO: check structure!
            return Packed(x)

        elif isinstance(x, list) or isinstance(x, tuple):
            n_passed = len(x)
            n_dynamic_params = len(self.params.dynamic.flatten())
            n_dynamic_modules = len(self.dynamic_modules)
            if n_passed not in [n_dynamic_modules, n_dynamic_params]:
                # TODO: give component and arg names
                raise ValueError(
                    f"{n_passed} dynamic args were passed, but {n_dynamic_params} parameters or "
                    f"{n_dynamic_modules} Tensor (1 per dynamic module) are required"
                )

            elif n_passed == n_dynamic_params:
                cur_offset = self.n_dynamic
                x_repacked = {self.name: x[:cur_offset]}
                for name, dynamic_module in self.dynamic_modules.items():
                    x_repacked[name] = x[cur_offset : cur_offset + dynamic_module.n_dynamic]
                    cur_offset += dynamic_module.n_dynamic
            elif n_passed == n_dynamic_modules:
                x_repacked = {}
                for i, name in enumerate(self.dynamic_modules.keys()):
                    x_repacked[name] = x[i] 
            return Packed(x_repacked)
        
        elif isinstance(x, Tensor):
            n_passed = x.shape[-1]
            n_expected = sum([module.dynamic_size for module in self.dynamic_modules.values()]) 
            if n_passed != n_expected:
                # TODO: give component and arg names
                raise ValueError(
                    f"{n_passed} flattened dynamic args were passed, but {n_expected} "
                    f"are required"
                )

            cur_offset = self.dynamic_size
            x_repacked = {self.name: x[..., :cur_offset]}
            for desc in self._childs.values():
                x_repacked[desc.name] = x[..., cur_offset : cur_offset + desc.dynamic_size]
                cur_offset += desc.dynamic_size
            return Packed(x_repacked)

        else:
            raise ValueError("Data structure not supported")

    def unpack(
        self, x: Optional[dict[str, Union[list[Tensor], dict[str, Tensor], Tensor]]]
    ) -> list[Tensor]:
        """
        Unpacks a dict of kwargs, list of args or flattened vector of args to retrieve
        this object's static and dynamic parameters.

        Args:
            x (Optional[dict[str, Union[list[Tensor], dict[str, Tensor], Tensor]]]):
                The packed object to be unpacked.

        Returns:
            list[Tensor]: Unpacked static and dynamic parameters of the object.

        Raises:
            ValueError: If the input is not a dict, list, tuple or tensor.
            ValueError: If a static parameter is passed dynamically.
            ValueError: If the argument type is invalid. It must be a dict containing key {self.name}
                and value containing args as list or flattened tensor, or kwargs.
        """
        my_x = defaultdict(list) if x is None else x[self.name]
        if isinstance(my_x, dict):
            # Parse dynamic kwargs
            args = []
            for name, p in self._params.items():
                if p.value is None:
                    # Dynamic Param
                    args.append(my_x[name])
                else:
                    if name in my_x:
                        raise ValueError(
                            f"{name} was passed dynamically as a kwarg for {self.name}, "
                            "but it is a static parameter"
                        )

                    args.append(p.value)

            return args
        elif isinstance(my_x, list) or isinstance(x, tuple):
            # Parse dynamic args
            vals = []
            offset = 0
            for param in self._params.values():
                if not param.dynamic:
                    vals.append(param.value)
                else:
                    vals.append(my_x[offset])
                    offset += 1

            return vals
        elif isinstance(my_x, Tensor):
            # Parse dynamic parameter vector
            vals = []
            offset = 0
            for param in self._params.values():
                if not param.dynamic:
                    vals.append(param.value)
                else:
                    size = prod(param.shape)
                    vals.append(my_x[..., offset : offset + size].reshape(param.shape))
                    offset += size

            return vals
        else:
            raise ValueError(
                f"invalid argument type: must be a dict containing key {self.name} "
                "and value containing args as list or flattened tensor, or kwargs"
            )

    def __getattribute__(self, key):
        """
        Enables accessing static params as attributes.

        Args:
            key (str): Name of the attribute.

        Returns:
            Any: The attribute value if found.

        Raises:
            AttributeError: If the attribute is not found and it's not a static parameter.
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
        static = NestedNamespaceDict()
        dynamic = NestedNamespaceDict()
        def _get_params(module):
            if module._childs != {}:
                for child in module._childs.values():
                    _get_params(child)
            if module.module_params.static:
                static[module.name] = module.module_params.static
            if module.module_params.dynamic:
                dynamic[module.name] = module.module_params.dynamic
        _get_params(self)
        return NestedNamespaceDict([("static", static), ("dynamic", dynamic)])

    @property
    def dynamic_modules(self) -> NamespaceDict[str, "Parametrized"]:
        # Only catch modules with dynamic parameters
        modules = NamespaceDict()
        def _get_childs(module):
            if module._childs != {}:
                for child in module._childs.values():
                    _get_childs(child)
            if module.module_params.dynamic:
                modules[module.name] = module
        _get_childs(self)
        return modules

    def __repr__(self) -> str:
        # TODO: change
        return str(self)

    def __str__(self) -> str:
        static, dynamic = itemgetter("static", "dynamic")(self.module_params)
        static_str = ", ".join(list(static.keys()))
        dynamic_str = ", ".join(list(dynamic.keys()))
        desc_dynamic_strs = []
        if self.n_dynamic > 0:
            desc_dynamic_strs.append(f"('{self.name}': {list(dynamic.keys())})")

        for n, d in self._childs.items():
            if d.n_dynamic > 0:
                desc_dynamic_strs.append(f"('{n}': {list(d.module_params.dynamic.keys())})")

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

        Args:
            show_dynamic_params (bool, optional): If true, the dynamic parameters are shown in the graph. Defaults to False.
            show_static_params (bool, optional): If true, the static parameters are shown in the graph. Defaults to False.

        Returns:
            graphviz.Digraph: The graph representation of the object.
        """
        import graphviz

        def add_component(p: Parametrized, dot):
            dot.attr("node", style="solid", color="black", shape="ellipse")
            dot.node(p.name, f"{p.__class__.__name__}('{p.name}')")

        def add_params(p: Parametrized, dot):
            static, dynamic = itemgetter("static", "dynamic")(p.module_params)

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

