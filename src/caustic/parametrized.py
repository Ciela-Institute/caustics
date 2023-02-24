from collections import OrderedDict
from math import prod
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

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
        if not isinstance(name, str):
            raise ValueError(f"name must be a string (received {name})")
        self.name = name
        self._parents: List[Parametrized] = []
        self._params: OrderedDict[str, Param] = OrderedDict()
        self._descendants: OrderedDict[str, Parametrized] = OrderedDict()
        self._dynamic_size = 0
        self._n_dynamic = 0
        self._n_static = 0

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """
        Put static Params for this component and its descendants on the given device.
        """
        for param in self._params.values():
            param.to(device, dtype)

        for desc in self._descendants.values():
            desc.to(device, dtype)

        return self

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

    def _can_add(self, p: "Parametrized") -> bool:
        """
        Returns False if a different model component with the same name is already
        in the DAG.
        """
        if self.name == p.name and self is not p:
            return False

        if p.name in self._descendants and self._descendants[p.name] is not p:
            return False

        # Propagate up through all parents
        for parent in self._parents:
            if not parent._can_add(p):
                return False

        return True

    def _merge_descendants_up(self, p: "Parametrized"):
        self._descendants[p.name] = p
        for desc in p._descendants.values():
            self._descendants[desc.name] = desc

        # Recur up into parents
        for parent in self._parents:
            parent._merge_descendants_up(p)

    def __setattr__(self, key, val):
        if isinstance(val, Param):
            raise ValueError(
                "cannot add Params directly as attributes: use add_param instead"
            )

        if isinstance(val, Parametrized):
            # Check if new component can be added
            if not self._can_add(val):
                val_str = (
                    str(val)
                    .replace("(\n    ", "(")
                    .replace("\n)", ")")
                    .replace("\n    ", " ")
                )
                raise KeyError(
                    f"cannot add {val_str}: a component with the name '{val.name}' "
                    "already exists in the model DAG"
                )

            # Check if its descendants can be added
            for name, desc in val._descendants.items():
                if not self._can_add(desc):
                    val_str = (
                        str(val)
                        .replace("(\n", "(")
                        .replace("\n)", ")")
                        .replace("\n    ", " ")
                    )
                    raise KeyError(
                        f"cannot add {val_str}: its descendant '{name}' already exists "
                        "in the model DAG"
                    )

            # Add new component and its descendants to this component's descendants,
            # as well as those of its parents
            self._merge_descendants_up(val)

            # Add this component as a parent for the new one
            val._parents.append(self)

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
        x: List[Tensor] | Dict[str, List[Tensor] | Tensor | Dict[str, Tensor]] | Tensor,
    ) -> Dict[str, Any]:
        """
        Converts a list or tensor into a dict that can subsequently be unpacked
        into arguments to this component and its descendants.
        """
        if isinstance(x, Dict):
            # TODO: check structure!
            return x
        elif isinstance(x, list):
            n_passed = len(x)
            n_expected = (
                sum([desc.n_dynamic for desc in self._descendants.values()])
                + self.n_dynamic
            )
            if n_passed != n_expected:
                # TODO: give arg names
                raise ValueError(
                    f"{n_passed} dynamic args were passed, but {n_expected} are "
                    "required"
                )

            cur_offset = self.n_dynamic
            x_repacked = {self.name: x[:cur_offset]}
            for desc in self._descendants.values():
                x_repacked[desc.name] = x[cur_offset : cur_offset + desc.n_dynamic]
                cur_offset += desc.n_dynamic

            return x_repacked
        elif isinstance(x, Tensor):
            n_passed = x.shape[-1]
            n_expected = (
                sum([desc.dynamic_size for desc in self._descendants.values()])
                + self.dynamic_size
            )
            if n_passed != n_expected:
                # TODO: give arg names
                raise ValueError(
                    f"{n_passed} flattened dynamic args were passed, but {n_expected}"
                    " are required"
                )

            cur_offset = self.dynamic_size
            x_repacked = {self.name: x[..., :cur_offset]}
            for desc in self._descendants.values():
                x_repacked[desc.name] = x[
                    ..., cur_offset : cur_offset + desc.dynamic_size
                ]
                cur_offset += desc.dynamic_size

            return x_repacked
        else:
            raise ValueError("can only repack a list or 1D tensor")

    def unpack(
        self, x: Dict[str, List[Tensor] | Dict[str, Tensor] | Tensor]
    ) -> List[Tensor]:
        """
        Unpacks a dict of kwargs, list of args or flattened vector of args to retrieve
        this object's static and dynamic parameters.
        """
        my_x = x[self.name]
        if isinstance(my_x, Dict):
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
        elif isinstance(my_x, List):
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
    def params(self) -> Dict[str, OrderedDict[str, Param]]:
        """
        Gets all of the static and dynamic Params for this component and its descendants
        in the correct order.
        """
        static = OrderedDict()
        dynamic = OrderedDict()
        for name, param in self._params.items():
            if param.static:
                static[name] = param
            else:
                dynamic[name] = param

        return {"static": static, "dynamic": dynamic}

    @property
    def static_params(self) -> OrderedDict[str, Param]:
        return self.params["static"]

    @property
    def dynamic_params(self) -> OrderedDict[str, Param]:
        return self.params["dynamic"]

    def __repr__(self) -> str:
        # TODO: change
        return str(self)

    def __str__(self) -> str:
        static, dynamic = itemgetter("static", "dynamic")(self.params)
        static_str = ", ".join(list(static.keys()))
        dynamic_str = ", ".join(list(dynamic.keys()))

        desc_dynamic_strs = []
        if self.n_dynamic > 0:
            desc_dynamic_strs.append(f"('{self.name}': {list(dynamic.keys())})")

        for n, d in self._descendants.items():
            if d.n_dynamic > 0:
                desc_dynamic_strs.append(f"('{n}': {list(d.params['dynamic'].keys())})")

        desc_dynamic_str = ", ".join(desc_dynamic_strs)

        return (
            f"{self.__class__.__name__}(\n"
            f"    name='{self.name}',\n"
            f"    static=[{static_str}],\n"
            f"    dynamic=[{dynamic_str}],\n"
            f"    x keys=[{desc_dynamic_str}]\n"
            f")"
        )

    def graph(
        self, show_dynamic_params: bool = False, show_static_params: bool = False
    ) -> "graphviz.Digraph":  # type: ignore
        from operator import itemgetter

        import graphviz

        def add_component(p: Parametrized, dot):
            dot.attr("node", style="solid", color="black", shape="ellipse")
            dot.node(p.name, f"{p.__class__.__name__}('{p.name}')")

        def add_params(p: Parametrized, dot):
            static, dynamic = itemgetter("static", "dynamic")(p.params)

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

        dot = graphviz.Digraph()
        add_component(self, dot)
        add_params(self, dot)

        for desc in self._descendants.values():
            add_component(desc, dot)

            for parent in desc._parents:
                add_component(parent, dot)

                dot.edge(parent.name, desc.name)

                add_params(desc, dot)

        return dot
