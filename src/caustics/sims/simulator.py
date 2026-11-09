# mypy: disable-error-code="import-untyped,var-annotated"
from inspect import signature
from typing import Annotated, TextIO

import yaml
from caskade import Module

import caustics

__all__ = ("NameType", "build_simulator")

NameType = Annotated[str | None, "Name of the simulator"]


def build_simulator(config: str | TextIO) -> Module:

    if isinstance(config, str):
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = yaml.safe_load(config)

    modules = {}
    for name, obj in config_dict.items():
        kwargs = obj.get("init_kwargs", {})
        for kwarg in kwargs:
            for subname, subobj in config_dict.items():
                if subname == name:  # only look at previous objects
                    break
                if subobj == kwargs[kwarg] and isinstance(kwargs[kwarg], dict):
                    # fill already constructed object
                    kwargs[kwarg] = modules[subname]

        # Get the caustics object, using a "." path if given
        base = caustics
        for part in obj["kind"].split("."):
            base = getattr(base, part)
        if "name" in signature(base).parameters:  # type: ignore[arg-type]
            kwargs["name"] = name
        # Instantiate the caustics object
        modules[name] = base(**kwargs)  # type: ignore[operator]

    # return the last object
    return modules[tuple(modules.keys())[-1]]
