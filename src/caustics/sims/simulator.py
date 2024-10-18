# mypy: disable-error-code="import-untyped,var-annotated"
from typing import Annotated, Optional, Union
from caskade import Module
import yaml
from pathlib import Path

import caustics

__all__ = ("NameType", "build_simulator")

NameType = Annotated[Optional[str], "Name of the simulator"]


def build_simulator(config: Union[str, Path]) -> Module:

    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)

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

        modules[name] = getattr(caustics, obj["kind"])(name=name, **kwargs)

    # return the last object
    return modules[tuple(modules.keys())[-1]]
