from functools import lru_cache
from collections import ChainMap
from typing import MutableMapping, Iterator, Optional

from caustics.parametrized import Parametrized
from caustics.utils import _import_func_or_class


class _KindRegistry(MutableMapping[str, "Parametrized | str"]):
    cosmology = {
        "FlatLambdaCDM": "caustics.cosmology.FlatLambdaCDM.FlatLambdaCDM",
    }
    single_lenses = {
        "EPL": "caustics.lenses.epl.EPL",
        "ExternalShear": "caustics.lenses.external_shear.ExternalShear",
        "PixelatedConvergence": "caustics.lenses.pixelated_convergence.PixelatedConvergence",
        "NFW": "caustics.lenses.nfw.NFW",
        "Point": "caustics.lenses.point.Point",
        "PseudoJaffe": "caustics.lenses.pseudo_jaffe.PseudoJaffe",
        "SIE": "caustics.lenses.sie.SIE",
        "SIS": "caustics.lenses.sis.SIS",
        "TNFW": "caustics.lenses.tnfw.TNFW",
        "MassSheet": "caustics.lenses.mass_sheet.MassSheet",
    }
    multi_lenses = {
        "SinglePlane": "caustics.lenses.singleplane.SinglePlane",
        "Multiplane": "caustics.lenses.multiplane.Multiplane",
    }
    light = {
        "Pixelated": "caustics.light.pixelated.Pixelated",
        "Sersic": "caustics.light.sersic.Sersic",
    }
    simulators = {"Lens_Source": "caustics.sims.lens_source.Lens_Source"}

    known_kinds = {
        **cosmology,
        **single_lenses,
        **multi_lenses,
        **light,
        **simulators,
    }

    def __init__(self) -> None:
        self._m: ChainMap[str, "Parametrized | str"] = ChainMap({}, self.known_kinds)  # type: ignore

    def __getitem__(self, item: str) -> Parametrized:
        kind_mod: "str | Parametrized | None" = self._m.get(item, None)
        if kind_mod is None:
            raise KeyError(f"{item} not in registry")
        if isinstance(kind_mod, str):
            cls = _import_func_or_class(kind_mod)
        else:
            cls = kind_mod  # type: ignore
        return cls  # type: ignore

    def __setitem__(self, item: str, value: "Parametrized | str") -> None:
        if not (
            (isinstance(value, type) and issubclass(value, Parametrized))
            or isinstance(value, str)
        ):
            raise ValueError(
                f"expected Parametrized subclass, got: {type(value).__name__!r}"
            )
        self._m[item] = value

    def __delitem__(self, __v: str) -> None:
        raise NotImplementedError("removal is unsupported")

    def __len__(self) -> int:
        return len(set(self._m))

    def __iter__(self) -> Iterator[str]:
        return iter(set(self._m))


_registry = _KindRegistry()


def available_kinds() -> list[str]:
    """
    Return a list of classes that are available in the registry.
    """
    return list(_registry)


def register_kind(
    name: str,
    cls: "Parametrized | str",
    *,
    clobber: bool = False,
) -> None:
    """register a UPath implementation with a protocol

    Parameters
    ----------
    name : str
        Protocol name to associate with the class
    cls : Parametrized or str
        The caustics parametrized subclass or a str representing the
        full path to the class like package.module.class.
    clobber:
        Whether to overwrite a protocol with the same name; if False,
        will raise instead.
    """
    if not clobber and name in _registry:
        raise ValueError(f"{name!r} is already in registry and clobber is False!")
    _registry[name] = cls


@lru_cache
def get_kind(
    name: str,
) -> Optional[Parametrized]:
    """Get a class from the registry by name.

    Parameters
    ----------
    kind : str
        The name of the kind to get.

    Returns
    -------
    cls : Parametrized
        The class associated with the given name.
    """
    return _registry[name]
