import pytest

import caustics
from caustics.models.registry import (
    _KindRegistry,
    available_kinds,
    register_kind,
    get_kind,
    _registry,
)
from caustics.parameter import Parameter
from caustics.parametrized import Parametrized


class TestKindRegistry:
    expected_attrs = [
        "cosmology",
        "single_lenses",
        "multi_lenses",
        "light",
        "simulators",
        "known_kinds",
        "_m",
    ]

    def test_constructor(self):
        registry = _KindRegistry()

        for attr in self.expected_attrs:
            assert hasattr(registry, attr)

    @pytest.mark.parametrize("kind", ["NonExistingClass", "SIE", caustics.Sersic])
    def test_getitem(self, kind, mocker):
        registry = _KindRegistry()

        if kind == "NonExistingClass":
            with pytest.raises(KeyError):
                registry[kind]
        elif isinstance(kind, str):
            cls = registry[kind]
            assert cls == getattr(caustics, kind)
        else:
            test_key = "TestSersic"
            registry.known_kinds[test_key] = kind
            cls = registry[test_key]
            assert cls == kind

    @pytest.mark.parametrize("kind", [Parameter, caustics.Sersic, "caustics.SIE"])
    def test_setitem(self, kind):
        registry = _KindRegistry()
        key = "TestSersic"
        if isinstance(kind, str):
            registry[key] = kind
            assert key in registry._m
        elif issubclass(kind, Parametrized):
            registry[key] = kind
            assert registry[key] == kind
        else:
            with pytest.raises(ValueError):
                registry[key] = kind

    def test_delitem(self):
        registry = _KindRegistry()
        with pytest.raises(NotImplementedError):
            del registry["Sersic"]

    def test_len(self):
        registry = _KindRegistry()
        assert len(registry) == len(set(registry._m))

    def test_iter(self):
        registry = _KindRegistry()
        assert set(registry) == set(registry._m)


def test_available_kinds():
    assert available_kinds() == list(_registry)


def test_register_kind():
    key = "TestSersic2"
    value = caustics.Sersic
    register_kind(key, value)
    assert key in _registry._m
    assert _registry[key] == value

    with pytest.raises(ValueError):
        register_kind("SIE", "caustics.SIE")


def test_get_kind():
    kind = "Sersic"
    cls = get_kind(kind)
    assert cls == caustics.Sersic
    kind = "NonExistingClass"
    with pytest.raises(KeyError):
        cls = get_kind(kind)
