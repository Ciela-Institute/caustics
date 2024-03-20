import pytest
import inspect
import typing
from typing import Annotated, Dict
from caustics.models.registry import _registry, get_kind
from caustics.models.utils import (
    create_pydantic_model,
    setup_pydantic_models,
    setup_simulator_models,
    _get_kwargs_field_definitions,
    PARAMS,
    INIT_KWARGS,
)
from caustics.models.base_models import Base
from caustics.parametrized import ClassParam


@pytest.fixture(params=_registry.known_kinds)
def kind(request):
    return request.param


@pytest.fixture
def parametrized_class(kind):
    return get_kind(kind)


def test_create_pydantic_model(kind):
    model = create_pydantic_model(kind)
    kind_cls = get_kind(kind)
    expected_fields = {"kind", "name", "params", "init_kwargs"}

    assert model.__base__ == Base
    assert model.__name__ == kind
    assert model._cls == kind_cls
    assert set(model.model_fields.keys()) == expected_fields


def test__get_kwargs_field_definitions(parametrized_class):
    kwargs_fd = _get_kwargs_field_definitions(parametrized_class)

    cls_signature = inspect.signature(parametrized_class)
    class_metadata = {
        k: {
            "dtype": v.annotation.__origin__,
            "default": v.default,
            "class_param": ClassParam(*v.annotation.__metadata__),
        }
        for k, v in cls_signature.parameters.items()
    }

    for k, v in class_metadata.items():
        if k != "name":
            if v["class_param"].isParam:
                assert k in kwargs_fd[PARAMS]
                assert isinstance(kwargs_fd[PARAMS][k], tuple)
                assert kwargs_fd[PARAMS][k][0] == v["dtype"]
                field_info = kwargs_fd[PARAMS][k][1]
            else:
                assert k in kwargs_fd[INIT_KWARGS]
                assert isinstance(kwargs_fd[INIT_KWARGS][k], tuple)
                assert kwargs_fd[INIT_KWARGS][k][0] == v["dtype"]
                field_info = kwargs_fd[INIT_KWARGS][k][1]

            if v["default"] == inspect._empty:
                # Skip empty defaults
                continue
            assert field_info.default == v["default"]


def _check_nested_discriminated_union(
    input_anno: type[Annotated], class_paths: Dict[str, str]
):
    # Check to see if the model selection is Annotated type
    assert typing.get_origin(input_anno) == Annotated
    # Check to see if the discriminator is "kind"
    assert input_anno.__metadata__[0].discriminator == "kind"

    if typing.get_origin(input_anno.__origin__) == typing.Union:
        models = input_anno.__origin__.__args__
    else:
        # For single models
        models = [input_anno.__origin__]

    # Check to see if the models are in the registry
    assert len(models) == len(class_paths)
    # Go through each model and check that it's pointing to the right class
    for model in models:
        assert model.__name__ in class_paths
        assert model._cls == get_kind(model.__name__)


def test_setup_pydantic_models():
    # light, lenses
    pydantic_models_annotated = setup_pydantic_models()

    registry_dict = {
        "light": _registry.light,
        "lenses": {
            **_registry.single_lenses,
            **_registry.multi_lenses,
        },
    }

    pm_anno_dict = {
        k: v for (k, v) in zip(list(registry_dict.keys()), pydantic_models_annotated)
    }

    for key, pydantic_model_anno in pm_anno_dict.items():
        class_paths = registry_dict[key]
        _check_nested_discriminated_union(pydantic_model_anno, class_paths)


def test_setup_simulator_models():
    simulators = setup_simulator_models()

    class_paths = _registry.simulators
    _check_nested_discriminated_union(simulators, class_paths)
