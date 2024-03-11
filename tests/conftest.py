import sys
import os
import torch
import pytest
import typing

# Add the helpers directory to the path so we can import the helpers
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from caustics.models.utils import setup_pydantic_models

CUDA_AVAILABLE = torch.cuda.is_available()

LIGHT_ANNOTATED, LENSES_ANNOTATED = setup_pydantic_models()


def _get_models(annotated):
    typehint = typing.get_args(annotated)[0]
    pydantic_models = typing.get_args(typehint)
    if isinstance(pydantic_models, tuple):
        pydantic_models = {m.__name__: m for m in pydantic_models}
    else:
        pydantic_models = {pydantic_models.__name__: pydantic_models}
    return pydantic_models


@pytest.fixture
def light_models():
    return _get_models(LIGHT_ANNOTATED)


@pytest.fixture
def lens_models():
    return _get_models(LENSES_ANNOTATED)


@pytest.fixture(params=["yaml", "no_yaml"])
def sim_source(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            "cpu", marks=pytest.mark.skipif(CUDA_AVAILABLE, reason="CUDA available")
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
        ),
    ]
)
def device(request):
    return torch.device(request.param)
