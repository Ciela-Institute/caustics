import sys
import os
import torch
import pytest
import matplotlib
import matplotlib.pyplot as plt

# Add the helpers directory to the path so we can import the helpers
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

# from caustics.models.utils import setup_pydantic_models

CUDA_AVAILABLE = torch.cuda.is_available()


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


@pytest.fixture(autouse=True)
def no_block_show(monkeypatch):
    def close_show(*args, **kwargs):
        # plt.savefig("/dev/null")  # or do nothing
        plt.close("all")

    monkeypatch.setattr(plt, "show", close_show)

    # Also ensure we are in a non-GUI backend
    matplotlib.use("Agg")
