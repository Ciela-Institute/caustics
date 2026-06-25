import sys
import os
import torch
import pytest
import matplotlib
import matplotlib.pyplot as plt
from caustics.backend_obj import backend

# Add the helpers directory to the path so we can import the helpers
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

# from caustics.models.utils import setup_pydantic_models

if backend.backend == "torch":
    CUDA_AVAILABLE = torch.cuda.is_available()
elif backend.backend == "jax":
    CUDA_AVAILABLE = any(d.platform == "gpu" for d in backend.jax.devices())


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
    if backend.backend == "torch":
        return torch.device(request.param)
    elif backend.backend == "jax":
        param = "gpu" if request.param == "cuda" else "cpu"
        print(
            f"Available {param} devices: {backend.jax.devices(param)}, sending {backend.jax.devices(param)[0]}"
        )
        return backend.jax.devices(param)[0]
    return None


@pytest.fixture(autouse=True)
def no_block_show(monkeypatch):
    def close_show(*args, **kwargs):
        # plt.savefig("/dev/null")  # or do nothing
        plt.close("all")

    monkeypatch.setattr(plt, "show", close_show)

    # Also ensure we are in a non-GUI backend
    matplotlib.use("Agg")
