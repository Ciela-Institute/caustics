import sys
import os
import torch
import pytest

# Add the helpers directory to the path so we can import the helpers
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

CUDA_AVAILABLE = torch.cuda.is_available()


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
