import sys
import os
import torch
import pytest

# Add the helpers directory to the path so we can import the helpers
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


@pytest.fixture(params=["cpu", "cuda"])
def tensor_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)
