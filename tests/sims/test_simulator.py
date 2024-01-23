import pytest
from pathlib import Path
import sys

import torch

from caustics.sims.state_dict import StateDict
from helpers.sims import extract_tensors


@pytest.fixture
def state_dict(simple_common_sim):
    return simple_common_sim.state_dict()


@pytest.fixture
def expected_tensors(simple_common_sim):
    return extract_tensors(simple_common_sim.params)


class TestSimulator:
    def test_state_dict(self, state_dict, expected_tensors):
        # Check state_dict type and default keys
        assert isinstance(state_dict, StateDict)

        # Trying to modify state_dict should raise TypeError
        with pytest.raises(TypeError):
            state_dict["params"] = -1

        # Check _metadata keys
        assert "software_version" in state_dict._metadata
        assert "created_time" in state_dict._metadata

        # Check params
        assert dict(state_dict) == expected_tensors

    def test_set_module_params(self, simple_common_sim):
        params = {"param1": torch.as_tensor(1), "param2": torch.as_tensor(2)}
        # Call the __set_module_params method
        simple_common_sim._Simulator__set_module_params(simple_common_sim, params)

        # Check if the module attributes have been set correctly
        assert simple_common_sim.param1 == params["param1"]
        assert simple_common_sim.param2 == params["param2"]

    def test_load_state_dict(self, simple_common_sim):
        fpath = simple_common_sim.state_dict().save()
        loaded_state_dict = StateDict.load(fpath)

        # Change a value in the simulator
        simple_common_sim.z_s = 3.0

        # Ensure that the simulator has been changed
        assert (
            loaded_state_dict[f"{simple_common_sim.name}.z_s"]
            != simple_common_sim.z_s.value
        )

        # Load the state dict form file
        simple_common_sim.load_state_dict(fpath)

        # Once loaded now the values should be the same
        assert (
            loaded_state_dict[f"{simple_common_sim.name}.z_s"]
            == simple_common_sim.z_s.value
        )

        # Cleanup after only for non-windows
        if not sys.platform.startswith("win"):
            Path(fpath).unlink(missing_ok=True)
