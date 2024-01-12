from typing import Dict
import pytest
import torch
from safetensors.torch import save, load
from datetime import datetime as dt
from caustics.parameter import Parameter
from caustics.namespace_dict import NamespaceDict, NestedNamespaceDict
from caustics.sims.state_dict import StateDict, IMMUTABLE_ERR
from caustics import __version__


class TestStateDict:
    simple_tensors = {"var1": torch.as_tensor(1.0), "var2": torch.as_tensor(2.0)}

    @pytest.fixture(scope="class")
    def simple_state_dict(self):
        return StateDict(self.simple_tensors)

    def test_constructor(self):
        time_format = "%Y-%m-%dT%H:%M:%S"
        time_str_now = dt.utcnow().strftime(time_format)
        state_dict = StateDict(self.simple_tensors)

        # Get the created time and format to nearest seconds
        sd_ct_dt = dt.fromisoformat(state_dict._metadata["created_time"])
        sd_ct_str = sd_ct_dt.strftime(time_format)

        # Check the default metadata and content
        assert hasattr(state_dict, "_metadata")
        assert state_dict._created is True
        assert state_dict._metadata["software_version"] == __version__
        assert sd_ct_str == time_str_now
        assert dict(state_dict) == self.simple_tensors

    def test_setitem(self, simple_state_dict):
        with pytest.raises(type(IMMUTABLE_ERR), match=str(IMMUTABLE_ERR)):
            simple_state_dict["var1"] = torch.as_tensor(3.0)

    def test_delitem(self, simple_state_dict):
        with pytest.raises(type(IMMUTABLE_ERR), match=str(IMMUTABLE_ERR)):
            del simple_state_dict["var1"]

    def test_from_params(self, simple_common_sim):
        params: NestedNamespaceDict = simple_common_sim.params

        # flatten function only exists for NestedNamespaceDict
        static_params: NamespaceDict = params["static"].flatten()
        tensors_dict: Dict[str, torch.Tensor] = {
            k: v.value for k, v in static_params.items()
        }

        expected_state_dict = StateDict(tensors_dict)

        # Full parameters
        state_dict = StateDict.from_params(params)
        assert state_dict == expected_state_dict

        # Static only
        state_dict = StateDict.from_params(static_params)
        assert state_dict == expected_state_dict

    def test_to_params(self, simple_state_dict):
        params = simple_state_dict.to_params()
        assert isinstance(params, NamespaceDict)

        for k, v in params.items():
            tensor_value = simple_state_dict[k]
            assert isinstance(v, Parameter)
            assert v.value == tensor_value

    def test__to_safetensors(self):
        state_dict = StateDict(self.simple_tensors)
        # Save to safetensors
        tensors_bytes = state_dict._to_safetensors()
        expected_bytes = save(state_dict, metadata=state_dict._metadata)

        # Reload to back to tensors dict
        # this is done because the information
        # might be stored in different arrangements
        # within the safetensors bytes
        loaded_tensors = load(tensors_bytes)
        loaded_expected_tensors = load(expected_bytes)
        assert loaded_tensors == loaded_expected_tensors
