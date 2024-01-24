from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import pytest
import torch
from collections import OrderedDict
from safetensors.torch import save, load
from datetime import datetime as dt
from caustics.parameter import Parameter
from caustics.namespace_dict import NamespaceDict, NestedNamespaceDict
from caustics.sims.state_dict import ImmutableODict, StateDict, IMMUTABLE_ERR, _sanitize
from caustics import __version__

from helpers.sims import extract_tensors


class TestImmutableODict:
    def test_constructor(self):
        odict = ImmutableODict(a=1, b=2, c=3)
        assert isinstance(odict, OrderedDict)
        assert odict == {"a": 1, "b": 2, "c": 3}
        assert hasattr(odict, "_created")
        assert odict._created is True

    def test_setitem(self):
        odict = ImmutableODict()
        with pytest.raises(type(IMMUTABLE_ERR), match=str(IMMUTABLE_ERR)):
            odict["key"] = "value"

    def test_delitem(self):
        odict = ImmutableODict(key="value")
        with pytest.raises(type(IMMUTABLE_ERR), match=str(IMMUTABLE_ERR)):
            del odict["key"]

    def test_setattr(self):
        odict = ImmutableODict()
        with pytest.raises(type(IMMUTABLE_ERR), match=str(IMMUTABLE_ERR)):
            odict.meta = {"key": "value"}


class TestStateDict:
    simple_tensors = {"var1": torch.as_tensor(1.0), "var2": torch.as_tensor(2.0)}

    @pytest.fixture(scope="class")
    def simple_state_dict(self):
        return StateDict(**self.simple_tensors)

    def test_constructor(self):
        time_format = "%Y-%m-%dT%H:%M:%S"
        time_str_now = dt.utcnow().strftime(time_format)
        state_dict = StateDict(**self.simple_tensors)

        # Get the created time and format to nearest seconds
        sd_ct_dt = dt.fromisoformat(state_dict._metadata["created_time"])
        sd_ct_str = sd_ct_dt.strftime(time_format)

        # Check the default metadata and content
        assert hasattr(state_dict, "_metadata")
        assert state_dict._created is True
        assert state_dict._metadata["software_version"] == __version__
        assert sd_ct_str == time_str_now
        assert dict(state_dict) == self.simple_tensors

    def test_constructor_with_metadata(self):
        time_format = "%Y-%m-%dT%H:%M:%S"
        time_str_now = dt.utcnow().strftime(time_format)
        metadata = {"created_time": time_str_now, "software_version": "0.0.1"}
        state_dict = StateDict(metadata=metadata, **self.simple_tensors)

        assert isinstance(state_dict._metadata, ImmutableODict)
        assert dict(state_dict._metadata) == dict(metadata)

    def test_setitem(self, simple_state_dict):
        with pytest.raises(type(IMMUTABLE_ERR), match=str(IMMUTABLE_ERR)):
            simple_state_dict["var1"] = torch.as_tensor(3.0)

    def test_delitem(self, simple_state_dict):
        with pytest.raises(type(IMMUTABLE_ERR), match=str(IMMUTABLE_ERR)):
            del simple_state_dict["var1"]

    def test_from_params(self, simple_common_sim):
        params: NestedNamespaceDict = simple_common_sim.params

        tensors_dict, all_params = extract_tensors(params, True)

        expected_state_dict = StateDict(**tensors_dict)

        # Full parameters
        state_dict = StateDict.from_params(params)
        assert state_dict == expected_state_dict

        # Static only
        state_dict = StateDict.from_params(all_params)
        assert state_dict == expected_state_dict

        # Check for TypeError when passing a NamespaceDict or NestedNamespaceDict
        with pytest.raises(TypeError):
            StateDict.from_params({"a": 1, "b": 2})

        # Check for TypeError when passing a NestedNamespaceDict
        # without the "static" and "dynamic" keys
        with pytest.raises(ValueError):
            StateDict.from_params(NestedNamespaceDict({"a": 1, "b": 2}))

    def test_to_params(self):
        params_with_none = {"var3": torch.ones(0), **self.simple_tensors}
        state_dict = StateDict(**params_with_none)
        params = StateDict(**params_with_none).to_params()
        assert isinstance(params, NamespaceDict)

        for k, v in params.items():
            tensor_value = state_dict[k]
            if tensor_value.nelement() > 0:
                assert isinstance(v, Parameter)
                assert v.value == tensor_value

    def test__to_safetensors(self):
        state_dict = StateDict(**self.simple_tensors)
        # Save to safetensors
        tensors_bytes = state_dict._to_safetensors()
        expected_bytes = save(_sanitize(state_dict), metadata=state_dict._metadata)

        # Reload to back to tensors dict
        # this is done because the information
        # might be stored in different arrangements
        # within the safetensors bytes
        loaded_tensors = load(tensors_bytes)
        loaded_expected_tensors = load(expected_bytes)
        assert loaded_tensors == loaded_expected_tensors

    def test_st_file_string(self, simple_state_dict):
        file_format = "%Y%m%dT%H%M%S_caustics.st"
        expected_file = simple_state_dict._created_time.strftime(file_format)

        assert simple_state_dict._StateDict__st_file == expected_file

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="Built-in open has different behavior on Windows",
    )
    def test_save(self, simple_state_dict):
        # Check for default save path
        expected_fpath = Path.cwd() / simple_state_dict._StateDict__st_file
        default_fpath = simple_state_dict.save()

        assert Path(default_fpath).exists()
        assert default_fpath == str(expected_fpath.absolute())

        # Cleanup after
        Path(default_fpath).unlink()

        # Check for specified save path
        with TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            # Correct extension and path in a tempdir
            fpath = tempdir / "test.st"
            saved_path = simple_state_dict.save(str(fpath.absolute()))

            assert Path(saved_path).exists()
            assert saved_path == str(fpath.absolute())

            # Wrong extension
            wrong_fpath = tempdir / "test.txt"
            with pytest.raises(ValueError):
                saved_path = simple_state_dict.save(str(wrong_fpath.absolute()))

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="Built-in open has different behavior on Windows",
    )
    def test_load(self, simple_state_dict):
        fpath = simple_state_dict.save()
        loaded_state_dict = StateDict.load(fpath)
        assert loaded_state_dict == simple_state_dict

        # Cleanup after
        Path(fpath).unlink()
