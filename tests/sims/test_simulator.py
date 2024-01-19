import pytest

from caustics.sims.state_dict import StateDict
from helpers.sims import extract_tensors, isEquals


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
        assert isEquals(dict(state_dict), expected_tensors)
