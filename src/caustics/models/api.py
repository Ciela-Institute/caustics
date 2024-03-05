# mypy: disable-error-code="import-untyped"
import yaml
from pathlib import Path
from typing import Union

from ..sims.simulator import Simulator
from ..io import from_file
from .utils import setup_simulator_models, create_model, Field
from .base_models import StateConfig


def build_simulator(config_path: Union[str, Path]) -> Simulator:
    """
    Build a simulator from the configuration
    """
    simulators = setup_simulator_models()
    Config = create_model(
        "Config", __base__=StateConfig, simulator=(simulators, Field(...))
    )

    # Load the yaml config
    yaml_bytes = from_file(config_path)
    config_json = yaml.safe_load(yaml_bytes)
    # Create config model
    config = Config(**config_json)

    # Get the simulator
    sim = config.simulator.model_obj()

    # Load state if available
    simulator_state = config.state
    if simulator_state is not None:
        sim.load_state_dict(simulator_state.load.path)

    return sim
