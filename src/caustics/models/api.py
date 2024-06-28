# mypy: disable-error-code="import-untyped"
import yaml
from pathlib import Path
from typing import Union

from ..sims.simulator import Simulator
from ..io import from_file


def build_simulator(config_path: Union[str, Path]) -> Simulator:
    """
    Build a simulator from the configuration
    """
    # Imports using Pydantic are placed here to make Pydantic a weak dependency
    from caustics.models.utils import setup_simulator_models, create_model, Field
    from caustics.models.base_models import StateConfig

    simulators = setup_simulator_models()
    Config = create_model(
        "Config", __base__=StateConfig, simulator=(simulators, Field(...))
    )

    # Load the yaml config
    yaml_bytes = from_file(config_path)
    config_dict = yaml.safe_load(yaml_bytes)
    # Create config model
    config = Config(**config_dict)

    # Get the simulator
    sim = config.simulator.model_obj()

    # Load state if available
    simulator_state = config.state
    if simulator_state is not None:
        sim.load_state_dict(simulator_state.load.path)

    return sim
