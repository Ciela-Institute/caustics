from typing import Dict, Annotated, Optional
from torch import Tensor

from ..parametrized import Parametrized
from .state_dict import StateDict
from ..namespace_dict import NestedNamespaceDict

__all__ = ("Simulator",)

NameType = Annotated[Optional[str], "Name of the simulator"]


class Simulator(Parametrized):
    """A caustics simulator using Parametrized framework.

    Defines a simulator class which is a callable function that
    operates on the Parametrized framework. Users define the `forward`
    method which takes as its first argument an object which can be
    packed, all other args and kwargs are simply passed to the forward
    method.

    See `Parametrized` for details on how to add/access parameters.

    """

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            packed_args = self.pack(args[0])
            rest_args = args[1:]
        else:
            packed_args = self.pack()
            rest_args = tuple()

        return self.forward(packed_args, *rest_args, **kwargs)

    @staticmethod
    def __set_module_params(module: Parametrized, params: Dict[str, Tensor]):
        for k, v in params.items():
            setattr(module, k, v)

    def state_dict(self) -> StateDict:
        return StateDict.from_params(self.params)

    def load_state_dict(self, file_path: str) -> "Simulator":
        """
        Loads and then sets the state of the simulator from a file

        Parameters
        ----------
        file_path : str | Path
            The file path to a safetensors file
            to load the state from

        Returns
        -------
        Simulator
            The simulator with the loaded state
        """
        loaded_state_dict = StateDict.load(file_path)
        self.set_state_dict(loaded_state_dict)
        return self

    def set_state_dict(self, state_dict: StateDict) -> "Simulator":
        """
        Sets the state of the simulator from a state dict

        Parameters
        ----------
        state_dict : StateDict
            The state dict to load from

        Returns
        -------
        Simulator
            The simulator with the loaded state
        """
        # TODO: Do some checks for the state dict metadata

        # Convert to nested namespace dict
        param_dicts = NestedNamespaceDict(state_dict)

        # Grab params for the current module
        self_params = param_dicts.pop(self.name)

        def _set_params(module):
            # Start from root, and move down the DAG
            if module.name in param_dicts:
                module_params = param_dicts[module.name]
                self.__set_module_params(module, module_params)
            if module._childs != {}:
                for child in module._childs.values():
                    _set_params(child)

        # Set the parameters of the current module
        self.__set_module_params(self, self_params)

        # Set the parameters of the children modules
        _set_params(self)
        return self
