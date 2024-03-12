from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
from ..parametrized import Parametrized


class Parameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class InitKwargs(Parameters):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Base(BaseModel):
    name: str = Field(..., description="Name of the object")
    kind: str = Field(..., description="Kind of the object")
    params: Optional[Parameters] = Field(None, description="Parameters of the object")
    init_kwargs: Optional[InitKwargs] = Field(
        None, description="Initiation keyword arguments for object creation"
    )

    # internal
    _cls: Any

    def __init__(self, **data):
        super().__init__(**data)

    def _get_init_kwargs_dump(self, init_kwargs: InitKwargs) -> Dict[str, Any]:
        """
        Get the model dump of the class parameters,
        if the field is a model then get the model object.

        Parameters
        ----------
        init_kwargs : ClassParams
            The class parameters to dump

        Returns
        -------
        dict
            The model dump of the class parameters
        """
        model_dict = {}
        for f in init_kwargs.model_fields_set:
            model = getattr(init_kwargs, f)
            if isinstance(model, Base):
                model_dict[f] = model.model_obj()
            elif isinstance(model, list):
                model_dict[f] = [m.model_obj() for m in model]
            else:
                model_dict[f] = getattr(init_kwargs, f)
        return model_dict

    @classmethod
    def _set_class(cls, parametrized_cls: Parametrized) -> type["Base"]:
        """
        Set the class of the object.

        Parameters
        ----------
        cls : Parametrized
            The class to set.
        """
        cls._cls = parametrized_cls
        return cls

    def model_obj(self) -> Any:
        if not self._cls:
            raise ValueError(
                "The class is not set. Please set the class before calling this method."
            )
        init_kwargs = (
            self._get_init_kwargs_dump(self.init_kwargs) if self.init_kwargs else {}
        )  # Capture None case
        params = self.params.model_dump() if self.params else {}  # Capture None case
        return self._cls(name=self.name, **init_kwargs, **params)


class FileInput(BaseModel):
    path: str = Field(..., description="The path to the file")


class StateDict(BaseModel):
    load: FileInput


class StateConfig(BaseModel):
    state: Optional[StateDict] = Field(
        None, description="State safetensor for the simulator"
    )
