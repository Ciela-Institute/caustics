from functools import wraps
from typing import Callable

def convert_params(method: Callable):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_convert_params_to_fiducial"):
            kwargs = self._convert_params_to_fiducial(*args, **kwargs)
        return method(self, *args, **kwargs)

    return wrapper
