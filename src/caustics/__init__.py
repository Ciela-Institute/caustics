from ._version import version as VERSION  # noqa

from . import constants, lenses, cosmology, packed, parametrized, light, utils, sims
from .tests import test

# from .demo import *

__version__ = VERSION
__author__ = "Ciela"

__all__ = [
    # Modules
    "constants",
    "lenses",
    "cosmology",
    "packed",
    "parametrized",
    "light",
    "utils",
    "sims",
    # Functions
    "test",
]
