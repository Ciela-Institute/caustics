from math import pi
from typing import Any, Optional

import torch
from torch import Tensor

from ..constants import G_over_c2, arcserc_to_rad, rad_to_arcsec
from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

DELTA = 200.0

__all__ = ("TNFW")

class TNFW(ThinLens):
	"""
	Truncated NFW lens class. This class models a lens using the Truncated Navarro-Frenk-White (TNFW) profile,
	with a truncation function (r_trunc^2)*(r^2+r_trunc^2)
	
	Attributes:
		z_l (Optional[Tensor]): Redshift of the lens. Default is None.
		
	"""

