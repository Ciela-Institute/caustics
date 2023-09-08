# from math import pi

# import lenstronomy.Util.param_util as param_util
import torch
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_AP
from astropy.cosmology import default_cosmology

# next three imports to get Rs_angle and alpha_Rs in arcsec for lenstronomy
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustic.cosmology import FlatLambdaCDM as CausticFlatLambdaCDM
from caustic.lenses import TNFW

h0_default = float(default_cosmology.get().h)
Om0_default = float(default_cosmology.get().Om0)
Ob0_default = float(default_cosmology.get().Ob0)