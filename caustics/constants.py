from math import pi

from astropy.constants.codata2018 import G as _G_astropy
from astropy.constants.codata2018 import c as _c_astropy

__all__ = ("rad_to_arcsec", "arcsec_to_rad", "c_km_s", "G", "G_over_c2", "c_Mpc_s", "km_to_Mpc")

rad_to_arcsec = 180 / pi * 60 ** 2
arcsec_to_rad = 1 / rad_to_arcsec
c_km_s = float(_c_astropy.to("km/s").value)
G = float(_G_astropy.to("pc * km^2 / (s^2 * solMass)").value)
G_over_c2 = float((_G_astropy / _c_astropy ** 2).to("Mpc/solMass").value)  # type: ignore
c_Mpc_s = float(_c_astropy.to("Mpc/s").value)
km_to_Mpc = 3.2407792896664e-20  # TODO: use astropy
