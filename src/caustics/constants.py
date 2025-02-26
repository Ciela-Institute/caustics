from math import pi

from astropy.constants.codata2018 import G as _G_astropy
from astropy.constants.codata2018 import c as _c_astropy

__all__ = (
    "rad_to_arcsec",
    "arcsec_to_rad",
    "rad_to_deg",
    "deg_to_rad",
    "c_km_s",
    "G",
    "G_over_c2",
    "c_Mpc_s",
    "km_to_Mpc",
)

# fmt: off
rad_to_arcsec = 180 / pi * 60**2
arcsec_to_rad = 1 / rad_to_arcsec
rad_to_deg = 180 / pi
deg_to_rad = 1 / rad_to_deg
arcsec_to_deg = 1 / 60**2
deg_to_arcsec = 60**2
c_km_s = float(_c_astropy.to("km/s").value)
G = float(_G_astropy.to("pc * km^2 / (s^2 * solMass)").value)
G_over_c2 = float((_G_astropy / _c_astropy**2).to("Mpc/solMass").value)  # type: ignore
c_Mpc_s = float(_c_astropy.to("Mpc/s").value)
km_to_Mpc = 3.2407792896664e-20  # TODO: use astropy
days_to_seconds = 24.0 * 60.0 * 60.0
# fmt: on
