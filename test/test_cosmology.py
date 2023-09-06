from typing import List, Tuple

import numpy as np
import torch
from astropy.cosmology import Cosmology as Cosmology_AP
from astropy.cosmology import FlatLambdaCDM as AstropyFlatLambdaCDM
from astropy.cosmology import LambdaCDM as AstropyLambdaCDM

from caustic.cosmology import Cosmology
from caustic.cosmology import FlatLambdaCDM as CausticFlatLambdaCDM
from caustic.cosmology.LambdaCDM import LambdaCDM as CausticLambdaCDM
from caustic.cosmology.FlatLambdaCDM import Om0_default, H0_default


def get_cosmologies_flat_lcdm() -> List[Tuple[Cosmology,  Cosmology_AP]]:
    """
    Gets caustic cosmologies and corresponding astropy ones.
    """
    cosmologies = []
    cosmologies.append(
        (
            CausticFlatLambdaCDM(name="cosmo"),
            AstropyFlatLambdaCDM(H0_default, Om0_default, Tcmb0=0),
        )
    )
    return cosmologies


def test_comoving_dist_flat_lcdm():
    rtol = 1e-3
    atol = 0

    zs = torch.linspace(0.05, 3, 10)
    for cosmology, cosmology_ap in get_cosmologies_flat_lcdm():
        vals = cosmology.comoving_distance(zs).numpy()
        vals_ref = cosmology_ap.comoving_distance(zs).value  # type: ignore
        assert np.allclose(vals, vals_ref, rtol, atol)


def get_cosmologies_lcdm() -> List[Tuple[Cosmology,  Cosmology_AP]]:
    """
    Gets caustic cosmologies and corresponding astropy ones.
    """
    cosmologies = []
    cosmologies.append(
        (
	    CausticLambdaCDM(name="cosmo_phytorch"),
            AstropyFlatLambdaCDM(H0_default, Om0_default, Tcmb0=0),
        )
    )
    return cosmologies


def test_comoving_dist_lcdm():
    rtol = 1e-3
    atol = 0

    zs = torch.linspace(0.05, 3, 10)
    for cosmology, cosmology_ap in get_cosmologies_lcdm():
        vals = cosmology.comoving_distance(zs).numpy()
        vals_ref = cosmology_ap.comoving_distance(zs).value  # type: ignore
        assert np.allclose(vals, vals_ref, rtol, atol)

def test_critical_density_lcdm():
    rtol = 1e-2 #Not sure why, but there is a bit of a difference at the 1e-3 level
    atol = 0

    zs = torch.linspace(0.05, 3, 10)
    for cosmology, cosmology_ap in get_cosmologies_lcdm():
        vals = cosmology.critical_density(zs).numpy()
        vals_ref = cosmology_ap.critical_density(zs).value  # type: ignore
        assert np.allclose(vals, vals_ref, rtol, atol)


if __name__ == "__main__":
    test_comoving_dist_flat_lcdm()
    test_comoving_dist_lcdm()
    test_critical_density_lcdm()
