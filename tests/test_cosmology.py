from typing import List, Tuple

import numpy as np
import torch
from astropy.cosmology import Cosmology as Cosmology_AP
from astropy.cosmology import FlatLambdaCDM as AstropyFlatLambdaCDM

from caustics.cosmology import Cosmology
from caustics.cosmology import FlatLambdaCDM as CausticFlatLambdaCDM
from caustics.cosmology import Om0_default, h0_default


def get_cosmologies() -> List[Tuple[Cosmology, Cosmology_AP]]:
    """
    Gets caustics cosmologies and corresponding astropy ones.
    """
    cosmologies = []
    cosmologies.append(
        (
            CausticFlatLambdaCDM(name="cosmo"),
            AstropyFlatLambdaCDM(h0_default, Om0_default, Tcmb0=0),
        )
    )
    return cosmologies


def test_comoving_dist(device):
    rtol = 1e-3
    atol = 0

    zs = torch.linspace(0.05, 3, 10, device=device)
    for cosmology, cosmology_ap in get_cosmologies():
        cosmology.to(device=device)

        vals = cosmology.comoving_distance(zs)
        vals_ref = cosmology_ap.comoving_distance(zs.cpu().numpy()).value / 1e2  # type: ignore
        assert np.allclose(vals.cpu().numpy(), vals_ref, rtol, atol)


def test_to_method_flatlambdacdm(device):
    cosmo = CausticFlatLambdaCDM()
    # Make sure private tensors are created on float32 by default
    assert cosmo._comoving_distance_helper_x_grid.dtype == torch.float32
    assert cosmo._comoving_distance_helper_y_grid.dtype == torch.float32
    cosmo.to(dtype=torch.float64, device=device)
    # Make sure distance helper get sent to proper dtype and device
    assert cosmo._comoving_distance_helper_x_grid.dtype == torch.float64
    assert cosmo._comoving_distance_helper_y_grid.dtype == torch.float64


if __name__ == "__main__":
    test_comoving_dist(None)
