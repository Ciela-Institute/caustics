import torch
import pytest

from caustics.utils import (
    meshgrid,
    gnomonic_plane_to_world,
    gnomonic_world_to_plane,
    pixel_to_world,
    world_to_pixel,
    pixel_to_world_sip,
    world_to_pixel_sip,
)
from caustics.constants import arcsec_to_deg


@pytest.mark.parametrize("crval", [(0.0, 0.0), (10.0, 90.0), (180.0, -45.0)])
def test_gnomonic_projection(crval):

    px, py = meshgrid(pixelscale=0.1 * arcsec_to_deg, nx=100, dtype=torch.float64)

    ra, dec = gnomonic_plane_to_world(
        px, py, crval=torch.tensor(crval, dtype=torch.float64)
    )
    px2, py2 = gnomonic_world_to_plane(
        ra, dec, crval=torch.tensor(crval, dtype=torch.float64)
    )

    assert torch.allclose(px, px2, atol=1e-5)
    assert torch.allclose(py, py2, atol=1e-5)


@pytest.mark.parametrize("crpix", [(0.0, 0.0), (50.0, 50.0), (-10.0, -100.0)])
@pytest.mark.parametrize("crval", [(0.0, 0.0), (10.0, 89.9), (180.0, -45.0)])
@pytest.mark.parametrize(
    "CD",
    [((1e-4, 0.0), (0.0, 1e-4)), ((-9.72e-6, -1.03e-5), (-9.63e-6, 8.70e-6))],
)
def test_linear_wcs(crpix, crval, CD):
    crpix = torch.tensor(crpix, dtype=torch.float64)
    crval = torch.tensor(crval, dtype=torch.float64)
    CD = torch.tensor(CD, dtype=torch.float64)

    px, py = torch.meshgrid(
        torch.arange(0, 100, dtype=torch.float64),
        torch.arange(0, 100, dtype=torch.float64),
        indexing="ij",
    )

    ra, dec = pixel_to_world(px, py, crpix=crpix, crval=crval, CD=CD)
    px2, py2 = world_to_pixel(ra, dec, crpix=crpix, crval=crval, CD=CD)

    assert torch.allclose(px, px2, atol=1e-5)
    assert torch.allclose(py, py2, atol=1e-5)


@pytest.mark.parametrize("crpix", [(0.0, 0.0), (50.0, 50.0), (-10.0, -100.0)])
@pytest.mark.parametrize("crval", [(0.0, 0.0), (10.0, 89.9), (180.0, -45.0)])
@pytest.mark.parametrize(
    "CD",
    [((1e-4, 0.0), (0.0, 1e-4)), ((-9.72e-6, -1.03e-5), (-9.63e-6, 8.70e-6))],
)
def test_sip_wcs(crpix, crval, CD):
    crpix = torch.tensor(crpix, dtype=torch.float64)
    crval = torch.tensor(crval, dtype=torch.float64)
    CD = torch.tensor(CD, dtype=torch.float64)
    # SIP coefficients from a random HST image
    sip_powers = torch.tensor(
        (
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 1),
            (5, 0),
        ),
        dtype=torch.float64,
    )
    sip_coefs = torch.tensor(
        (
            (2.26e-6, -9.78e-6),
            (8.90e-11, -3.71e-10),
            (1.40e-13, -1.60e-13),
            (-1.65e-17, -1.64e-17),
            (-7.53e-6, 6.45e-6),
            (-5.04e-10, -2.88e-11),
            (-1.19e-14, 9.53e-15),
            (6.30e-18, -1.76e-17),
            (8.52e-6, -2.96e-6),
            (-1.12e-10, -4.00e-10),
            (9.68e-14, -1.14e-13),
            (-5.05e-18, -1.18e-17),
            (-4.89e-10, 1.17e-10),
            (3.77e-14, -4.20e-14),
            (-9.51e-18, 2.45e-18),
            (1.97e-14, 1.90e-14),
            (2.42e-18, -8.86e-18),
            (4.85e-18, -6.06e-18),
        ),
        dtype=torch.float64,
    )

    px, py = torch.meshgrid(
        torch.arange(0, 100, dtype=torch.float64),
        torch.arange(0, 100, dtype=torch.float64),
        indexing="ij",
    )

    ra, dec = pixel_to_world_sip(
        px, py, crpix=crpix, crval=crval, CD=CD, powers=sip_powers, coefs=sip_coefs
    )
    px2, py2 = world_to_pixel_sip(
        ra, dec, crpix=crpix, crval=crval, CD=CD, powers=sip_powers, coefs=-sip_coefs
    )

    assert torch.allclose(px, px2, atol=1e-3)
    assert torch.allclose(py, py2, atol=1e-3)
