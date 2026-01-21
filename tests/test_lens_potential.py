"""
Check for internal consistency of the lensing potential for all ThinLens objects.
"""

import torch
import pytest

import caustics


def test_lens_potential_vs_deflection(device):
    """
    Check for internal consistency of the lensing potential for all ThinLens objects against the deflection angles. The gradient of the potential should equal the deflection angle.
    """
    # Define a grid of points to test.
    x, y = caustics.utils.meshgrid(0.2, 10, 10, device=device)

    # Define a source redshift.
    z_s = 1.0
    # Define a lens redshift.
    z_l = 0.5

    # Define a cosmology.
    cosmo = caustics.FlatLambdaCDM(name="cosmo")

    # Define a list of lens models.
    lenses = [
        caustics.EPL(cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.EPL._null_params),
        caustics.ExternalShear(
            cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.ExternalShear._null_params
        ),
        caustics.Multipole(
            cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.Multipole._null_params
        ),
        caustics.MassSheet(
            cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.MassSheet._null_params
        ),
        caustics.NFW(
            cosmology=cosmo,
            z_l=z_l,
            z_s=z_s,
            **caustics.NFW._null_params,
        ),
        caustics.PixelatedConvergence(
            cosmology=cosmo,
            z_l=z_l,
            z_s=z_s,
            **caustics.PixelatedConvergence._null_params,
            pixelscale=0.1,
        ),
        caustics.PixelatedPotential(
            cosmology=cosmo,
            z_l=z_l,
            z_s=z_s,
            **caustics.PixelatedPotential._null_params,
            pixelscale=0.2,
        ),
        caustics.Point(
            cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.Point._null_params
        ),
        caustics.PseudoJaffe(
            cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.PseudoJaffe._null_params
        ),
        caustics.SIE(cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.SIE._null_params),
        caustics.SIS(cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.SIS._null_params),
        caustics.TNFW(
            cosmology=cosmo,
            z_l=z_l,
            z_s=z_s,
            **caustics.TNFW._null_params,
        ),
    ]

    # Define a list of lens model names.
    names = list(L.name for L in lenses)
    # Loop over the lenses.
    for lens, name in zip(lenses, names):
        print(f"Testing lens: {name}")
        lens.to(device=device)
        # Compute the deflection angle.
        ax, ay = lens.reduced_deflection_angle(x, y)

        # Compute deflection angles using the lensing potential.
        phi_ax, phi_ay = super(lens.__class__, lens).reduced_deflection_angle(x, y)

        # Check that the gradient of the lensing potential equals the deflection angle.
        if name in ["NFW", "TNFW"]:
            # Special functions in NFW and TNFW are not highly accurate, so we relax the tolerance
            assert torch.allclose(phi_ax, ax, atol=1e-3, rtol=1e-3)
            assert torch.allclose(phi_ay, ay, atol=1e-3, rtol=1e-3)
        elif name in ["PixelatedConvergence"]:
            # PixelatedConvergence potential is defined by bilinear interpolation so it is very imprecise
            # border pixels of convergence map known to have bad derivatives due to interp to zero
            phi_ax[:, 2] = ax[:, 2]
            phi_ax[:, 7] = ax[:, 7]
            phi_ay[2] = ay[2]
            phi_ay[7] = ay[7]
            assert torch.allclose(phi_ax, ax, rtol=1e-1, atol=1e-2)
            assert torch.allclose(phi_ay, ay, rtol=1e-1, atol=1e-2)
        else:
            assert torch.allclose(phi_ax, ax, atol=1e-5)
            assert torch.allclose(phi_ay, ay, atol=1e-5)


def test_lens_potential_vs_convergence(device):
    """
    Check for internal consistency of the lensing potential for all ThinLens objects against the convergence. The laplacian of the potential should equal the convergence.
    """
    # Define a grid of points to test.
    x, y = caustics.utils.meshgrid(0.2, 10, 10, device=device)
    x, y = x.clone().detach(), y.clone().detach()

    # Define a source redshift.
    z_s = 1.0
    # Define a lens redshift.
    z_l = 0.5

    # Define a cosmology.
    cosmo = caustics.FlatLambdaCDM(name="cosmo")

    # Define a list of lens models.
    lenses = [
        caustics.EPL(cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.EPL._null_params),
        caustics.ExternalShear(
            cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.ExternalShear._null_params
        ),
        caustics.Multipole(
            cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.Multipole._null_params
        ),
        caustics.MassSheet(
            cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.MassSheet._null_params
        ),
        caustics.NFW(
            cosmology=cosmo,
            z_l=z_l,
            z_s=z_s,
            **caustics.NFW._null_params,
        ),
        # caustics.PixelatedConvergence(
        #     cosmology=cosmo,
        #     z_l=z_l,
        #     z_s=z_s,
        #     **caustics.PixelatedConvergence._null_params,
        #     pixelscale=0.2,
        #     n_pix=10,
        # ),  # cannot compute Hessian of PixelatedConvergence potential, always returns zeros due to bilinear interpolation
        caustics.PixelatedPotential(
            cosmology=cosmo,
            z_l=z_l,
            z_s=z_s,
            **caustics.PixelatedPotential._null_params,
            pixelscale=0.2,
        ),
        # caustics.Point(cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.Point._null_params), # Point mass convergence is delta function
        caustics.PseudoJaffe(
            cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.PseudoJaffe._null_params
        ),
        caustics.SIE(cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.SIE._null_params),
        caustics.SIS(cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.SIS._null_params),
        caustics.TNFW(cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.TNFW._null_params),
    ]

    # Define a list of lens model names.
    names = list(L.name for L in lenses)
    # Loop over the lenses.
    for lens, name in zip(lenses, names):
        print(f"Testing lens: {name}")
        lens.to(device=device)
        # Compute the convergence.
        try:
            kappa = lens.convergence(x, y)
        except NotImplementedError:
            continue

        # Compute the convergence from the lensing potential.
        phi_kappa = super(lens.__class__, lens).convergence(x, y)

        # Check that the laplacian of the lensing potential equals the convergence.
        if name.strip("_0") in ["NFW", "TNFW"]:
            print(torch.abs(phi_kappa - kappa) / kappa)
            assert torch.allclose(phi_kappa, kappa, rtol=1e-3, atol=1e-3)
        elif name.strip("_0") in ["PixelatedConvergence", "PixelatedPotential"]:
            assert torch.allclose(phi_kappa, kappa, rtol=1e-4, atol=1e-4)
        else:
            assert torch.allclose(phi_kappa, kappa, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("chunk_size", [10, 100])
def test_base_deflection_chunks(chunk_size, device):
    """
    Test the chunked and unchunked computation of reduced deflection angles for
    a lens model and ensure they produce consistent results.

    Parameters:
        chunk_size: Size of chunks for the iterative solver.
    """
    x, y = caustics.utils.meshgrid(0.2, 10, 10, device=device)

    # Define a source redshift.
    z_s = 1.0
    # Define a lens redshift.
    z_l = 0.5

    # Define a cosmology.
    cosmo = caustics.FlatLambdaCDM(name="cosmo")

    # Define a lens model.
    lens = caustics.EPL(cosmology=cosmo, z_l=z_l, z_s=z_s, **caustics.EPL._null_params)
    # compute reduced deflections angle with and without chunks
    phi_ax, phi_ay = super(lens.__class__, lens).reduced_deflection_angle(x, y)
    phi_ax_chunked, phi_ay_chunked = super(
        lens.__class__, lens
    ).reduced_deflection_angle(x, y, chunk_size)

    assert torch.allclose(phi_ax, phi_ax_chunked)
    assert torch.allclose(phi_ay, phi_ay_chunked)
