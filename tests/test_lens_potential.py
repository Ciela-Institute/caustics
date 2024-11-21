"""
Check for internal consistency of the lensing potential for all ThinLens objects.
"""

import torch

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
        caustics.EPL(cosmology=cosmo, z_l=z_l, **caustics.EPL._null_params),
        caustics.ExternalShear(
            cosmology=cosmo, z_l=z_l, **caustics.ExternalShear._null_params
        ),
        caustics.Multipole(cosmology=cosmo, z_l=z_l, **caustics.Multipole._null_params),
        caustics.MassSheet(cosmology=cosmo, z_l=z_l, **caustics.MassSheet._null_params),
        caustics.NFW(
            cosmology=cosmo,
            z_l=z_l,
            **caustics.NFW._null_params,
            use_case="differentiable",
        ),
        caustics.PixelatedConvergence(
            cosmology=cosmo,
            z_l=z_l,
            **caustics.PixelatedConvergence._null_params,
            pixelscale=0.1,
        ),
        caustics.PixelatedPotential(
            cosmology=cosmo,
            z_l=z_l,
            **caustics.PixelatedPotential._null_params,
            pixelscale=0.2,
        ),
        caustics.Point(cosmology=cosmo, z_l=z_l, **caustics.Point._null_params),
        caustics.PseudoJaffe(
            cosmology=cosmo, z_l=z_l, **caustics.PseudoJaffe._null_params
        ),
        caustics.SIE(cosmology=cosmo, z_l=z_l, **caustics.SIE._null_params),
        caustics.SIS(cosmology=cosmo, z_l=z_l, **caustics.SIS._null_params),
        caustics.TNFW(
            cosmology=cosmo,
            z_l=z_l,
            **caustics.TNFW._null_params,
            use_case="differentiable",
        ),
    ]

    # Define a list of lens model names.
    names = list(L.name for L in lenses)
    # Loop over the lenses.
    for lens, name in zip(lenses, names):
        print(f"Testing lens: {name}")
        lens.to(device=device)
        # Compute the deflection angle.
        ax, ay = lens.reduced_deflection_angle(x, y, z_s)

        # Ensure the x,y coordinates track gradients
        x = x.detach().requires_grad_()
        y = y.detach().requires_grad_()

        # Compute the lensing potential.
        phi = lens.potential(x, y, z_s)
        # Compute the gradient of the lensing potential.
        phi_ax, phi_ay = torch.autograd.grad(
            phi, (x, y), grad_outputs=torch.ones_like(phi)
        )

        # Check that the gradient of the lensing potential equals the deflection angle.
        if name in ["NFW", "TNFW"]:
            # Special functions in NFW and TNFW are not highly accurate, so we relax the tolerance
            assert torch.allclose(phi_ax, ax, atol=1e-3, rtol=1e-3)
            assert torch.allclose(phi_ay, ay, atol=1e-3, rtol=1e-3)
        elif name in ["PixelatedConvergence"]:
            # PixelatedConvergence potential is defined by bilinear interpolation so it is very imprecise
            assert torch.allclose(phi_ax, ax, rtol=1e0)
            assert torch.allclose(phi_ay, ay, rtol=1e0)
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
        caustics.EPL(cosmology=cosmo, z_l=z_l, **caustics.EPL._null_params),
        caustics.ExternalShear(
            cosmology=cosmo, z_l=z_l, **caustics.ExternalShear._null_params
        ),
        caustics.Multipole(cosmology=cosmo, z_l=z_l, **caustics.Multipole._null_params),
        caustics.MassSheet(cosmology=cosmo, z_l=z_l, **caustics.MassSheet._null_params),
        # caustics.NFW(
        #     cosmology=cosmo,
        #     z_l=z_l,
        #     **caustics.NFW._null_params,
        #     use_case="differentiable",
        # ), # Cannot vmap NFW when in differentiable mode
        # caustics.PixelatedConvergence(
        #     cosmology=cosmo,
        #     z_l=z_l,
        #     **caustics.PixelatedConvergence._null_params,
        #     pixelscale=0.2,
        #     n_pix=10,
        # ),  # cannot compute Hessian of PixelatedConvergence potential, always returns zeros due to bilinear interpolation
        caustics.PixelatedPotential(
            cosmology=cosmo,
            z_l=z_l,
            **caustics.PixelatedPotential._null_params,
            pixelscale=0.2,
        ),
        # caustics.Point(cosmology=cosmo, z_l=z_l, **caustics.Point._null_params), # Point mass convergence is delta function
        caustics.PseudoJaffe(
            cosmology=cosmo, z_l=z_l, **caustics.PseudoJaffe._null_params
        ),
        caustics.SIE(cosmology=cosmo, z_l=z_l, **caustics.SIE._null_params),
        caustics.SIS(cosmology=cosmo, z_l=z_l, **caustics.SIS._null_params),
        # caustics.TNFW(
        #     cosmology=cosmo, z_l=z_l, **caustics.TNFW._null_params, use_case="differentiable"
        # ), # Cannot vmap TNFW when in differentiable mode
    ]

    # Define a list of lens model names.
    names = list(L.name for L in lenses)
    # Loop over the lenses.
    for lens, name in zip(lenses, names):
        print(f"Testing lens: {name}")
        lens.to(device=device)
        # Compute the convergence.
        try:
            kappa = lens.convergence(x, y, z_s)
        except NotImplementedError:
            continue

        # Compute the laplacian of the lensing potential.
        phi_H = torch.vmap(
            torch.vmap(
                torch.func.hessian(lens.potential, (0, 1)), in_dims=(0, 0, None)
            ),
            in_dims=(0, 0, None),
        )(x, y, z_s)
        phi_kappa = 0.5 * (phi_H[0][0] + phi_H[1][1])

        # Check that the laplacian of the lensing potential equals the convergence.
        if name.strip("_0") in ["NFW", "TNFW"]:
            assert torch.allclose(phi_kappa, kappa, atol=1e-4)
        elif name.strip("_0") in ["PixelatedConvergence", "PixelatedPotential"]:
            assert torch.allclose(phi_kappa, kappa, atol=1e-4)
        else:
            assert torch.allclose(phi_kappa, kappa, atol=1e-6)
