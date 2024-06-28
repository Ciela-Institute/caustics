import caustics
import torch


def test_enclosed_mass_runs(device):
    """
    Check that the enclosed mass profile runs without error.
    """
    # Define a grid of points to test.
    x, y = caustics.utils.meshgrid(0.2, 10, 10, device=device)

    cosmo = caustics.FlatLambdaCDM(name="cosmo")

    # Define the enclosed mass profile.
    enclosed_mass = caustics.EnclosedMass(
        cosmology=cosmo,
        enclosed_mass=lambda r, p: 1 - torch.exp(-r / p),
        z_l=0.5,
        **caustics.EnclosedMass._null_params,
    )
    enclosed_mass.to(device)
    # Calculate the enclosed mass profile.
    z_s = torch.tensor(1.0, device=device)
    ax, ay = enclosed_mass.reduced_deflection_angle(x, y, z_s)
    assert torch.all(torch.isfinite(ax))
    assert torch.all(torch.isfinite(ay))
    k = enclosed_mass.convergence(x, y, z_s)
    assert torch.all(torch.isfinite(k))


if __name__ == "__main__":
    test_enclosed_mass_runs(torch.device("cpu"))
