import caustics
import torch


def test_microlens_simulator_selfconsistent():
    cosmology = caustics.FlatLambdaCDM()
    sie = caustics.SIE(cosmology=cosmology, name="lens")
    src = caustics.Sersic(name="source")

    x = torch.tensor([
    #   z_s  z_l   x0   y0   q    phi     b    x0   y0   q     phi    n    Re   Ie
        1.5, 0.5, -0.2, 0.0, 0.4, 1.5708, 1.7, 0.0, 0.0, 0.5, -0.985, 1.3, 1.0, 5.0
    ])  # fmt: skip
    fov = torch.tensor((-1, -0.5, -0.25, 0.25))
    sim = caustics.Microlens(lens=sie, source=src)
    res_mcmc = sim(x, fov=fov, method="mcmc", N_mcmc=10000)
    res_grid = sim(x, fov=fov, method="grid", N_grid=100)

    # Ensure the flux estimates agree within 5sigma
    assert (res_mcmc[0] - res_grid[0]).abs().detach().cpu().numpy() < 5 * res_grid[
        1
    ].detach().cpu().numpy()
    # Ensure the uncertainties are small
    assert res_mcmc[1].abs().detach().cpu().numpy() < 1e-2
    assert res_grid[1].abs().detach().cpu().numpy() < 1e-2
