import caustics
from caustics.backend_obj import backend


def test_microlens_simulator_selfconsistent():
    cosmology = caustics.FlatLambdaCDM()
    sie = caustics.SIE(cosmology=cosmology, name="lens")
    src = caustics.Sersic(name="source")

    x = backend.as_array([
        #   z_s  z_l   x0   y0   q    phi     b    x0   y0   q     phi    n    Re   Ie
        1.5, 0.5, -0.2, 0.0, 0.4, 1.5708, 1.7, 0.0, 0.0, 0.5, -0.985, 1.3, 1.0, 5.0
    ])  # fmt: skip
    fov = backend.as_array((-1, -0.5, -0.25, 0.25))
    sim = caustics.Microlens(lens=sie, source=src)
    res_mcmc = sim(x, fov=fov, method="mcmc", N_mcmc=10000)
    res_grid = sim(x, fov=fov, method="grid", N_grid=100)

    # Ensure the flux estimates agree within 5sigma
    assert backend.to_numpy(
        backend.abs(res_mcmc[0] - res_grid[0])
    ) < 5 * backend.to_numpy(res_grid[1])
    # Ensure the uncertainties are small
    assert backend.to_numpy(backend.abs(res_mcmc[1])) < 1e-2
    assert backend.to_numpy(backend.abs(res_grid[1])) < 1e-2
