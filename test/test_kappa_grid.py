import torch

from caustic.cosmology import FlatLambdaCDMCosmology
from caustic.lenses import KappaGrid, PseudoJaffe
from caustic.utils import get_meshgrid


def _setup(n_pix, mode, use_next_fast_len):
    fov = 25.0
    res = fov / n_pix
    thx, thy = get_meshgrid(res, n_pix, n_pix)

    z_l = torch.tensor(0.5)
    z_s = torch.tensor(2.1)

    cosmology = FlatLambdaCDMCosmology("cosmology")
    # Use PseudoJaffe since it is compact: 99.16% of its mass is contained in
    # the circle circumscribing this image plane
    lens_pj = PseudoJaffe("pf", cosmology)

    thx0 = 0.0
    thy0 = 0.0
    th_core = 0.04
    th_s = 0.2
    rho_0 = 1.0
    kappa_0 = lens_pj.kappa_0(z_l, z_s, rho_0, th_core, th_s, cosmology)
    # z_l, thx0, thy0, kappa_0, th_core, th_s
    x_pj = torch.tensor([z_l, thx0, thy0, kappa_0, th_core, th_s])

    # Exact calculations
    Psi = lens_pj.Psi(thx, thy, z_l, lens_pj.x_to_dict(x_pj))
    Psi -= Psi.min()
    alpha_x, alpha_y = lens_pj.alpha(thx, thy, z_l, lens_pj.x_to_dict(x_pj))

    # Approximate calculations
    lens_kap = KappaGrid(
        "kg",
        fov,
        n_pix,
        cosmology,
        z_l=z_l,
        kappa_map_shape=(1, 1, n_pix, n_pix),
        mode=mode,
        use_next_fast_len=use_next_fast_len,
    )
    kappa_map = lens_pj.kappa(thx, thy, z_l, lens_pj.x_to_dict(x_pj))
    x_kap = kappa_map.flatten()

    Psi_approx = lens_kap.Psi(thx, thy, z_l, lens_kap.x_to_dict(x_kap))
    Psi_approx = Psi_approx[0, 0]
    Psi_approx -= Psi_approx.min()
    # Try to remove unobservable constant offset
    Psi_approx += torch.mean(Psi - Psi_approx)

    alpha_x_approx, alpha_y_approx = lens_kap.alpha(
        thx, thy, z_l, lens_kap.x_to_dict(x_kap)
    )
    alpha_x_approx = alpha_x_approx[0, 0]
    alpha_y_approx = alpha_y_approx[0, 0]

    return Psi, Psi_approx, alpha_x, alpha_x_approx, alpha_y, alpha_y_approx


def test_Psi_alpha():
    """
    Tests whether KappaGrid is fairly accurate using a large image.
    """
    for use_next_fast_len in [True, False]:
        Psi, Psi_approx, alpha_x, alpha_x_approx, alpha_y, alpha_y_approx = _setup(
            1000, "fft", use_next_fast_len
        )
        _check_center(Psi, Psi_approx, atol=1e-21)
        _check_center(alpha_x, alpha_x_approx, atol=1e-19)
        _check_center(alpha_y, alpha_y_approx, atol=1e-20)


def test_consistency():
    """
    Checks whether using fft and conv2d give the same results.
    """
    for n_pix in [3, 4, 100]:
        for use_next_fast_len in [True, False]:
            _, Psi_fft, _, alpha_x_fft, _, alpha_y_fft = _setup(
                n_pix, "fft", use_next_fast_len
            )
            _, Psi_conv2d, _, alpha_x_conv2d, _, alpha_y_conv2d = _setup(
                n_pix, "conv2d", use_next_fast_len
            )
            assert torch.allclose(Psi_fft, Psi_conv2d, atol=1e-20, rtol=0)
            assert torch.allclose(alpha_x_fft, alpha_x_conv2d, atol=1e-20, rtol=0)
            assert torch.allclose(alpha_y_fft, alpha_y_conv2d, atol=1e-20, rtol=0)


def _check_center(x, x_approx, rtol=1e-5, atol=1e-8, half_buffer=20):
    n_pix = x.shape[-1]
    idx_before = n_pix // 2 - half_buffer
    idx_after = n_pix // 2 + half_buffer
    assert torch.allclose(x[:idx_before], x_approx[:idx_before], rtol, atol)
    assert torch.allclose(x[idx_after:], x_approx[idx_after:], rtol, atol)
    assert torch.allclose(
        x[idx_before:idx_after, :idx_before],
        x_approx[idx_before:idx_after, :idx_before],
        rtol,
        atol,
    )
    assert torch.allclose(
        x[idx_before:idx_after, idx_after:],
        x_approx[idx_before:idx_after, idx_after:],
        rtol,
        atol,
    )


if __name__ == "__main__":
    test_consistency()
