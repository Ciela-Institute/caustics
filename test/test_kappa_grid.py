import torch

from caustic.cosmology import FlatLambdaCDMCosmology
from caustic.lenses import KappaGrid, PseudoJaffe
from caustic.utils import get_meshgrid


def _setup(n_pix, mode):
    fov = 25.0
    res = fov / n_pix
    thx, thy = get_meshgrid(res, n_pix, n_pix)

    cosmology = FlatLambdaCDMCosmology()
    z_l = torch.tensor(0.5)
    z_s = torch.tensor(2.1)

    # Use PseudoJaffe since it is compact: 99.16% of its mass is contained in
    # the circle circumscribing this image plane
    lens = PseudoJaffe()
    thx0 = torch.tensor(0.0)
    thy0 = torch.tensor(0.0)
    r_core = torch.tensor(0.04)
    r_s = torch.tensor(0.2)
    kappa_0 = lens.kappa_0(z_l, z_s, cosmology, torch.tensor(1.0), r_core, r_s)

    kappa_lens = KappaGrid(fov, n_pix, mode=mode)
    kappa_map = lens.kappa(
        thx, thy, z_l, z_s, cosmology, thx0, thy0, kappa_0, r_core, r_s
    )
    # Shape required by KappaGrid
    kappa_map = kappa_map[None, None, :, :]

    Psi = lens.Psi(thx, thy, z_l, z_s, cosmology, thx0, thy0, kappa_0, r_core, r_s)
    Psi -= Psi.min()

    Psi_approx = kappa_lens.Psi(thx, thy, None, None, None, kappa_map)
    Psi_approx = Psi_approx[0, 0]
    Psi_approx -= Psi_approx.min()
    # Try to remove unobservable constant offset
    Psi_approx += torch.mean(Psi - Psi_approx)

    alpha_x, alpha_y = lens.alpha(
        thx, thy, z_l, z_s, cosmology, thx0, thy0, kappa_0, r_core, r_s
    )

    alpha_x_approx, alpha_y_approx = kappa_lens.alpha(
        thx, thy, None, None, None, kappa_map
    )
    alpha_x_approx = alpha_x_approx[0, 0]
    alpha_y_approx = alpha_y_approx[0, 0]

    return Psi, Psi_approx, alpha_x, alpha_x_approx, alpha_y, alpha_y_approx


def test_Psi_alpha():
    """
    Tests whether KappaGrid is fairly accurate using a large image.
    """
    Psi, Psi_approx, alpha_x, alpha_x_approx, alpha_y, alpha_y_approx = _setup(
        1000, "fft"
    )
    _check_center(Psi, Psi_approx, atol=1e-21)
    _check_center(alpha_x, alpha_x_approx, atol=8e-20)
    _check_center(alpha_y, alpha_y_approx, atol=1e-20)


def test_consistency():
    """
    Checks whether using fft and conv2d give the same results.
    """
    for n_pix in [3, 4, 100]:
        _, Psi_fft, _, alpha_x_fft, _, alpha_y_fft = _setup(n_pix, "fft")
        _, Psi_conv2d, _, alpha_x_conv2d, _, alpha_y_conv2d = _setup(n_pix, "conv2d")
        assert torch.allclose(Psi_fft, Psi_conv2d, atol=1e-20, rtol=0)
        assert torch.allclose(alpha_x_fft, alpha_x_conv2d, atol=1e-20, rtol=0)
        assert torch.allclose(alpha_y_fft, alpha_y_conv2d, atol=1e-20, rtol=0)


def _check_center(x, x_approx, rtol=0.00001, atol=1e-8, half_buffer=20):
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
