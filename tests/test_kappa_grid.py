import torch

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import PixelatedConvergence, PseudoJaffe
from caustics.utils import get_meshgrid


def _setup(n_pix, mode, use_next_fast_len, padding="zero"):
    # TODO understand why this test fails for resolutions != 0.025
    res = 0.025
    thx, thy = get_meshgrid(res, n_pix, n_pix)

    z_l = torch.tensor(0.5)
    z_s = torch.tensor(2.1)

    cosmology = FlatLambdaCDM(name="cosmology")
    # Use PseudoJaffe since it is compact: 99.16% of its mass is contained in
    # the circle circumscribing this image plane
    lens_pj = PseudoJaffe(name="pj", cosmology=cosmology)

    thx0 = torch.tensor(7.0)
    thy0 = torch.tensor(3.0)
    th_core = torch.tensor(0.04)
    th_s = torch.tensor(0.2)
    rho_0 = torch.tensor(1.0)

    d_l = cosmology.angular_diameter_distance(z_l)
    arcsec_to_rad = 1 / (180 / torch.pi * 60**2)

    kappa_0 = lens_pj.central_convergence(
        z_l,
        z_s,
        rho_0,
        th_core * d_l * arcsec_to_rad,
        th_s * d_l * arcsec_to_rad,
        cosmology.critical_surface_density(z_l, z_s),
    )
    # z_l, thx0, thy0, kappa_0, th_core, th_s
    x_pj = torch.tensor([z_l, thx0, thy0, kappa_0, th_core, th_s])

    # Exact calculations
    Psi = lens_pj.potential(thx, thy, z_l, lens_pj.pack(x_pj))
    Psi -= Psi.min()
    alpha_x, alpha_y = lens_pj.reduced_deflection_angle(
        thx, thy, z_l, lens_pj.pack(x_pj)
    )

    # Approximate calculations
    lens_kap = PixelatedConvergence(
        res,
        n_pix,
        cosmology,
        z_l=z_l,
        shape=(n_pix, n_pix),
        convolution_mode=mode,
        use_next_fast_len=use_next_fast_len,
        name="kg",
        padding=padding,
    )
    kappa_map = lens_pj.convergence(thx, thy, z_l, lens_pj.pack(x_pj))
    x_kap = kappa_map.flatten()

    Psi_approx = lens_kap.potential(thx, thy, z_l, lens_kap.pack(x_kap))
    Psi_approx -= Psi_approx.min()
    # Try to remove unobservable constant offset
    Psi_approx += torch.mean(Psi - Psi_approx)

    alpha_x_approx, alpha_y_approx = lens_kap.reduced_deflection_angle(
        thx, thy, z_l, lens_kap.pack(x_kap)
    )

    return Psi, Psi_approx, alpha_x, alpha_x_approx, alpha_y, alpha_y_approx


def test_Psi_alpha():
    """
    Tests whether PixelatedConvergence is fairly accurate using a large image.
    """
    for use_next_fast_len in [True, False]:
        Psi, Psi_approx, alpha_x, alpha_x_approx, alpha_y, alpha_y_approx = _setup(
            1000, "fft", use_next_fast_len
        )
        _check_center(Psi, Psi_approx, 780, 620, atol=1e-20)
        _check_center(alpha_x, alpha_x_approx, 780, 620, atol=1e-20)
        _check_center(alpha_y, alpha_y_approx, 780, 620, atol=1e-20)


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


def test_padoptions():
    """
    Checks whether using fft and conv2d give the same results.
    """
    _, Psi_fft_circ, _, alpha_x_fft_circ, _, alpha_y_fft_circ = _setup(
        100,
        "fft",
        True,
        "circular",
    )
    _, Psi_fft_tile, _, alpha_x_fft_tile, _, alpha_y_fft_tile = _setup(
        100,
        "fft",
        True,
        "tile",
    )
    assert torch.allclose(Psi_fft_circ, Psi_fft_tile, atol=1e-20, rtol=0)
    assert torch.allclose(alpha_x_fft_circ, alpha_x_fft_tile, atol=1e-20, rtol=0)
    assert torch.allclose(alpha_y_fft_circ, alpha_y_fft_tile, atol=1e-20, rtol=0)


def _check_center(
    x, x_approx, center_c, center_r, rtol=1e-5, atol=1e-8, half_buffer=20
):
    idx_before_r = center_r - half_buffer
    idx_after_r = center_r + half_buffer
    idx_before_c = center_c - half_buffer
    idx_after_c = center_c + half_buffer
    assert torch.allclose(x[:idx_before_r], x_approx[:idx_before_r], rtol, atol)
    assert torch.allclose(x[idx_after_r:], x_approx[idx_after_r:], rtol, atol)
    assert torch.allclose(
        x[idx_before_r:idx_after_r, :idx_before_c],
        x_approx[idx_before_r:idx_after_r, :idx_before_c],
        rtol,
        atol,
    )
    assert torch.allclose(
        x[idx_before_r:idx_after_r, idx_after_c:],
        x_approx[idx_before_r:idx_after_r, idx_after_c:],
        rtol,
        atol,
    )
