from math import pi

import lenstronomy.Util.param_util as param_util
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM_ap
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper
import numpy as np

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Multiplane, PixelatedConvergence, SinglePlane
from caustics.utils import meshgrid
from caustics.backend_obj import backend

import pytest


def test(device):
    rtol = 0
    atol = 5e-3

    # Setup
    z_s = backend.as_array(1.5, dtype=backend.float32)

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = backend.as_array(
        [p for _xs in xs for p in _xs], dtype=backend.float32, device=device
    )

    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=backend.float32, device=device)
    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[
            SIE(name=f"sie_{i}", cosmology=cosmology, z_s=z_s) for i in range(len(xs))
        ],
        z_s=z_s,
    )

    # lenstronomy
    kwargs_ls = []
    for _xs in xs:
        e1, e2 = param_util.phi_q2_ellipticity(phi=_xs[4], q=_xs[3])
        kwargs_ls.append(
            {
                "theta_E": _xs[5],
                "e1": e1,
                "e2": e2,
                "center_x": _xs[1],
                "center_y": _xs[2],
            }
        )

    # Use same cosmology
    cosmo_ap = FlatLambdaCDM_ap(
        backend.to_numpy(cosmology.h0.value),
        backend.to_numpy(cosmology.Om0.value),
        Tcmb0=0,
    )
    lens_ls = LensModel(
        lens_model_list=["SIE" for _ in range(len(xs))],
        z_source=z_s.item(),
        lens_redshift_list=[_xs[0] for _xs in xs],
        cosmo=cosmo_ap,
        multi_plane=True,
    )

    with pytest.warns(UserWarning):
        lens_test_helper(
            lens,
            lens_ls,
            x,
            kwargs_ls,
            rtol,
            atol,
            test_Psi=False,
            test_kappa=False,
            device=device,
        )


def test_multiplane_time_delay(device):
    # Setup
    z_s = backend.as_array(1.5, dtype=backend.float32, device=device)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=backend.float32, device=device)

    n_pix = 10
    res = 0.05
    upsample_factor = 2
    thx, thy = meshgrid(
        res / upsample_factor,
        upsample_factor * n_pix,
        dtype=backend.float32,
        device=device,
    )

    # Parameters
    xs = [
        [0.5, 0.9, -0.4, 0.9999, 3 * pi / 4, 0.8],
        [0.7, 0.0, 0.5, 0.9999, -pi / 6, 0.7],
        [1.1, 0.4, 0.3, 0.9999, pi / 4, 0.9],
    ]
    x = backend.as_array(
        [p for _xs in xs for p in _xs], dtype=backend.float32, device=device
    )

    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[SIE(name=f"sie_{i}", cosmology=cosmology) for i in range(len(xs))],
        z_s=z_s,
    )
    lens.to(device=device)

    assert backend.all(backend.isfinite(lens.time_delay(thx, thy, x)))
    assert backend.all(
        backend.isfinite(
            lens.time_delay(
                thx,
                thy,
                x,
                geometric_time_delay=True,
                shapiro_time_delay=False,
            )
        )
    )
    assert backend.all(
        backend.isfinite(
            lens.time_delay(
                thx,
                thy,
                x,
                geometric_time_delay=False,
                shapiro_time_delay=True,
            )
        )
    )


@pytest.mark.parametrize(
    "shapiro_time_delay,geometric_time_delay",
    [(True, True), (True, False), (False, True)],
)
def test_single_plane_time_delay_equivalence(
    device, shapiro_time_delay, geometric_time_delay
):
    z_s = backend.as_array(1.5, dtype=backend.float32, device=device)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=backend.float32, device=device)
    plane = SIE(
        name="sie",
        cosmology=cosmology,
        z_l=0.5,
        x0=0.1,
        y0=-0.2,
        q=0.8,
        phi=0.3,
        Rein=0.9,
    )
    lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=[plane],
        z_s=z_s,
    )
    lens.to(dtype=backend.float32, device=device)
    thx, thy = meshgrid(0.05, 10, dtype=backend.float32, device=device)

    expected_bx, expected_by = plane.raytrace(thx, thy)
    actual_bx, actual_by = lens.raytrace(thx, thy)
    assert backend.allclose(actual_bx, expected_bx, rtol=1e-4, atol=1e-4)
    assert backend.allclose(actual_by, expected_by, rtol=1e-4, atol=1e-4)

    expected = plane.time_delay(
        thx,
        thy,
        shapiro_time_delay=shapiro_time_delay,
        geometric_time_delay=geometric_time_delay,
    )
    actual = lens.time_delay(
        thx,
        thy,
        shapiro_time_delay=shapiro_time_delay,
        geometric_time_delay=geometric_time_delay,
    )

    assert backend.allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shapiro_time_delay,geometric_time_delay",
    [(True, True), (True, False), (False, True)],
)
def test_coplanar_lenses_time_delay(
    device, shapiro_time_delay, geometric_time_delay
):
    z_l = 0.5
    z_s = backend.as_array(1.5, dtype=backend.float32, device=device)
    cosmology = FlatLambdaCDM(name="cosmo")
    cosmology.to(dtype=backend.float32, device=device)

    def make_lenses(prefix, lens_redshift):
        return [
            SIE(
                name=f"{prefix}_{i}",
                cosmology=cosmology,
                z_l=lens_redshift,
                x0=x0,
                y0=y0,
                q=q,
                phi=phi,
                Rein=Rein,
            )
            for i, (x0, y0, q, phi, Rein) in enumerate(
                [(0.1, -0.2, 0.8, 0.3, 0.9), (-0.2, 0.1, 0.7, -0.4, 0.6)]
            )
        ]

    expected_lens = SinglePlane(
        name="singleplane",
        cosmology=cosmology,
        lenses=make_lenses("single_sie", None),
        z_l=z_l,
        z_s=z_s,
    )
    actual_lens = Multiplane(
        name="multiplane",
        cosmology=cosmology,
        lenses=make_lenses("multi_sie", z_l),
        z_s=z_s,
    )
    expected_lens.to(dtype=backend.float32, device=device)
    actual_lens.to(dtype=backend.float32, device=device)
    thx, thy = meshgrid(0.05, 10, dtype=backend.float32, device=device)

    expected_bx, expected_by = expected_lens.raytrace(thx, thy)
    actual_bx, actual_by = actual_lens.raytrace(thx, thy)
    assert backend.allclose(actual_bx, expected_bx, rtol=1e-4, atol=1e-4)
    assert backend.allclose(actual_by, expected_by, rtol=1e-4, atol=1e-4)

    expected = expected_lens.time_delay(
        thx,
        thy,
        shapiro_time_delay=shapiro_time_delay,
        geometric_time_delay=geometric_time_delay,
    )
    actual = actual_lens.time_delay(
        thx,
        thy,
        shapiro_time_delay=shapiro_time_delay,
        geometric_time_delay=geometric_time_delay,
    )

    assert backend.all(backend.isfinite(actual))
    assert backend.allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_params(device):
    z_s = 1
    n_planes = 10
    cosmology = FlatLambdaCDM()
    pixel_size = 0.04
    pixels = 16
    z = np.linspace(1e-2, 1, n_planes)
    planes = []
    for p in range(n_planes):
        lens = PixelatedConvergence(
            name=f"plane_{p}",
            pixelscale=pixel_size,
            cosmology=cosmology,
            z_l=z[p],
            x0=0.0,
            y0=0.0,
            shape=(pixels, pixels),
            padding="tile",
        )
        lens.to(device=device)
        planes.append(lens)
    multiplane_lens = Multiplane(cosmology=cosmology, lenses=planes, z_s=z_s)
    multiplane_lens.to(device=device)
    z_s = backend.as_array(z_s)
    x, y = meshgrid(pixel_size, 32, device=device)
    params = [backend.randn(pixels, pixels, device=device) for i in range(10)]

    # Test out the computation of a few quantities to make sure params are passed correctly

    # First case, params as list of tensors
    kappa_eff = multiplane_lens.effective_convergence_div(x, y, params)
    assert kappa_eff.shape == backend.Size([32, 32])
    alphax, alphay = multiplane_lens.effective_reduced_deflection_angle(x, y, params)
    assert alphax.shape == backend.Size([32, 32])
    assert alphay.shape == backend.Size([32, 32])

    # Second case, params given as a kwargs
    kappa_eff = multiplane_lens.effective_convergence_div(x, y, params=params)
    assert kappa_eff.shape == backend.Size([32, 32])
    alphax, alphay = multiplane_lens.effective_reduced_deflection_angle(
        x, y, params=params
    )
    assert alphax.shape == backend.Size([32, 32])
    assert alphay.shape == backend.Size([32, 32])

    # Test that we can pass a dictionary
    params = {
        "lenses": {
            f"plane_{p}": [backend.randn(pixels, pixels, device=device)]
            for p in range(n_planes)
        }
    }

    kappa_eff = multiplane_lens.effective_convergence_div(x, y, params)
    assert kappa_eff.shape == backend.Size([32, 32])
    alphax, alphay = multiplane_lens.effective_reduced_deflection_angle(x, y, params)


if __name__ == "__main__":
    test(None)
