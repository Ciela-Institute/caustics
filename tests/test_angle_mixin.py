import numpy as np
from torch import pi
import torch
import caustics

import pytest


def test_angle_mixin_init():
    cosmology = caustics.FlatLambdaCDM(name="cosmo")

    # check init angle system
    lens = caustics.SIE(
        cosmology=cosmology,
        angle_system="e1_e2",
        e1=0.1,
        e2=0.2,
    )
    assert np.allclose(lens.e1.value.item(), 0.1)
    assert np.allclose(lens.e2.value.item(), 0.2)

    # check init angle system
    lens = caustics.SIE(
        cosmology=cosmology,
        angle_system="c1_c2",
        c1=0.1,
        c2=0.2,
    )
    assert np.allclose(lens.c1.value.item(), 0.1)
    assert np.allclose(lens.c2.value.item(), 0.2)

    # check init angle system
    lens = caustics.EPL(
        cosmology=cosmology,
        angle_system="e1_e2",
        e1=0.1,
        e2=0.2,
    )
    assert np.allclose(lens.e1.value.item(), 0.1)
    assert np.allclose(lens.e2.value.item(), 0.2)

    # check init angle system
    lens = caustics.EPL(
        cosmology=cosmology,
        angle_system="c1_c2",
        c1=0.1,
        c2=0.2,
    )
    assert np.allclose(lens.c1.value.item(), 0.1)
    assert np.allclose(lens.c2.value.item(), 0.2)

    # check init angle system
    lens = caustics.Sersic(
        angle_system="e1_e2",
        e1=0.1,
        e2=0.2,
    )
    assert np.allclose(lens.e1.value.item(), 0.1)
    assert np.allclose(lens.e2.value.item(), 0.2)

    # check init angle system
    lens = caustics.Sersic(
        angle_system="c1_c2",
        c1=0.1,
        c2=0.2,
    )
    assert np.allclose(lens.c1.value.item(), 0.1)
    assert np.allclose(lens.c2.value.item(), 0.2)


def test_angle_mixin():
    cosmology = caustics.FlatLambdaCDM(name="cosmo")
    lens = caustics.SIE(
        name="sie",
        cosmology=cosmology,
        z_l=0.5,
        z_s=1.0,
        x0=0.0,
        y0=0.0,
        q=0.5,
        phi=pi / 4,
        Rein=1.0,
    )

    # Check default
    assert lens.angle_system == "q_phi"

    # Check set to ellipticity coords
    lens.angle_system = "e1_e2"
    assert lens.angle_system == "e1_e2"
    assert hasattr(lens, "e1")
    assert hasattr(lens, "e2")
    # Check setting e1 e2 to get q phi
    lens.e1 = 0.3
    lens.e2 = 0.4
    assert np.allclose(lens.q.value.item(), 0.5 / 1.5)
    assert np.allclose(lens.phi.value.item(), np.arctan2(0.4, 0.3) / 2)
    q, phi = caustics.func.e1e2_to_qphi(torch.tensor(0.3), torch.tensor(0.4))
    assert np.allclose(lens.q.value.item(), q.item())
    assert np.allclose(lens.phi.value.item(), phi.item())
    e1, e2 = caustics.func.qphi_to_e1e2(q, phi)
    assert np.allclose(e1.item(), 0.3)
    assert np.allclose(e2.item(), 0.4)

    # Check reset to q_phi
    lens.angle_system = "q_phi"
    assert lens.angle_system == "q_phi"
    assert lens.q.value is None
    assert not hasattr(lens, "e1")
    assert not hasattr(lens, "e2")
    lens.angle_system = "e1_e2"
    lens.angle_system = "c1_c2"

    with pytest.raises(ValueError):
        lens.angle_system = "weird"

    # Check set to consistent ellipticity coords
    lens.angle_system = "c1_c2"
    assert lens.angle_system == "c1_c2"
    assert hasattr(lens, "c1")
    assert hasattr(lens, "c2")
    assert not hasattr(lens, "e1")
    assert not hasattr(lens, "e2")
    # Check setting e1 e2 to get q phi
    lens.c1 = 0.3
    lens.c2 = 0.4
    assert np.allclose(lens.q.value.item(), 1 - 0.5 / 1.5)
    assert np.allclose(lens.phi.value.item(), np.arctan2(0.4, 0.3) / 2)
    q, phi = caustics.func.c1c2_to_qphi(torch.tensor(0.3), torch.tensor(0.4))
    assert np.allclose(lens.q.value.item(), q.item())
    assert np.allclose(lens.phi.value.item(), phi.item())
    c1, c2 = caustics.func.qphi_to_c1c2(q, phi)
    assert np.allclose(c1.item(), 0.3)
    assert np.allclose(c2.item(), 0.4)

    # Check reset to q_phi
    lens.angle_system = "q_phi"
    assert lens.angle_system == "q_phi"
    assert lens.q.value is None
    assert not hasattr(lens, "c1")
    assert not hasattr(lens, "c2")
    lens.angle_system = "c1_c2"
    lens.angle_system = "e1_e2"
