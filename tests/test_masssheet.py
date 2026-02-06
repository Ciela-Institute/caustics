from io import StringIO

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import MassSheet
from caustics.utils import meshgrid
from caustics.sims import build_simulator
from caustics.backend_obj import backend

import pytest


@pytest.mark.parametrize("convergence", [-1.0, 0.0, 1.0])
def test_masssheet(sim_source, device, convergence):
    z_s = backend.as_array(1.2)
    if sim_source == "yaml":
        yaml_str = f"""\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: sheet
            kind: MassSheet
            init_kwargs:
                z_s: {float(z_s)}
                cosmology: *cosmology
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = MassSheet(name="sheet", cosmology=cosmology, z_s=z_s)

    lens.to(device=device)

    # Parameters
    x = backend.as_array([0.5, 0.0, 0.0, convergence])

    thx, thy = meshgrid(0.01, 10, device=device)

    ax, ay = lens.reduced_deflection_angle(thx, thy, x)

    p = lens.potential(thx, thy, x)

    c = lens.convergence(thx, thy, x)

    assert backend.all(backend.isfinite(ax))
    assert backend.all(backend.isfinite(ay))
    assert backend.all(backend.isfinite(p))
    assert backend.all(backend.isfinite(c))
