import torch
from torch import vmap
from utils import setup_image_simulator, setup_simulator


def test_vmapped_simulator():
    sim, (sim_params, cosmo_params, lens_params, source_params) = setup_simulator(
        batched_params=True
    )
    n_pix = sim.n_pix
    print(sim.params)

    # test list input
    x = sim_params + cosmo_params + lens_params + source_params
    print(x[0].shape)
    assert vmap(sim)(x).shape == torch.Size([2, n_pix, n_pix])

    # test tensor input
    x_tensor = torch.stack(x, dim=1)
    print(x_tensor.shape)
    assert vmap(sim)(x_tensor).shape == torch.Size([2, n_pix, n_pix])

    # Test dictionary input: Only module with dynamic parameters are required
    x_dict = {
        "simulator": sim_params,
        "cosmo": cosmo_params,
        "source": source_params,
        "lens": lens_params,
    }
    print(x_dict)
    assert vmap(sim)(x_dict).shape == torch.Size([2, n_pix, n_pix])

    # Test semantic list (one tensor per module)
    x_semantic = [sim_params, cosmo_params, lens_params, source_params]
    assert vmap(sim)(x_semantic).shape == torch.Size([2, n_pix, n_pix])


def test_vmapped_simulator_with_pixelated_modules():
    sim, (cosmo_params, lens_params, kappa, source) = setup_image_simulator(
        batched_params=True
    )
    n_pix = sim.n_pix
    print(sim.params)

    # test list input
    x = cosmo_params + lens_params + kappa + source
    print(x[2].shape)
    assert vmap(sim)(x).shape == torch.Size([2, n_pix, n_pix])

    # test tensor input: Does not work well with images since it would require
    # unflattening the images in caustics
    # x_tensor = torch.concat([_x.view(2, -1) for _x in x], dim=1)
    # print(x_tensor.shape)
    # assert vmap(sim)(x_tensor).shape == torch.Size([2, n_pix, n_pix])

    # Test dictionary input: Only module with dynamic parameters are required
    x_dict = {
        "cosmo": cosmo_params,
        "source": source,
        "lens": lens_params,
        "kappa": kappa,
    }
    print(x_dict)
    assert vmap(sim)(x_dict).shape == torch.Size([2, n_pix, n_pix])

    # Test passing tensor in source and kappa instead of list
    x_dict = {
        "cosmo": cosmo_params,
        "source": source[0],
        "lens": lens_params,
        "kappa": kappa[0],
    }
    print(x_dict)
    assert vmap(sim)(x_dict).shape == torch.Size([2, n_pix, n_pix])

    # Test semantic list (one tensor per module)
    x_semantic = [cosmo_params, lens_params, kappa, source]
    assert vmap(sim)(x_semantic).shape == torch.Size([2, n_pix, n_pix])
