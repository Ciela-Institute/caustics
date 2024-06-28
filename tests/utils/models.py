import yaml
import torch
import numpy as np

import caustics


def mock_from_file(mocker, yaml_str):
    # Mock the from_file function
    # this way, we don't need to use a real file
    mocker.patch("caustics.models.api.from_file", return_value=yaml_str.encode("utf-8"))


def obj_to_yaml(obj_dict: dict):
    yaml_string = yaml.safe_dump(obj_dict, sort_keys=False)
    string_list = yaml_string.split("\n")
    id_str = string_list[0] + f" &{string_list[0]}".strip(":")
    string_list[0] = id_str
    return "\n".join(string_list).replace("'", "")


def setup_complex_multiplane_yaml():
    # initialization stuff for lenses
    cosmology = caustics.FlatLambdaCDM(name="cosmo")
    cosmo = {
        cosmology.name: {
            "name": cosmology.name,
            "kind": cosmology.__class__.__name__,
        }
    }
    cosmology.to(dtype=torch.float32)
    n_pix = 100
    res = 0.05
    upsample_factor = 2
    fov = res * n_pix
    thx, thy = caustics.utils.meshgrid(
        res / upsample_factor,
        upsample_factor * n_pix,
        dtype=torch.float32,
    )
    z_s = torch.tensor(1.5, dtype=torch.float32)
    all_lenses = []
    all_single_planes = []

    N_planes = 10
    N_lenses = 2  # per plane

    z_plane = np.linspace(0.1, 1.0, N_planes)
    planes = []

    for p, z_p in enumerate(z_plane):
        lenses = []
        lens_keys = []

        if p == N_planes // 2:
            lens = caustics.NFW(
                cosmology=cosmology,
                z_l=z_p,
                x0=torch.tensor(0.0),
                y0=torch.tensor(0.0),
                m=torch.tensor(10**11),
                c=torch.tensor(10.0),
                s=torch.tensor(0.001),
            )
            lenses.append(lens)
            all_lenses.append(
                {
                    lens.name: {
                        "name": lens.name,
                        "kind": lens.__class__.__name__,
                        "params": {
                            k: float(v.value)
                            for k, v in lens.module_params.static.items()
                        },
                        "init_kwargs": {"cosmology": f"*{cosmology.name}"},
                    }
                }
            )
            lens_keys.append(f"*{lens.name}")
        else:
            for _ in range(N_lenses):
                lens = caustics.NFW(
                    cosmology=cosmology,
                    z_l=z_p,
                    x0=torch.tensor(np.random.uniform(-fov / 2.0, fov / 2.0)),
                    y0=torch.tensor(np.random.uniform(-fov / 2.0, fov / 2.0)),
                    m=torch.tensor(10 ** np.random.uniform(8, 9)),
                    c=torch.tensor(np.random.uniform(4, 40)),
                    s=torch.tensor(0.001),
                )
                lenses.append(lens)
                all_lenses.append(
                    {
                        lens.name: {
                            "name": lens.name,
                            "kind": lens.__class__.__name__,
                            "params": {
                                k: float(v.value)
                                for k, v in lens.module_params.static.items()
                            },
                            "init_kwargs": {"cosmology": f"*{cosmology.name}"},
                        }
                    }
                )
                lens_keys.append(f"*{lens.name}")

        single_plane = caustics.lenses.SinglePlane(
            z_l=z_p, cosmology=cosmology, lenses=lenses, name=f"plane_{p}"
        )
        planes.append(single_plane)
        all_single_planes.append(
            {
                single_plane.name: {
                    "name": single_plane.name,
                    "kind": single_plane.__class__.__name__,
                    "params": {
                        k: float(v.value)
                        for k, v in single_plane.module_params.static.items()
                    },
                    "init_kwargs": {
                        "lenses": lens_keys,
                        "cosmology": f"*{cosmology.name}",
                    },
                }
            }
        )

    lens = caustics.lenses.Multiplane(
        name="multiplane", cosmology=cosmology, lenses=planes
    )
    multi_dict = {
        lens.name: {
            "name": lens.name,
            "kind": lens.__class__.__name__,
            "init_kwargs": {
                "lenses": [f"*{p.name}" for p in planes],
                "cosmology": f"*{cosmology.name}",
            },
        }
    }
    lenses_yaml = (
        [obj_to_yaml(cosmo)]
        + [obj_to_yaml(lens) for lens in all_lenses]
        + [obj_to_yaml(plane) for plane in all_single_planes]
        + [obj_to_yaml(multi_dict)]
    )
    
    source_yaml = obj_to_yaml({
        "source": {
            "name": "source",
            "kind": "Sersic",
        }
    })
    
    lenslight_yaml = obj_to_yaml({
        "lnslight": {
            "name": "lnslight",
            "kind": "Sersic",
        }
    })
    
    sim_yaml = obj_to_yaml({
        "simulator": {
            "name": "sim",
            "kind": "LensSource",
            "init_kwargs": {
                "lens": f"*{lens.name}",
                "source": "*source",
                "lens_light": "*lnslight",
                "pixelscale": 0.05,
                "pixels_x": 100,
            }
        }
    })
    
    all_yaml_list = lenses_yaml + [source_yaml, lenslight_yaml, sim_yaml]
    return "\n".join(all_yaml_list)
