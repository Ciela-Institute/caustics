import numpy as np
import caustics

lens_slider_configs = {
    "EPL": [
        ["x0", "EPL X Position", [-2.0, 2.0, 0.0]],
        ["y0", "EPL Y Position", [-2.0, 2.0, 0.25]],
        ["q", "EPL Axis Ratio", [0.1, 1.0, 0.82]],
        ["phi", "EPL Rotation Angle", [0.0, 3.14, 8 * (3.14 / 180) + 3.14 / 2]],
        ["theta_E", "EPL Einstein Radius", [0.0, 2.0, 1.606]],
        ["t", "EPL Power Law Slope", [0.0, 2.0, 1.0]],
    ],
    "Shear": [
        [
            "gamma1",
            "Shear First Component",
            [-1.0, 1.0, 0.036 * np.cos(2 * 3 * (np.pi / 180))],
        ],
        [
            "gamma2",
            "Shear Second Component",
            [-1.0, 1.0, 0.036 * np.sin(2 * 3 * (np.pi / 180))],
        ],
    ],
    "Mass Sheet": [
        ["kappa", "Mass Sheet Convergence", [0.0, 1.0, 0.5]],
    ],
    "NFW": [
        ["x0", "NFW X Position", [-2.0, 2.0, 0.0]],
        ["y0", "NFW Y Position", [-2.0, 2.0, 0.25]],
        ["m", "NFW Mass with R200", [1e12, 1e15, 1e13]],
        ["c", "NFW Concentration", [5.0, 40.0, 20.0]],
    ],
    "SIS": [
        ["x0", "SIS X Position", [-2.0, 2.0, 0.0]],
        ["y0", "SIS Y Position", [-2.0, 2.0, 0.25]],
        ["theta_E", "SIS Einstein Radius", [0.0, 2.0, 1.606]],
    ],
    # "Pixelated Convergence": [
    #     ["x0", "Pix Conv X Position", [-2.0, 2.0, 0.0]],
    #     ["y0", "Pix Conv Y Position", [-2.0, 2.0, 0.25]],
    # ],
    "Point": [
        ["x0", "Point X Position", [-2.0, 2.0, 0.0]],
        ["y0", "Point Y Position", [-2.0, 2.0, 0.25]],
        ["th_ein", "Point Einstein Radius", [0.0, 2.0, 1.606]],
    ],
    "Pseudo-Jaffe": [
        ["x0", "P-Jaffe X Position", [-2.0, 2.0, 0.0]],
        ["y0", "P-Jaffe Y Position", [-2.0, 2.0, 0.25]],
        ["mass", "P-Jaffe Mass", [1e9, 1e11, 1e10]],
        ["core_radius", "P-Jaffe Core Radius", [0.0, 2.0, 0.5]],
        ["scale_radius", "P-Jaffe Scale Radius", [2.0, 10.0, 5.0]],
    ],
    "SIE": [
        ["x0", "SIE X Position", [-2.0, 2.0, 0.0]],
        ["y0", "SIE Y Position", [-2.0, 2.0, 0.25]],
        ["q", "SIE Axis Ratio", [0.1, 1.0, 0.82]],
        ["phi", "SIE Rotation Angle", [0.0, 3.14, 8 * (3.14 / 180) + 3.14 / 2]],
        ["b", "SIE Einstein Radius", [0.0, 2.0, 1.606]],
    ],
    "TNFW": [
        ["x0", "TNFW X Position", [-2.0, 2.0, 0.0]],
        ["y0", "TNFW Y Position", [-2.0, 2.0, 0.25]],
        ["m", "TNFW Mass with R200", [1e9, 1e11, 1e10]],
        ["c", "TNFW Concentration", [5.0, 40.0, 2.0]],
    ],
}

source_slider_configs = {
    "Sersic": [
        ["x0", "Sersic X Position", [-2.0, 2.0, 0.0]],
        ["y0", "Sersic Y Position", [-2.0, 2.0, -0.2 + 0.25]],
        ["q", "Sersic Axis Ratio", [0.1, 1.0, 0.5]],
        ["phi", "Sersic Rotation Angle", [0.0, 3.14, 0.0]],
        ["n", "Sersic Index", [0.1, 10.0, 0.8]],
        ["Re", "Sersic Scale Length", [0.0, 2.0, 1.25]],
    ],
    # "Pixelated": [
    #     ["x0", "Pix X Position", [-2.0, 2.0, 0.0]],
    #     ["y0", "Pix Y Position", [-2.0, 2.0, -0.2 + 0.25]],
    # ],
}

name_map = {
    "EPL": caustics.EPL,
    "Shear": caustics.ExternalShear,
    "Mass Sheet": caustics.MassSheet,
    "NFW": caustics.NFW,
    "SIS": caustics.SIS,
    # "Pixelated Convergence": caustics.PixelatedConvergence,
    "Point": caustics.Point,
    "Pseudo-Jaffe": caustics.PseudoJaffe,
    "SIE": caustics.SIE,
    "TNFW": caustics.TNFW,
    "Sersic": caustics.Sersic,
    # "Pixelated": caustics.Pixelated,
}

default_params = {
    "EPL": {},
    "Shear": {"x0": 0.0, "y0": 0.0},
    "Mass Sheet": {"x0": 0.0, "y0": 0.0},
    "NFW": {},
    "SIS": {},
    # "Pixelated Convergence": {},
    "Point": {},
    "Pseudo-Jaffe": {},
    "SIE": {},
    "TNFW": {},
    "Sersic": {"Ie": 10.0},
    # "Pixelated": {},
}
