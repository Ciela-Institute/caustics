import torch
from torch.nn.functional import avg_pool2d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from ..utils import get_meshgrid
from ..cosmology import FlatLambdaCDM
from ..lenses import SIE
from ..sources import Sersic

__all__ = ("lens_source", )

def lens_source():
    n_pix = 100
    res = 0.05
    upsample_factor = 4
    fov = res * n_pix
    thx, thy = get_meshgrid(res/upsample_factor, upsample_factor*n_pix, upsample_factor*n_pix, dtype=torch.float32)
    z_l = torch.tensor(0.5, dtype=torch.float32)
    z_s = torch.tensor(1.5, dtype=torch.float32)
    cosmology = FlatLambdaCDM(name = "cosmo")
    cosmology.to(dtype=torch.float32)

    # Sersic source, demo lensed source    
    fig, axarr = plt.subplots(2, 3, figsize = (18,9))
    axarr[0][0].axis("off")
    axarr[0][1].axis("off")
    axarr[0][2].axis("off")
    axarr[1][0].axis("off")
    axarr[1][1].axis("off")
    axarr[1][2].axis("off")

    axarr[0][0].set_title("Sersic source")
    axarr[0][1].set_title("lens log(convergence)")
    axarr[0][2].set_title("Sersic lensed")
    
    imarr = [
        axarr[0][0].imshow(np.random.rand(n_pix,n_pix), origin="lower", vmin = 0, vmax = 10),
        axarr[0][1].imshow(np.random.rand(n_pix,n_pix), origin="lower", vmin = 0, vmax = 10),
        axarr[0][2].imshow(np.random.rand(n_pix,n_pix), origin="lower", vmin = 0, vmax = 10),
    ]

    slider_vals = {
        "x0_lens": {"low": -2.5, "high": 2.5, "init": 0.},
        "y0_lens": {"low": -2.5, "high": 2.5, "init": 0.},
        "q_lens": {"low": 0.1, "high": 0.9, "init": 0.5},
        "phi_lens": {"low": 0, "high": np.pi, "init": np.pi/2},
        "b_lens": {"low": 0.1, "high": 2.5, "init": 1.},
        "x0_src": {"low": -2.5, "high": 2.5, "init": 0.},
        "y0_src": {"low": -2.5, "high": 2.5, "init": 0.},
        "q_src": {"low": 0.1, "high": 0.9, "init": 0.5},
        "phi_src": {"low": 0, "high": np.pi, "init": np.pi/2},
        "n_src": {"low": 0.36, "high": 4, "init": 2.},
        "Re_src": {"low": 0.1, "high": 3, "init": 1.},
    }

    sliders = {}
    N = len(slider_vals)
    for s, key in zip(range(N), slider_vals.keys()):
        ax_slider = plt.axes([0.25, 0.4 - s * 0.39/N, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        sliders[key] = Slider(ax_slider, key, slider_vals[key]["low"], slider_vals[key]["high"], valinit = slider_vals[key]["init"])

    def plot_lens_distortion(*args):
        lens = SIE(
            cosmology, 
            z_l, 
            torch.tensor(sliders["x0_lens"].val), 
            torch.tensor(sliders["y0_lens"].val), 
            torch.tensor(sliders["q_lens"].val),
            torch.tensor(sliders["phi_lens"].val),
            torch.tensor(sliders["b_lens"].val),
            name = "sie", 
        )
        source = Sersic(
            torch.tensor(sliders["x0_src"].val), 
            torch.tensor(sliders["y0_src"].val), 
            torch.tensor(sliders["q_src"].val), 
            torch.tensor(sliders["phi_src"].val),
            torch.tensor(sliders["n_src"].val),
            torch.tensor(sliders["Re_src"].val),
            torch.tensor(1.),
            name = "sersic",
        )
        brightness = avg_pool2d(source.brightness(thx, thy)[None, None, :, :], upsample_factor)[0,0]
        imarr[0].set_array(brightness.detach().cpu().numpy())
        imarr[0].set_clim(vmin = brightness.min().item(), vmax = brightness.max().item())
        kappa = avg_pool2d(lens.convergence(thx, thy, z_s)[None, None, :, :], upsample_factor)[0,0]
        kappa = torch.log10(kappa)
        imarr[1].set_array(kappa.detach().cpu().numpy())
        imarr[1].set_clim(vmin = kappa.min().item(), vmax = kappa.max().item())
        beta_x, beta_y = lens.raytrace(thx, thy, z_s)
        mu = avg_pool2d(source.brightness(beta_x, beta_y)[None, None, :, :], upsample_factor)[0,0]
        imarr[2].set_array(mu.detach().cpu().numpy())
        imarr[2].set_clim(vmin = mu.min().item(), vmax = mu.max().item())
        fig.canvas.draw_idle()
    
    plot_lens_distortion()

    for s, key in zip(range(N), slider_vals.keys()):
        sliders[key].on_changed(plot_lens_distortion)

    plt.show()
