import numpy as np
import os
from glob import glob
import pandas as pd
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid


def display(arr, ax=None, lim=1, mid=0, title=None, fs=12, norm="Centered", cmap="bwr", cbar=True, axis=False):
    show = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        show = True

    norm_kw = {}
    if norm == "Centered":
        norm_kw.update({"norm": colors.CenteredNorm(vcenter=mid)})
    elif norm == "Log":
        norm_kw.update({"norm": colors.LogNorm()})

    im = ax.imshow(arr, origin='lower', extent=(-lim, lim, -lim, lim),
                   cmap=cmap, **norm_kw)

    if cbar:
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    tx = None
    if title is not None:
        tx = ax.set_title(title, fontsize=fs)

    if not axis:
        ax.axis("off")

    if show:
        plt.show()
        plt.close()

    return im, tx

