import matplotlib.pyplot as plt
import torch
from icecream import ic

from caustic.cosmology import FlatLambdaCDMCosmology
from caustic.lenses import SIS
from caustic.lenses.kappa_grid import KappaGrid
from caustic.utils import get_meshgrid


def test_consistency():
    # TODO: make sure FFT and analytic calculations agree
    ...


# device = "cpu"
# dtype = torch.float64
# kappa_mode = "conv2d"
#
# cosmology = FlatLambdaCDMCosmology()
# res = 0.05
# n_pix = 100
# fov = res * n_pix
# thx, thy = get_meshgrid(res, n_pix, n_pix, device=device, dtype=dtype)
#
# z_l = torch.tensor(1.0, device=device, dtype=dtype)
# z_s = torch.tensor(2.5, device=device, dtype=dtype)
# thx0 = torch.tensor(0.0, device=device, dtype=dtype)
# thy0 = torch.tensor(0.0, device=device, dtype=dtype)
# th_ein = torch.tensor(1.5, device=device, dtype=dtype)
# sis = SIS(thx0, thy0, th_ein, z_l, cosmology, device)
#
# kappa_sis = KappaGrid(
#     sis.kappa(thx, thy, z_s)[None, None, :, :],
#     fov,
#     kappa_mode,
#     dtype,
#     z_l,
#     cosmology,
#     device,
# )
#
# ic(kappa_sis.xconv_kernel.max())
# ic(sis.kappa(thx, thy, z_s).max())
#
# fig, axes = plt.subplots(2, 3, figsize=(12, 8))
#
# ax = axes[0, 0]
# im = ax.imshow(sis.kappa(thx, thy, z_s).log10().squeeze().cpu(), origin="lower")
# plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# ax.set_title(r"$\log_{10} \kappa$")
# alpha_x, alpha_y = sis.alpha(thx, thy, z_s)
# ax = axes[0, 1]
# im = ax.imshow(alpha_x.squeeze().cpu(), origin="lower", cmap="bwr")
# plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# ax.set_title(r"$\alpha_x$")
# ax = axes[0, 2]
# im = ax.imshow(alpha_y.squeeze().cpu(), origin="lower", cmap="bwr")
# plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# ax.set_title(r"$\alpha_y$")
#
# ax = axes[1, 0]
# im = ax.imshow(kappa_sis._kappa.log().squeeze().cpu(), origin="lower")
# plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# alpha_x, alpha_y = kappa_sis._method()
# ax = axes[1, 1]
# im = ax.imshow(alpha_x.squeeze().cpu(), origin="lower", cmap="bwr")
# plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# ax = axes[1, 2]
# im = ax.imshow(alpha_y.squeeze().cpu(), origin="lower", cmap="bwr")
# plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#
# fig.tight_layout()
# fig.suptitle(f"{n_pix} pixels, {kappa_mode}, {dtype}")
# plt.savefig(
#     f"/Users/amcoogan/Downloads/figures/{n_pix}_pixels_{kappa_mode}_{dtype}.png"
# )
