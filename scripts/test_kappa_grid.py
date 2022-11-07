import matplotlib.pyplot as plt
from icecream import ic
import torch

from caustic.cosmology import FlatLambdaCDMCosmology
from caustic.lenses import SIS
from caustic.lenses.kappa_grid import KappaGrid
from caustic.utils import get_meshgrid

device = "cpu"
dtype = torch.float64
kappa_mode = "fft"

cosmology = FlatLambdaCDMCosmology()
res = 0.05
nx = ny = 100
fov = res * nx
thx, thy = get_meshgrid(res, nx, ny, device=device, dtype=dtype)

z_l = torch.tensor(1.0, device=device, dtype=dtype)
z_s = torch.tensor(2.5, device=device, dtype=dtype)
thx0 = torch.tensor(0.0, device=device, dtype=dtype)
thy0 = torch.tensor(0.0, device=device, dtype=dtype)
th_ein = torch.tensor(1.5, device=device, dtype=dtype)
sis = SIS(thx0, thy0, th_ein, z_l, cosmology, device)

kappa_sis = KappaGrid(
    sis.kappa(thx, thy, z_s)[None, None, :, :],
    0.0,
    0.0,
    fov,
    kappa_mode,
    dtype,
    z_l,
    cosmology,
    device,
)

ic(kappa_sis.xconv_kernel.max())
ic(sis.kappa(thx, thy, z_s).max())

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

ax = axes[0, 0]
im = ax.imshow(sis.kappa(thx, thy, z_s).log10().squeeze().cpu(), origin="lower")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(r"$\log_{10} \kappa$")
alpha_x, alpha_y = sis.alpha(thx, thy, z_s)
ax = axes[0, 1]
im = ax.imshow(alpha_x.squeeze().cpu(), origin="lower")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(r"$\alpha_x$")
ax = axes[0, 2]
im = ax.imshow(alpha_y.squeeze().cpu(), origin="lower")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(r"$\alpha_y$")

ax = axes[1, 0]
im = ax.imshow(kappa_sis._kappa.log().squeeze().cpu(), origin="lower")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
alpha_x, alpha_y = kappa_sis._method()
ax = axes[1, 1]
im = ax.imshow(alpha_x.squeeze().cpu(), origin="lower")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax = axes[1, 2]
im = ax.imshow(alpha_y.squeeze().cpu(), origin="lower")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
fig.suptitle(f"{kappa_mode} {dtype}")
plt.show()
