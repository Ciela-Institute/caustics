import h5py
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as T
from caustic.lenses import KappaGrid
import torch

kap = torch.Tensor(h5py.File("../../data/data_1.h5", "r")["kappa"][0][None, None])
kap = T.CenterCrop(size=31)(kap)
kappa = KappaGrid(kap)
# alpha_x, alpha_y = kappa._fft_method()
# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# axs[0].imshow(alpha_x[0, 0], cmap="seismic")
# axs[1].imshow(alpha_y[0, 0], cmap="seismic")
alpha_x, alpha_y = kappa._fft_method()
alpha_x_true, alpha_y_true = kappa._conv2d_method()
fig, axs = plt.subplots(3, 2, figsize=(8, 12))
axs[0, 0].imshow(alpha_x[0, 0], cmap="seismic")
axs[0, 1].imshow(alpha_y[0, 0], cmap="seismic")
axs[1, 0].imshow(alpha_x_true[0, 0], cmap="seismic")
axs[1, 1].imshow(alpha_y_true[0, 0], cmap="seismic")
axs[2, 0].imshow(alpha_x_true[0, 0] - alpha_x[0, 0], cmap="seismic")
im = axs[2, 1].imshow(alpha_y_true[0, 0] - alpha_y[0, 0], cmap="seismic")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
print(np.abs(alpha_x_true[0, 0] - alpha_x[0, 0]).max())
print(np.abs(alpha_y_true[0, 0] - alpha_y[0, 0]).max())
#
plt.show()


