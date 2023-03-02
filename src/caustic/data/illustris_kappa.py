from typing import Union

import torch
from torch import Tensor

from .hdf5dataset import HDF5Dataset

__all__ = ("IllustrisKappaDataset",)


class IllustrisKappaDataset:
    def __init__(
        self,
        path: str,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.key = "kappa"
        self.ds = HDF5Dataset(path, [self.key], device, dtype)
        assert self[0].shape[-1] == self[0].shape[-2]
        self.n_pix = self[0].shape[-1]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i: Union[int, slice]) -> Tensor:
        return self.ds[i][self.key]
