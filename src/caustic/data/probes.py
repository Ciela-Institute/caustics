from typing import Union

import torch
from torch import Tensor

from .hdf5dataset import HDF5Dataset

__all__ = ("PROBESDataset",)


class PROBESDataset:
    def __init__(
        self,
        path: str,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.key = "galaxies"
        self.ds = HDF5Dataset(path, [self.key], device, dtype)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i: Union[int, slice]) -> Tensor:
        """
        Returns image `i` with channel as first dimension.
        """
        return self.ds[i][self.key].movedim(-1, 0)
