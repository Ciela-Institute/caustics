from typing import Dict, List, Union

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ("HDF5Dataset",)


class HDF5Dataset(Dataset):
    """
    Light-weight HDF5 dataset that reads all the data into tensors. Assumes all
    groups in dataset have the same length.
    """

    def __init__(
        self,
        path: str,
        keys: List[str],
        device: torch.device = torch.device("cpu"),
        dtypes: Union[Dict[str, torch.dtype], torch.dtype] = torch.float32,
    ):
        """
        Args:
            path: location of dataset.
            keys: dataset keys to read.
            dtypes: either a numpy datatype to which the items will be converted
                or a dictionary specifying the datatype corresponding to each key.
        """
        super().__init__()
        self.keys = keys
        self.dtypes = dtypes
        with h5py.File(path, "r") as f:
            if isinstance(dtypes, dict):
                self.data = {
                    k: torch.tensor(f[k][:], device=device, dtype=dtypes[k])
                    for k in keys
                }
            else:
                self.data = {
                    k: torch.tensor(f[k][:], device=device, dtype=dtypes) for k in keys
                }

    def __len__(self):
        return len(self.data[self.keys[0]])

    def __getitem__(self, i: Union[int, slice]) -> Dict[str, Tensor]:
        """
        Retrieves the data at index `i` for each key.
        """
        return {k: self.data[k][i] for k in self.keys}
