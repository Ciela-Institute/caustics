from typing import Dict, List, Union

import h5py
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from torch.utils.data import Dataset

__all__ = ("HDF5Dataset",)


class HDF5Dataset(Dataset):
    """
    Light-weight HDF5 dataset that reads all the data into RAM. Assumes all groups
    in the file being read have the same length.
    """

    def __init__(
        self,
        path: str,
        keys: List[str],
        dtypes: Union[Dict[str, DTypeLike], DTypeLike] = np.float32,
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
                self.data = {k: f[k][:].astype(dtypes[k]) for k in keys}
            else:
                self.data = {k: f[k][:].astype(dtypes) for k in keys}

    def __len__(self):
        return len(self.data[self.keys[0]])

    def __getitem__(self, i: Union[float, slice]) -> Dict[str, ArrayLike]:
        """
        Retrieves the data at index `i` for each key.
        """
        return {k: self.data[k][i] for k in self.keys}
