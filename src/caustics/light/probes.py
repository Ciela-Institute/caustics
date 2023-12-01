import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ("PROBESDataset",)


# TODO: move elsewhere?
class PROBESDataset(Dataset):
    def __init__(self, filepath, channels):
        self.filepath = filepath
        self.channels = list(np.sort(np.atleast_1d(channels)))
        with h5py.File(self.filepath, "r") as hf:
            self.size = hf["galaxies"].shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        with h5py.File(self.filepath, "r") as hf:
            # Move channel first
            return torch.movedim(
                torch.tensor(
                    hf["galaxies"][index][..., self.channels], dtype=torch.float32
                ),
                -1,
                -3,
            )
