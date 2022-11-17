from abc import ABC

import torch


class Base(ABC):
    def __init__(self, device: torch.device):
        self.device = device
