from abc import ABC


class Base(ABC):
    def __init__(self, cosmology, device):
        self.cosmology = cosmology
        self.device = device
