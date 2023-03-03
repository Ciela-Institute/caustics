import torch
from torch import Tensor

from .base import ThinLens


class SinglePlane(ThinLens):
    """
    Single lens plane containing multiple thin lenses.
    """

    def __init__(
        self,
        lenses: dict[str, ThinLens],
        device=torch.device("cpu"),
        dtype=torch.float32,
    ):
        super().__init__(device, dtype)
        self.lenses = lenses

    def alpha(
        self, thx, thy, z_l, z_s, cosmology, lens_args: list[tuple[str, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        ax = torch.zeros_like(thx)
        ay = torch.zeros_like(thx)
        for name, args in lens_args:
            lens = self.lenses[name]
            ax_cur, ay_cur = lens.alpha(thx, thy, z_l, z_s, cosmology, *args)
            ax = ax + ax_cur
            ay = ay + ay_cur
        return ax, ay

    def kappa(
        self, thx, thy, z_l, z_s, cosmology, lens_args: list[tuple[str, Tensor]]
    ) -> Tensor:
        kappa = torch.zeros_like(thx)
        for name, args in lens_args:
            lens = self.lenses[name]
            kappa_cur = lens.kappa(thx, thy, z_l, z_s, cosmology, *args)
            kappa = kappa + kappa_cur
        return kappa

    def Psi(
        self, thx, thy, z_l, z_s, cosmology, lens_args: list[tuple[str, Tensor]]
    ) -> Tensor:
        Psi = torch.zeros_like(thx)
        for name, args in lens_args:
            lens = self.lenses[name]
            Psi_cur = lens.Psi(thx, thy, z_l, z_s, cosmology, *args)
            Psi = Psi + Psi_cur
        return Psi
