from typing import Any, Optional

from torch import Tensor

from ..cosmology import Cosmology
from ..utils import derotate, translate_rotate
from .base import ThinLens

__all__ = ("SIE",)


class SIE(ThinLens):
    """
    References:
        Keeton 2001, https://arxiv.org/abs/astro-ph/0102341
    """

    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        phi: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
        s: float = 0.0,
    ):
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("b", b)
        self.s = s

    def _get_psi(self, x, y, q):
        return (q**2 * (x**2 + self.s**2) + y**2).sqrt()

    def alpha(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        z_l, thx0, thy0, q, phi, b = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = self._get_psi(thx, thy, q)
        f = (1 - q**2).sqrt()
        ax = b * q.sqrt() / f * (f * thx / (psi + self.s)).atan()
        ay = b * q.sqrt() / f * (f * thy / (psi + q**2 * self.s)).atanh()

        return derotate(ax, ay, phi)

    def Psi(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        z_l, thx0, thy0, q, phi, b = self.unpack(x)

        ax, ay = self.alpha(thx, thy, z_s, x)
        ax, ay = derotate(ax, ay, -phi)
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        return thx * ax + thy * ay

    def kappa(
        self, thx: Tensor, thy: Tensor, z_s: Tensor, x: Optional[dict[str, Any]] = None
    ) -> Tensor:
        z_l, thx0, thy0, q, phi, b = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = self._get_psi(thx, thy, q)
        return 0.5 * q.sqrt() * b / psi
