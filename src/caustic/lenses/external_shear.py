from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

from torch import Tensor

from ..cosmology import Cosmology
from ..utils import translate_rotate
from .base import ThinLens

__all__ = ("ExternalShear",)


class ExternalShear(ThinLens):
    def __init__(
        self,
        name: str,
        cosmology: Cosmology,
        z_l: Optional[Tensor] = None,
        thx0: Optional[Tensor] = None,
        thy0: Optional[Tensor] = None,
        gamma_1: Optional[Tensor] = None,
        gamma_2: Optional[Tensor] = None,
    ):
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("gamma_1", gamma_1)
        self.add_param("gamma_2", gamma_2)

    def alpha(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tuple[Tensor, Tensor]:
        z_l, thx0, thy0, gamma_1, gamma_2 = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        # Meneghetti eq 3.83
        a1 = thx * gamma_1 + thy * gamma_2
        a2 = thx * gamma_2 - thy * gamma_1
        return a1, a2  # I'm not sure but I think no derotation necessary

    def Psi(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tensor:
        z_l, thx0, thy0, gamma_1, gamma_2 = self.unpack(x)

        ax, ay = self.alpha(thx, thy, z_s, x)
        thx, thy = translate_rotate(thx, thy, thx0, thy0)
        return 0.5 * (thx * ax + thy * ay)

    def kappa(
        self,
        thx: Tensor,
        thy: Tensor,
        z_s: Tensor,
        x: Dict[str, Any] = defaultdict(list),
    ) -> Tensor:
        raise NotImplementedError("convergence undefined for external shear")
