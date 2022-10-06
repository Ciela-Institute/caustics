from typing import Callable, List

import torch
from torch import Tensor, device

from .base import Base
from .cosmology import AbstractCosmology, FlatLambdaCDMCosmology
from .lenses import AbstractLens
from .sources import AbstractSource


class LensingSystem(Base):
    def __init__(
        self,
        lenses: List[AbstractLens],
        z_lenses,
        sources: List[AbstractSource],
        z_sources,
        instrument: Callable[[Tensor], Tensor] = lambda x: x,
        multiplane_cls=None,
        cosmology: AbstractCosmology = FlatLambdaCDMCosmology(),
        device: device = torch.device("cpu"),
    ):
        super().__init__(cosmology, device)
        # TODO: how should we handle upsampling for pixelation?
        # Sort sources and lenses by redshift
        self.z_lenses, lens_sort_idxs = z_lenses.sort()
        self.lenses = [lenses[i] for i in lens_sort_idxs]
        self.z_sources, source_sort_idxs = z_sources.sort()
        self.sources = [sources[i] for i in source_sort_idxs]
        self.instrument = instrument

        if len(lenses) > 1:
            if multiplane_cls is not None:
                self.multiplane_handler = multiplane_cls(
                    self.lenses, self.z_lenses, cosmology, device
                )
            else:
                raise ValueError(
                    "if there are multiple lenses, a multiplane lensing class must "
                    "be specified"
                )

    def brightness(self, thx, thy, t=None, w=None):
        # TODO: handle multiple sources
        if self.multiplane_handler is not None:
            thx_src, thy_src = self.multiplane_handler.raytrace(
                thx, thy, self.z_sources
            )
            return self.sources[0].brightness(thx_src, thy_src)
        else:
            thx_src, thy_src = self.lenses[0].raytrace(
                thx, thy, self.z_lenses[0], self.z_sources
            )
            return self.sources[0].brightness(thx_src, thy_src)
