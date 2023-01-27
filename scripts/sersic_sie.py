from math import ceil, pi
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor

from caustic.base import Base
from caustic.cosmology import FlatLambdaCDMCosmology
from caustic.lenses import SIE
from caustic.sources import Sersic
from caustic.utils import get_meshgrid


class LensingSystem(Base):
    """
    Simulator consisting of a Sersic source and SIE lens at fixed redshifts. Implements
    pixelization by computing the observation on a finer grid before downsampling
    to the target resolution. Includes a PSF.
    """

    def __init__(
        self,
        res: float = 0.1,
        n_pix: int = 50,
        psf_stdev: float = 0.15,
        upsample_factor: int = 4,
        z_l: Tensor = torch.tensor(0.7),
        z_s: Tensor = torch.tensor(2.1),
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device)
        self.res = res
        self.n_pix = n_pix
        self.psf_stdev = psf_stdev
        self.upsample_factor = upsample_factor
        self.z_l = z_l
        self.z_s = z_s

        # Upsampled image grid
        res_fine = res / upsample_factor
        self.thx_fine, self.thy_fine = get_meshgrid(
            res_fine,
            upsample_factor * n_pix,
            upsample_factor * n_pix,
            device=device,
        )
        # Cosmology
        self.cosmology = FlatLambdaCDMCosmology()
        # Lens
        self.lens = SIE(device)
        # Source
        self.src = Sersic(device)
        # Set up PSF: create normalized Gaussian kernel wide enough to enclose
        # 3.5 x psf_stdev pixels on the upsampled image grid
        n_pix_psf_fine = ceil(2 * psf_stdev * 3.5 / (res / upsample_factor) + 1)
        thx_psf, thy_psf = get_meshgrid(
            res_fine, n_pix_psf_fine, n_pix_psf_fine, device=device
        )
        self.psf_kernel = (-(thx_psf**2 + thy_psf**2) / (2 * psf_stdev**2)).exp()
        self.psf_kernel /= self.psf_kernel.sum()
        # Expand to 4D
        self.psf_kernel = self.psf_kernel[None, None, :, :]

    def __call__(
        self, lens_kwargs: Dict[str, Tensor], src_kwargs: Dict[str, Tensor]
    ) -> Tensor:
        """
        Runs the simulator.
        """
        # Ray-trace
        bx, by = self.lens.raytrace(
            self.thx_fine,
            self.thy_fine,
            self.z_l,
            self.z_s,
            self.cosmology,
            **lens_kwargs
        )
        # Evaluate source
        mu = self.src.brightness(bx, by, **src_kwargs)
        # Apply PSF
        mu = F.conv2d(mu[None, None, :, :], self.psf_kernel, padding="same")
        # Downsample to target resolution through local averaging
        mu = F.avg_pool2d(mu, self.upsample_factor)[0, 0]
        return mu


def main():
    # Choose some lens and source parameters
    lens_kwargs = {
        "thx0": torch.tensor(0.22),
        "thy0": torch.tensor(-0.4),
        "q": torch.tensor(0.7),
        "phi": torch.tensor(pi / 3),
        "b": torch.tensor(1.4),
    }
    src_kwargs = {
        "thx0": torch.tensor(0.05),
        "thy0": torch.tensor(0.01),
        "phi": torch.tensor(0.8),
        "q": torch.tensor(0.5),
        "index": torch.tensor(1.5),
        "th_e": torch.tensor(0.2),
        "I_e": torch.tensor(100),
    }

    # Simulate noise-free image
    sim = LensingSystem()
    mu = sim(lens_kwargs, src_kwargs)
    # Add some Poisson noise
    image = torch.poisson(mu)

    # Plot and show observation
    fov = sim.n_pix * sim.res
    plt.imshow(
        image,
        origin="lower",
        cmap="inferno",
        extent=(-fov / 2, fov / 2, -fov / 2, fov / 2),
    )
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel(r"$x$ ['']")
    plt.ylabel(r"$y$ ['']")
    plt.show()


if __name__ == "__main__":
    main()
