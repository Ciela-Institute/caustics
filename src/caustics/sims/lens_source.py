from copy import copy

from scipy.fft import next_fast_len
from torch.nn.functional import avg_pool2d, conv2d
from typing import Optional
import torch

from .simulator import Simulator

__all__ = ("Lens_Source",)


class Lens_Source(Simulator):
    """Lens image of a source.

    Striaghtforward simulator to sample a lensed image of a source
    object. Constructs a sampling grid internally based on the
    pixelscale and gridding parameters. It can automatically upscale
    and fine sample an image. This is the most straightforward
    simulator to view the image if you aready have a lens and source
    chosen.

    Example usage::

       import matplotlib.pyplot as plt
       import caustics

       cosmo = caustics.FlatLambdaCDM()
       lens = caustics.lenses.SIS(cosmology = cosmo, x0 = 0., y0 = 0., th_ein = 1.)
       source = caustics.sources.Sersic(x0 = 0., y0 = 0., q = 0.5, phi = 0.4, n = 2., Re = 1., Ie = 1.)
       sim = caustics.sims.Lens_Source(lens, source, pixelscale = 0.05, gridx = 100, gridy = 100, upsample_factor = 2, z_s = 1.)

       img = sim()
       plt.imshow(img, origin = "lower")
       plt.show()

    Attributes:
      lens: caustics lens mass model object
      source: caustics light object which defines the background source
      pixelscale: pixelscale of the sampling grid.
      pixels_x: number of pixels on the x-axis for the sampling grid
      lens_light (optional): caustics light object which defines the lensing object's light
      psf (optional): An image to convolve with the scene. Note that if ``upsample_factor > 1`` the psf must also be at the higher resolution.
      pixels_y (optional): number of pixels on the y-axis for the sampling grid. If left as ``None`` then this will simply be equal to ``gridx``
      upsample_factor (default 1): Amount of upsampling to model the image. For example ``upsample_factor = 2`` indicates that the image will be sampled at double the resolution then summed back to the original resolution (given by pixelscale and gridx/y).
      psf_pad (default True): If convolving the PSF it is important to sample the model in a larger FOV equal to half the PSF size in order to account for light that scatters from outside the requested FOV inwards. Internally this padding will be added before sampling, then cropped off before returning the final image to the user.
      z_s (optional): redshift of the source
      name (default "sim"): a name for this simulator in the parameter DAG.

    """

    def __init__(
        self,
        lens,
        source,
        pixelscale: float,
        pixels_x: int,
        lens_light=None,
        psf=None,
        pixels_y: Optional[int] = None,
        upsample_factor: int = 1,
        psf_pad=True,
        psf_mode="fft",
        z_s=None,
        name: str = "sim",
    ):
        super().__init__(name)

        # Lensing models
        self.lens = lens
        self.source = source
        self.lens_light = lens_light
        if psf is None:
            self.psf = None
        else:
            self.psf = torch.as_tensor(psf)
            self.psf /= psf.sum()  # ensure normalized
        self.add_param("z_s", z_s)

        # Image grid
        if pixels_y is None:
            pixels_y = pixels_x
        self.gridding = (pixels_x, pixels_y)

        # PSF padding if needed
        self.psf_mode = psf_mode
        if psf_pad and self.psf is not None:
            self.psf_pad = (self.psf.shape[1] // 2 + 1, self.psf.shape[0] // 2 + 1)
        else:
            self.psf_pad = (0, 0)

        # Build the imaging grid
        self.upsample_factor = upsample_factor
        self.n_pix = (
            self.gridding[0] + self.psf_pad[0] * 2,
            self.gridding[1] + self.psf_pad[1] * 2,
        )
        tx = torch.linspace(
            -0.5 * (pixelscale * self.n_pix[0]),
            0.5 * (pixelscale * self.n_pix[0]),
            self.n_pix[0] * upsample_factor,
        )
        ty = torch.linspace(
            -0.5 * (pixelscale * self.n_pix[1]),
            0.5 * (pixelscale * self.n_pix[1]),
            self.n_pix[1] * upsample_factor,
        )
        self.grid = torch.meshgrid(tx, ty, indexing="xy")

        if self.psf is not None:
            self.psf_fft = self._fft2_padded(self.psf)

    def _fft2_padded(self, x):
        """
        Compute the 2D Fast Fourier Transform (FFT) of a tensor with zero-padding.

        Args:
            x (Tensor): The input tensor to be transformed.

        Returns:
            Tensor: The 2D FFT of the input tensor with zero-padding.
        """
        npix = copy(self.n_pix)
        npix = (next_fast_len(npix[0]), next_fast_len(npix[1]))
        self._s = npix

        return torch.fft.rfft2(x, self._s)

    def _unpad_fft(self, x):
        """
        Remove padding from the result of a 2D FFT.

        Args:
            x (Tensor): The input tensor with padding.

        Returns:
            Tensor: The input tensor without padding.
        """
        return torch.roll(x, (-self.psf_pad[0], -self.psf_pad[1]), dims=(-2, -1))[
            ..., : self.n_pix[0], : self.n_pix[1]
        ]

    def forward(
        self,
        params,
        source_light=True,
        lens_light=True,
        lens_source=True,
        psf_convolve=True,
    ):
        """
        params: Packed object
        source_light: when true the source light will be sampled
        lens_light: when true the lens light will be sampled
        lens_source: when true, the source light model will be lensed by the lens mass distribution
        psf_convolve: when true the image will be convolved with the psf
        """
        (z_s,) = self.unpack(params)

        # Sample the source light
        if source_light:
            if lens_source:
                # Source is lensed by the lens mass distribution
                bx, by = self.lens.raytrace(*self.grid, z_s, params)
                mu = self.source.brightness(bx, by, params)
            else:
                # Source is imaged without lensing
                mu = self.source.brightness(*self.grid, params)
        else:
            # Source is not added to the scene
            mu = torch.zeros_like(self.grid[0])

        # Sample the lens light
        if lens_light and self.lens_light is not None:
            mu += self.lens_light.brightness(*self.grid, params)

        # Convolve the PSF
        if psf_convolve and self.psf is not None:
            if self.psf_mode == "fft":
                mu_fft = self._fft2_padded(mu)
                mu = self._unpad_fft(
                    torch.fft.irfft2(mu_fft * self.psf_fft, self._s).real
                )
            elif self.psf_mode == "conv2d":
                mu = conv2d(
                    mu[None, None], self.psf[None, None], padding="same"
                ).squeeze()
            else:
                raise ValueError(
                    f"psf_mode should be one of 'fft' or 'conv2d', not {self.psf_mode}"
                )

        # Return to the desired image
        mu_native_resolution = avg_pool2d(
            mu[None, None], self.upsample_factor, divisor_override=1
        ).squeeze()
        mu_clipped = mu_native_resolution[
            self.psf_pad[1] : self.gridding[1] + self.psf_pad[1],
            self.psf_pad[0] : self.gridding[0] + self.psf_pad[0],
        ]
        return mu_clipped
