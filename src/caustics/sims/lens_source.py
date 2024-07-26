from copy import copy

from scipy.fft import next_fast_len
from torch.nn.functional import avg_pool2d, conv2d
from typing import Optional, Annotated, Literal, Union
import torch
from torch import Tensor

from .simulator import Simulator, NameType
from ..utils import (
    meshgrid,
    gaussian_quadrature_grid,
    gaussian_quadrature_integrator,
)
from ..lenses.base import Lens
from ..light.base import Source


__all__ = ("LensSource",)


class LensSource(Simulator):
    """Lens image of a source.

    Straightforward simulator to sample a lensed image of a source object.
    Constructs a sampling grid internally based on the pixelscale and gridding
    parameters. It can automatically upscale and fine sample an image. This is
    the most straightforward simulator to view the image if you already have a
    lens and source chosen.

    Example usage:

    .. code:: python

       import matplotlib.pyplot as plt
       import caustics

       cosmo = caustics.FlatLambdaCDM()
       lens = caustics.lenses.SIS(cosmology=cosmo, x0=0.0, y0=0.0, th_ein=1.0)
       source = caustics.sources.Sersic(x0=0.0, y0=0.0, q=0.5, phi=0.4, n=2.0, Re=1.0, Ie=1.0)
       sim = caustics.sims.LensSource(
           lens, source, pixelscale=0.05, pixels_x=100, upsample_factor=2, z_s=1.0
       )

       img = sim()
       plt.imshow(img, origin="lower")
       plt.show()

    Attributes
    ----------
    lens: Lens
        caustics lens mass model object
    source: Source
        caustics light object which defines the background source
    pixelscale: float
        pixelscale of the sampling grid.
    pixels_x: int
        number of pixels on the x-axis for the sampling grid
    lens_light: Source, optional
        caustics light object which defines the lensing object's light
    psf: Tensor, optional
        An image to convolve with the scene. Note that if ``upsample_factor >
        1`` the psf must also be at the higher resolution.
    pixels_y: Optional[int]
        number of pixels on the y-axis for the sampling grid. If left as
        ``None`` then this will simply be equal to ``gridx``
    upsample_factor (default 1)
        Amount of upsampling to model the image. For example ``upsample_factor =
        2`` indicates that the image will be sampled at double the resolution
        then summed back to the original resolution (given by pixelscale and
        gridx/y).
    quad_level: int (default None)
        sub pixel integration resolution. This will use Gaussian quadrature to
        sample the image at a higher resolution, then integrate the image back
        to the original resolution. This is useful for high accuracy integration
        of the image, but may increase memory usage and runtime.
    z_s: optional
        redshift of the source
    name: string (default "sim")
        a name for this simulator in the parameter DAG.

    Notes:
    -----
    - The simulator will automatically pad the image to half the PSF size to
      ensure valid convolution. This is done by default, but can be turned off
      by setting ``psf_pad = False``. This is only relevant if you are using a
      PSF.
    - The upsample factor will increase the resolution of the image by the given
      factor. For example, ``upsample_factor = 2`` will sample the image at
      double the resolution, then sum back to the original resolution. This is
      used when a PSF is provided at high resolution than the original image.
      Not that the when a PSF is used, the upsample_factor must equal the PSF
      upsampling level.
    - For arbitrary pixel integration accuracy using the quad_level parameter.
      This will use Gaussian quadrature to sample the image at a higher
      resolution, then integrate the image back to the original resolution. This
      is useful for high accuracy integration of the image, but is not
      recommended for large images as it will be slow. The quad_level and
      upsample_factor can be used together to achieve high accuracy integration
      of the image convolved with a PSF.
    - A `Pixelated` light source is defined by bilinear interpolation of the
      provided image. This means that sub-pixel integration is not required for
      accurate integration of the pixels. However, if you are using a PSF then
      you should still use upsample_factor (if your PSF is supersampled) to
      ensure that everything is sampled at the PSF resolution.

    """  # noqa: E501

    def __init__(
        self,
        lens: Annotated[Lens, "caustics lens mass model object"],
        source: Annotated[
            Source, "caustics light object which defines the background source"
        ],
        pixelscale: Annotated[float, "pixelscale of the sampling grid"],
        pixels_x: Annotated[
            int, "number of pixels on the x-axis for the sampling grid"
        ],
        lens_light: Annotated[
            Optional[Source],
            "caustics light object which defines the lensing object's light",
        ] = None,
        pixels_y: Annotated[
            Optional[int], "number of pixels on the y-axis for the sampling grid"
        ] = None,
        upsample_factor: Annotated[int, "Amount of upsampling to model the image"] = 1,
        quad_level: Annotated[Optional[int], "sub pixel integration resolution"] = None,
        psf_mode: Annotated[
            Literal["off", "fft", "conv2d"], "Mode for convolving psf"
        ] = "fft",
        psf_shape: Annotated[Optional[tuple[int, ...]], "The shape of the psf"] = None,
        z_s: Annotated[
            Optional[Union[Tensor, float]], "Redshift of the source", True
        ] = None,
        psf: Annotated[
            Optional[Union[Tensor, list]], "An image to convolve with the scene", True
        ] = [[1.0]],
        x0: Annotated[
            Optional[Union[Tensor, float]],
            "center of the fov for the lens source image",
            True,
        ] = 0.0,
        y0: Annotated[
            Optional[Union[Tensor, float]],
            "center of the fov for the lens source image",
            True,
        ] = 0.0,
        name: NameType = "sim",
    ):
        super().__init__(name)

        # Lensing models
        self.lens = lens
        self.source = source
        self.lens_light = lens_light

        # Configure PSF
        if psf == [[1.0]]:
            self._psf_mode = "off"
        else:
            self._psf_mode = psf_mode
        if psf is not None:
            psf = torch.as_tensor(psf)
        self._psf_shape = psf.shape if psf is not None else psf_shape

        # Build parameters
        self.add_param("z_s", z_s)
        self.add_param("psf", psf, self.psf_shape)
        self.add_param("x0", x0)
        self.add_param("y0", y0)
        self._pixelscale = pixelscale

        # Image grid
        self._pixels_x = pixels_x
        self._pixels_y = pixels_x if pixels_y is None else pixels_y
        self._upsample_factor = upsample_factor
        self._quad_level = quad_level

        # Build the imaging grid
        self._build_grid()

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        super().to(device, dtype)
        self._grid = tuple(x.to(device, dtype) for x in self._grid)  # type: ignore[has-type]
        self._weights = self._weights.to(device, dtype)  # type: ignore[has-type]

        return self

    @property
    def upsample_factor(self):
        return self._upsample_factor

    @upsample_factor.setter
    def upsample_factor(self, value):
        self._upsample_factor = value
        self._build_grid()

    @property
    def pixels_x(self):
        return self._pixels_x

    @pixels_x.setter
    def pixels_x(self, value):
        self._pixels_x = value
        self._build_grid()

    @property
    def pixels_y(self):
        return self._pixels_y

    @pixels_y.setter
    def pixels_y(self, value):
        self._pixels_y = value
        self._build_grid()

    @property
    def quad_level(self):
        return self._quad_level

    @quad_level.setter
    def quad_level(self, value):
        self._quad_level = value
        self._build_grid()

    @property
    def pixelscale(self):
        return self._pixelscale

    @pixelscale.setter
    def pixelscale(self, value):
        self._pixelscale = value
        self._build_grid()

    @property
    def psf_shape(self):
        return self._psf_shape

    @psf_shape.setter
    def psf_shape(self, value):
        self._psf_shape = value
        self._build_grid()

    @property
    def psf_mode(self):
        return self._psf_mode

    @psf_mode.setter
    def psf_mode(self, value):
        self._psf_mode = value
        self._build_grid()

    def _build_grid(self):
        if self.psf_mode != "off":
            self._psf_pad = (self.psf_shape[1] // 2 + 1, self.psf_shape[0] // 2 + 1)
        else:
            self._psf_pad = (0, 0)

        self._n_pix = (
            self.pixels_x + self._psf_pad[0] * 2,
            self.pixels_y + self._psf_pad[1] * 2,
        )
        self._grid = meshgrid(
            self.pixelscale / self.upsample_factor,
            self._n_pix[0] * self.upsample_factor,
            self._n_pix[1] * self.upsample_factor,
        )
        self._weights = torch.ones(
            (1, 1), dtype=self._grid[0].dtype, device=self._grid[0].device
        )
        if self.quad_level is not None and self.quad_level > 1:
            finegrid_x, finegrid_y, weights = gaussian_quadrature_grid(
                self.pixelscale / self.upsample_factor, *self._grid, self.quad_level
            )
            self._grid = (finegrid_x, finegrid_y)
            self._weights = weights
        else:
            self._grid = (self._grid[0].unsqueeze(-1), self._grid[1].unsqueeze(-1))

    def _fft2_padded(self, x):
        """
        Compute the 2D Fast Fourier Transform (FFT) of a tensor with zero-padding.

        Args:
            x (Tensor): The input tensor to be transformed.

        Returns:
            Tensor: The 2D FFT of the input tensor with zero-padding.
        """
        npix = copy(self._n_pix)
        npix = (next_fast_len(npix[0]), next_fast_len(npix[1]))
        self._s = npix

        return torch.fft.rfft2(x, self._s)

    def _unpad_fft(self, x):
        """
        Remove padding from the result of a 2D FFT.

        Parameters
        ---------
        x: Tensor
            The input tensor with padding.

        Returns
        -------
        Tensor
            The input tensor without padding.
        """
        return torch.roll(
            x, (1 - self._psf_pad[0], 1 - self._psf_pad[1]), dims=(-2, -1)
        )[..., : self._s[0], : self._s[1]]

    def forward(
        self,
        params,
        source_light=True,
        lens_light=True,
        lens_source=True,
        psf_convolve=True,
    ):
        """
        forward function

        Parameters
        ----------
        params:
            Packed object
        source_light: boolean
            when true the source light will be sampled
        lens_light: boolean
            when true the lens light will be sampled
        lens_source: boolean
            when true, the source light model will be lensed by the lens mass distribution
        psf_convolve: boolean
            when true the image will be convolved with the psf
        """
        z_s, psf, x0, y0 = self.unpack(params)

        # Automatically turn off light for missing objects
        if self.source is None:
            source_light = False
        if self.lens_light is None:
            lens_light = False
        if psf.shape == (1, 1):
            psf_convolve = False

        grid = (self._grid[0] + x0, self._grid[1] + y0)

        # Sample the source light
        if source_light:
            if lens_source:
                # Source is lensed by the lens mass distribution
                bx, by = self.lens.raytrace(*grid, z_s, params)
                mu_fine = self.source.brightness(bx, by, params)
                mu = gaussian_quadrature_integrator(mu_fine, self._weights)
            else:
                # Source is imaged without lensing
                mu_fine = self.source.brightness(*grid, params)
                mu = gaussian_quadrature_integrator(mu_fine, self._weights)
        else:
            # Source is not added to the scene
            mu = torch.zeros_like(grid[0][..., 0])  # chop off quad dim

        # Sample the lens light
        if lens_light and self.lens_light is not None:
            mu_fine = self.lens_light.brightness(*grid, params)
            mu += gaussian_quadrature_integrator(mu_fine, self._weights)

        # Convolve the PSF
        if psf_convolve:
            if self.psf_mode == "fft":
                mu_fft = self._fft2_padded(mu)
                psf_fft = self._fft2_padded(psf / psf.sum())
                mu = self._unpad_fft(torch.fft.irfft2(mu_fft * psf_fft, self._s).real)
            elif self.psf_mode == "conv2d":
                mu = (
                    conv2d(
                        mu[None, None], (psf.T / psf.sum())[None, None], padding="same"
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            else:
                raise ValueError(
                    f"psf_mode should be one of 'fft' or 'conv2d', not {self.psf_mode}"
                )

        # Return to the desired image
        mu_native_resolution = (
            avg_pool2d(mu[None, None], self.upsample_factor).squeeze(0).squeeze(0)
        )
        mu_clipped = mu_native_resolution[
            self._psf_pad[1] : self.pixels_y + self._psf_pad[1],
            self._psf_pad[0] : self.pixels_x + self._psf_pad[0],
        ]
        return mu_clipped
