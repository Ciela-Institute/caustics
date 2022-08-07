import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from caustic.models.layers import BilinearInterpolation


"""
Comments:
This is a complete simulator, working with Sersic and Gaussian sources, it has a generate batch capability and works with 
torch. 

We will need to rethink some of these function. For example, Sersic should only have a single implementation in the code, 
not here. (same for SIE and Shear etc.)

For SIE, we should implement the derivatives explicitly like it is done here for speed up purposes.

I implemented a bunch of extra function for gaussian psf corruption and gaussian noise. These is probably a smarter way to 
do this in a future version of the simulator.
"""
PRIORS_BOUNDS = {
  # Singular Isothermal Elliptical Profile parameters
  "r_ein": (0.5, 2), # Einstein radius (arcsec)
  "q": (0.3, 1), # axis ratio (1 = circle, 0 < q < 1 is an ellipse)
  "phi": (0, np.pi),  # orientation of the ellipse [0, 2pi]
  "x0": (-0.1, 0.1),  # x coordinate of the lens (arcsec)
  "y0": (-0.1, 0.1),  # y coordinate of the lens (arcsec)
  "gamma_ext": (0., 0.05), # amplitude of the constant shear field
  "phi_ext": (0, np.pi),  # orientation of the constant shear field
  # sersic source params
  "xs": (-0.2, 0.2),        # x coordinate of the source (arcsec)
  "ys": (-0.2, 0.2),        # y coordinate of the source (arcsec)
  "qs": (0.3, 1),           # axis ratio of the source Sérsic profile
  "phi_s": (0, np.pi),       # Orientation of the source Sérsic profile
  "n": (0.5, 2),            # Sérsic index (1=exponential profile for disc, 4=de Vaucouleur for bulge)
  "r_eff": (0.1, 0.3),      # Effective radius of the source (arcsec)
  "I_eff": (10, 20),        # Effective intensity of the source
  # # sersic lens light params # not implemented yet
  # "n_lens_light": (0.5, 2), # Sérsic index of the lens light
  # "r_eff_lens_light": (0.7, 1.2), # Effective radius of the lens light
  # "amp_lens_light": (50, 60), # Effective amplitude of the lens light
}

PARAMS = [
  "r_ein",
  "q",
  "phi",
  "x0",
  "y0",
  "gamma_ext",
  "phi_ext",
  "xs",
  "ys",
  "qs",
  "phi_s",
  "n",
  "r_eff",
  "I_eff",
  # "n_lens_light",
  # "r_eff_lens_light",
  # "amp_lens_light",
]

class SIEShear:
    """
    This model produces convolved images of (pixelated or Sersic or Gaussian) background galaxy with
    a SIE+Shear lens model.
    """

    def __init__(
            self,
            pixels=128,
            image_fov=7.68,
            src_fov=3.0,
            psf_cutout_size=15,
            device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    ):
        self.src_fov = src_fov
        self.pixels = pixels
        self.image_fov = image_fov
        self.s_scale = image_fov / pixels / 10000  # Make profile non-singular
        # coordinates for image
        x = torch.linspace(-1, 1, self.pixels) * self.image_fov / 2
        xx, yy = torch.meshgrid(x, x)
        # reshape for broadcast to [batch_size, x and y, pixels, pixels]
        self.theta1 = torch.Tensor(xx).view(1, 1, pixels, pixels).to(device)
        self.theta2 = torch.Tensor(yy).view(1, 1, pixels, pixels).to(device)
        # coordinates for psf
        self.r_squared = self.theta1 ** 2 + self.theta2 ** 2
        self.r_squared = T.CenterCrop(psf_cutout_size)(self.r_squared)

        self.resampler_layer = BilinearInterpolation(pixels, pixels).to(device)

    def sersic_source(self, beta1, beta2, xs, ys, q, phi_s, n, r_eff, I_eff):
        bn = 2 * n - 1 / 3  # approximate solution to gamma(2n;b_n) = 0.5 * Gamma(2n) for n > 0.36
        # shift and rotate coordinates to major/minor axis system
        beta1 = beta1 - xs
        beta2 = beta2 - ys
        beta1, beta2 = self._rotate(beta1, beta2, phi_s)
        r = torch.sqrt(beta1 ** 2 + beta2 ** 2 / q ** 2)
        return I_eff * torch.exp(-bn * (r / r_eff) ** (1 / n) - 1)

    def gaussian_source(self, beta1, beta2, xs, ys, q, phi_s, w):
        # shift and rotate coordinates to major/minor axis system
        beta1 = beta1 - xs
        beta2 = beta2 - ys
        beta1, beta2 = self._rotate(beta1, beta2, phi_s)
        rho_sq = beta1 ** 2 + beta2 ** 2 / q ** 2
        return torch.exp(-0.5 * rho_sq / w ** 2)

    # ================== Pixelated Source ====================
    def kappa_field(
            self,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.
    ):
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        if q > 0.95:
            return 0.5 * r_ein / torch.sqrt(theta1 ** 2 + theta2 ** 2 / q ** 2 + self.s_scale ** 2)
        else:
            b, s = self._param_conv(q, r_ein)
            return 0.5 * b / torch.sqrt(theta1 ** 2 + theta2 ** 2 / q ** 2 + s ** 2)

    def lens_source(
            self,
            source,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            psf_fwhm=0.05
    ):
        if q > 0.95:
            alpha1, alpha2 = torch.tensor_split(self.approximate_deflection_angles(r_ein, q, phi, x0, y0), 2, dim=1)
        else:
            alpha1, alpha2 = torch.tensor_split(self.analytical_deflection_angles(r_ein, q, phi, x0, y0), 2, dim=1)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        # lens equation
        beta1 = self.theta1 - alpha1 - alpha1_ext
        beta2 = self.theta2 - alpha2 - alpha2_ext
        x_src_pix, y_src_pix = self.src_coord_to_pix(beta1, beta2)
        warp = torch.concat([x_src_pix, y_src_pix], dim=1)
        im = self.resampler_layer(source, warp)
        psf = self.psf_model(psf_fwhm)
        im = self.convolve_with_psf(im, psf)
        return im

    def noisy_lens_source(
            self,
            source,
            noise_rms: float = 1e-3,
            psf_fwhm: float = 0.,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.05,
    ):
        im = self.lens_source(source, r_ein, q, phi, x0, y0, gamma_ext, phi_ext, psf_fwhm)
        peak = torch.amax(im, dim=(1, 2, 3), keepdim=True)
        im += torch.normal(mean=torch.zeros_like(im), std=noise_rms * peak)
        return im

    # =============== Gaussian Source ========================
    def lens_source_gaussian_func(
            self,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            xs: float = 0.,
            ys: float = 0.,
            qs: float = 1.,
            phi_s: float = 0.,
            w: float = 0.1,
            psf_fwhm: float = 0.05
    ):
        if q > 0.95:
            alpha1, alpha2 = torch.tensor_split(self.approximate_deflection_angles(r_ein, q, phi, x0, y0), 2, dim=1)
        else:
            alpha1, alpha2 = torch.tensor_split(self.analytical_deflection_angles(r_ein, q, phi, x0, y0), 2, dim=1)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        # lens equation
        beta1 = self.theta1 - alpha1 - alpha1_ext
        beta2 = self.theta2 - alpha2 - alpha2_ext
        # sample intensity from functional form
        im = self.gaussian_source(beta1, beta2, xs, ys, qs, phi_s, w)
        psf = self.psf_model(psf_fwhm)
        im = self.convolve_with_psf(im, psf)
        return im

    def noisy_lens_gaussian_source(
            self,
            noise_rms: float = 1e-3,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            xs: float = 0,
            ys: float = 0,
            qs: float = 1.,
            phi_s: float = 0.,
            w: float = 0.1,
            psf_fwhm: float = 0.05
    ):
        im = self.lens_source_gaussian_func(r_ein, q, phi, x0, y0, gamma_ext, phi_ext, xs, ys, qs, phi_s, w, psf_fwhm)
        im += torch.normal(torch.zeros_like(im), std=noise_rms)
        return im

    # =============== Sersic Source ========================
    def lens_source_sersic_func(
            self,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            xs: float = 0.,
            ys: float = 0.,
            qs: float = 1.,
            phi_s: float = 0.,
            n: float = 1,  # 1=exponential profile for disc, 4=Vaucouleur
            r_eff: float = 0.1,
            psf_fwhm: float = 0.05,
            I_eff: float = 10.
    ):
        if q > 0.95:
            alpha1, alpha2 = torch.tensor_split(self.approximate_deflection_angles(r_ein, q, phi, x0, y0), 2, dim=1)
        else:
            alpha1, alpha2 = torch.tensor_split(self.analytical_deflection_angles(r_ein, q, phi, x0, y0), 2, dim=1)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        # lens equation
        beta1 = self.theta1 - alpha1 - alpha1_ext
        beta2 = self.theta2 - alpha2 - alpha2_ext
        im = self.sersic_source(beta1, beta2, xs, ys, qs, phi_s, n, r_eff, I_eff)
        psf = self.psf_model(psf_fwhm)
        im = self.convolve_with_psf(im, psf)
        return im

    def noisy_lens_sersic_func(
            self,
            noise_rms: float = 1e-1,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            xs: float = 0,
            ys: float = 0,
            qs: float = 1.,
            phi_s: float = 0.,
            n: float = 1,
            r_eff: float = 0.1,
            psf_fwhm: float = 0.05
    ):
        im = self.lens_source_sersic_func(r_ein, q, phi, x0, y0, gamma_ext, phi_ext, xs, ys, qs, phi_s, n, r_eff, psf_fwhm)
        peaks = torch.amax(im, dim=(1, 2, 3), keepdim=True)
        im += torch.normal(mean=torch.zeros_like(im), std=noise_rms * peaks)
        return im

    def lens_source_sersic_func_vec(self, x, psf_fwhm):
        # assume x has shape [batch_size, 13]
        r_ein, q, phi, x0, y0, gamma_ext, phi_ext, xs, ys, qs, phi_s, n, r_eff, I_eff = [_x[:, None, None] for _x in torch.tensor_split(x, 14, dim=-1)]
        alpha1, alpha2 = torch.tensor_split(self.analytical_deflection_angles(r_ein, q, phi, x0, y0), 2, dim=1)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        # lens equation
        beta1 = self.theta1 - alpha1 - alpha1_ext
        beta2 = self.theta2 - alpha2 - alpha2_ext
        im = self.sersic_source(beta1, beta2, xs, ys, qs, phi_s, n, r_eff, I_eff)
        psf = self.psf_model(psf_fwhm)
        im = self.convolve_with_psf(im, psf)
        return im

    def generate_batch(self, batch_size, psf_fwhm):
        x = torch.concat(
            [torch.rand(size=(batch_size, 1)) * (PRIORS_BOUNDS[p][1] - PRIORS_BOUNDS[p][0]) + PRIORS_BOUNDS[p][0]
             for p in PARAMS],
            dim=1
        )
        lensed_image = self.lens_source_sersic_func_vec(x, psf_fwhm)
        return lensed_image, x # INPUT, LABELS

    def noisy_lens_sersic_func_vec(self, x, noise_rms, psf_fwhm):
        im = self.lens_source_sersic_func_vec(x, psf_fwhm)
        peaks = torch.amax(im, dim=(1, 2, 3))
        noise_rms = noise_rms * peaks
        im += torch.normal(mean=torch.zeros_like(im), std=noise_rms)
        return im, noise_rms

    def lens_gaussian_source_func_given_alpha(
            self,
            alpha,
            xs: float = 0,
            ys: float = 0,
            q: float = 1.,
            phi_s: float = 0.,
            w: float = 0.1,
            psf_fwhm: float = 0.05
    ):
        alpha1, alpha2 = torch.tensor_split(alpha, 2, dim=1)
        beta1 = self.theta1 - alpha1
        beta2 = self.theta2 - alpha2
        im = self.gaussian_source(beta1, beta2, xs, ys, q, phi_s, w)
        psf = self.psf_model(psf_fwhm)
        im = self.convolve_with_psf(im, psf)
        return im

    def src_coord_to_pix(self, x, y):
        dx = self.src_fov / (self.pixels - 1)
        xmin = -0.5 * self.src_fov
        ymin = -0.5 * self.src_fov
        i_coord = (x - xmin) / dx
        j_coord = (y - ymin) / dx
        return i_coord, j_coord

    def external_shear_potential(self, gamma_ext, phi_ext):
        rho = torch.sqrt(self.theta1 ** 2 + self.theta2 ** 2)
        varphi = torch.atan2(self.theta2 ** 2, self.theta1 ** 2)
        return 0.5 * gamma_ext * rho ** 2 * torch.cos(2 * (varphi - phi_ext))

    def external_shear_deflection(self, gamma_ext, phi_ext):
        if not isinstance(phi_ext, torch.Tensor):
            phi_ext = torch.Tensor(np.atleast_1d(phi_ext))
        # see Meneghetti Lecture Scripts equation 3.83 (constant shear equation)
        alpha1 = gamma_ext * (self.theta1 * torch.cos(phi_ext) + self.theta2 * torch.sin(phi_ext))
        alpha2 = gamma_ext * (-self.theta1 * torch.sin(phi_ext) + self.theta2 * torch.cos(phi_ext))
        return alpha1, alpha2

    def potential(self, r_ein, q, phi, x0, y0):  # arcsec^2
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        if q > 0.95:
            return r_ein * torch.sqrt(theta1 ** 2 + theta2 ** 2 / q ** 2 + self.s_scale ** 2)
        else:
            b, s = self._param_conv(q, r_ein)
            phi_x, phi_y = torch.tensor_split(self.analytical_deflection_angles(r_ein, q, phi, x0, y0), 2, dim=1)
            theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
            psi = torch.sqrt(q ** 2 * (s ** 2 + theta1 ** 2) + theta2 ** 2)
            varphi = theta1 * phi_x + theta2 * phi_y
            varphi -= b * q * s * torch.log(torch.sqrt(psi + s) ** 2 + (1 - q ** 2) * theta1 ** 2)
            varphi += b * q * s * np.log((1 + q) * s)
            return varphi

    def approximate_deflection_angles(self, r_ein, q, phi, x0, y0):
        b, s = self._param_conv(q, r_ein)
        # rotate to major/minor axis coordinates
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        psi = torch.sqrt(q ** 2 * (s ** 2 + theta1 ** 2) + theta2 ** 2)
        alpha1 = b * theta1 / (psi + s)
        alpha2 = b * theta2 / (psi + q**2 * s)
        # # rotate back to original orientation of coordinate system
        alpha1, alpha2 = self._rotate(alpha1, alpha2, -phi)
        return torch.concat([alpha1, alpha2], dim=1)  # stack alphas into tensor of shape [batch_size, 2, pix, pix]

    def analytical_deflection_angles(self, r_ein, q, phi, x0, y0):
        if not isinstance(q, torch.Tensor):
            q = torch.Tensor(np.atleast_1d(q))
        b, s = self._param_conv(q, r_ein)
        # rotate to major/minor axis coordinates
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        psi = torch.sqrt(q ** 2 * (s ** 2 + theta1 ** 2) + theta2 ** 2)
        alpha1 = b / torch.sqrt(1. - q ** 2) * torch.atan(np.sqrt(1. - q ** 2) * theta1 / (psi + s))
        alpha2 = b / torch.sqrt(1. - q ** 2) * torch.atanh(np.sqrt(1. - q ** 2) * theta2 / (psi + s * q ** 2))
        # # rotate back
        alpha1, alpha2 = self._rotate(alpha1, alpha2, -phi)
        return torch.concat([alpha1, alpha2], dim=1)

    def rotated_and_shifted_coords(self, x0, y0, phi):
        ###
        # Important to shift then rotate, we move to the point of view of the
        # lens before rotating the lens (rotation and translation are not commutative).
        ###
        theta1 = self.theta1 - x0
        theta2 = self.theta2 - y0
        rho = torch.sqrt(theta1 ** 2 + theta2 ** 2)
        varphi = torch.atan2(theta2, theta1) - phi
        theta1 = rho * torch.cos(varphi)
        theta2 = rho * torch.sin(varphi)
        return theta1, theta2

    @staticmethod
    def _rotate(x, y, angle):
        if not isinstance(angle, torch.Tensor):
            angle = torch.Tensor(np.atleast_1d(angle))
        return x * torch.cos(angle) + y * torch.sin(angle), -x * torch.sin(angle) + y * torch.cos(angle)

    def _param_conv(self, q, r_ein):
        if not isinstance(q, torch.Tensor):
            q = torch.Tensor(np.atleast_1d(q))
        r_ein_conv = 2. * q * r_ein / torch.sqrt(1. + q ** 2)
        b = r_ein_conv * torch.sqrt((1 + q ** 2) / 2)
        s = self.s_scale * torch.sqrt((1 + q ** 2) / (2 * q ** 2))
        return b, s

    @staticmethod
    def _qphi_to_ellipticity(q, phi):
        e1 = (1. - q) / (1. + q) * torch.cos(2 * phi)
        e2 = (1. - q) / (1. + q) * torch.sin(2 * phi)
        return e1, e2

    @staticmethod
    def _ellipticity_to_qphi(e1, e2):
        if not isinstance(e1, torch.Tensor):
            e1 = torch.Tensor(np.atleast_1d(e1))
            e2 = torch.Tensor(np.atleast_1d(e2))
        phi = torch.atan2(e2, e1) / 2
        c = torch.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - c) / (1 + c)
        return q, phi

    @staticmethod
    def _shear_polar_to_cartesian(r, phi):
        x = r * torch.cos(2 * phi)
        y = r * torch.sin(2 * phi)
        return x, y

    @staticmethod
    def _shear_cartesian_to_polar(x, y):
        r = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x) / 2
        return r, phi

    def model_to_physical(self, x):
        # Method used to compute likelihood given model predictions
        r_ein, e1, e2, x0, y0, gamma1, gamma2, xs, ys, e1s, e2s, n, r_eff = torch.tensor_split(x, 13, dim=-1)
        q, phi = self._ellipticity_to_qphi(e1, e2)
        q = q - F.relu(q - 0.95)
        gamma, gamma_phi = self._shear_cartesian_to_polar(gamma1, gamma2)
        qs, phi_s = self._ellipticity_to_qphi(e1s, e2s)
        n = F.relu(n) + 0.5  # prevents explosion at n = 0
        r_eff = F.relu(r_eff) + self.src_fov / self.pixels
        z = torch.concat([r_ein, q, phi, x0, y0, gamma, gamma_phi, xs, ys, qs, phi_s, n, r_eff], dim=-1)
        return z

    def physical_to_model(self, z):
        # method used to compute model loss in logit space
        r_ein, q, phi, x0, y0, gamma, gamma_phi, xs, ys, qs, phi_s, n, r_eff = torch.tensor_split(z, 13, dim=-1)
        e1, e2 = self._qphi_to_ellipticity(q, phi)
        gamma1, gamma2 = self._shear_polar_to_cartesian(gamma, gamma_phi)
        e1s, e2s = self._qphi_to_ellipticity(qs, phi_s)
        x = torch.concat([r_ein, e1, e2, x0, y0, gamma1, gamma2, xs, ys, e1s, e2s, n, r_eff], dim=-1)
        return x

    def psf_model(self, psf_fwhm: float):
        psf_sigma = psf_fwhm / (2 * np.sqrt(2. * np.log(2.)))
        psf = torch.exp(-0.5 * self.r_squared / psf_sigma ** 2)
        psf /= torch.sum(psf, dim=(1, 2, 3), keepdim=True)
        return psf

    def convolve_with_psf(self, images, psf):
        B, C, *D = images.shape
        # stack channel dim in batch size, since we want them to be independent
        out = F.conv2d(images.view(B * C, 1, *D), psf, stride=1, padding="same")
        return out.view(B, C, *D)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    phys = SIEShear(pixels=128)

    lensed_images, x = phys.generate_batch(16, 0.01)
    fig, axs = plt.subplots(4, 4, figsize=(12, 16))
    for i in range(4):
        for j in range(4):
            k = 4 * i + j
            axs[i, j].imshow(lensed_images[k, 0, ...], cmap="magma")
            axs[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
