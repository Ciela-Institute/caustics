from typing import Optional, Union

from torch import Tensor
from .base import PointSpreadFunction


class Zernike(PointSpreadFunction):
    def __init__(
            self, 
            order_n: int,
            reference_psf: Tensor = None,
            coefficients: list[Union[Tensor, float, None], ...] = None,
            # TODO specify coordinate system in advance for this one I think, similar to PixelatedConvergence
            name: str = None
            ):
        super().__init__(name)
        self.reference_psf = reference_psf
        if coefficients is None:
            coefficients = [None] * order_n
        for n in order_n:
            self.add_param(f"A_{n}", coefficients[n])
    
    # Ref code from Connor's AstroPhot
    # def iter_nm(self, n):
        # nm = []
        # for n_i in range(n + 1):
            # for m_i in range(-n_i, n_i + 1, 2):
                # nm.append((n_i, m_i))
        # return nm

    # @staticmethod
    # @lru_cache(maxsize=1024)
    # def coefficients(n, m):
        # C = []
        # for k in range(int((n - abs(m)) / 2) + 1):
            # C.append(
                # (
                    # k,
                    # (-1) ** k
                    # * binom(n - k, k)
                    # * binom(n - 2 * k, (n - abs(m)) / 2 - k),
                # )
            # )
        # return C
    
    # def Z_n_m(self, rho, phi, n, m, efficient=True):
        # Z = torch.zeros_like(rho)
        # if efficient:
            # T_cache = {0: None}
            # R_cache = {}
        # for k, c in self.coefficients(n, m):
            # if efficient:
                # if (n - 2 * k) not in R_cache:
                    # R_cache[n - 2 * k] = rho ** (n - 2 * k)
                # R = R_cache[n - 2 * k]
                # if m not in T_cache:
                    # if m < 0:
                        # T_cache[m] = torch.sin(abs(m) * phi)
                    # elif m > 0:
                        # T_cache[m] = torch.cos(m * phi)
                # T = T_cache[m]
            # else:
                # R = rho ** (n - 2 * k)
                # if m < 0:
                    # T = torch.sin(abs(m) * phi)
                # elif m > 0:
                    # T = torch.cos(m * phi)

            # if m == 0:
                # Z += c * R
            # elif m < 0:
                # Z += c * R * T
            # else:
                # Z += c * R * T
        # return Z


    # def evaluate_model(self, X=None, Y=None, image=None, parameters=None):
        # if X is None:
            # Coords = image.get_coordinate_meshgrid()
            # X, Y = Coords - parameters["center"].value[..., None, None]

        # phi = self.angular_metric(X, Y, image, parameters)

        # r = self.radius_metric(X, Y, image, parameters)
        # r = r / self.r_scale

        # G = torch.zeros_like(X)

        # i = 0
        # A = image.pixel_area * parameters["Anm"].value
        # for n, m in self.nm_list:
            # G += A[i] * self.Z_n_m(r, phi, n, m)
            # i += 1

        # G[r > 1] = 0.0

        # return G
    
    def kernel(self, x: Tensor, y: Tensor, params: Optional["Packed"] = None) -> Tensor:
        return super().kernel(x, y, params)
