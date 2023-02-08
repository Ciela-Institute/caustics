import torch

from ..utils import to_elliptical, translate_rotate
from .base import Source


class Sersic(Source):
    def __init__(self, device=torch.device("cpu"), use_lenstronomy_k=False):
        """
        Args:
            lenstronomy_k_mode: set to `True` to calculate k in the Sersic exponential
                using the same formula as lenstronomy. Intended primarily for testing.
        """
        super().__init__(device)
        self.lenstronomy_k_mode = use_lenstronomy_k

    def brightness(self, thx, thy, thx0, thy0, q, phi, index, th_e, I_e, s=None):
        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        ex, ey = to_elliptical(thx, thy, q)
        e = (ex**2 + ey**2).sqrt() + s

        if self.lenstronomy_k_mode:
            k = 1.9992 * index - 0.3271
        else:
            k = 2 * index - 1 / 3 + 4 / 405 / index + 46 / 25515 / index**2

        exponent = -k * ((e / th_e) ** (1 / index) - 1)
        return I_e * exponent.exp()


# class Sersic(Source):
#     def __init__(self, thx0, thy0, q, phi, index, th_e, I_e, device=torch.device("cpu"), use_lenstronomy_k=False):
#         """
#         Args:
#             lenstronomy_k_mode: set to `True` to calculate k in the Sersic exponential
#                 using the same formula as lenstronomy. Intended primarily for testing.
#         """
#         super().__init__(device)
#         self.lenstronomy_k_mode = use_lenstronomy_k
#         self.thx0 = torch.nn.Parameter(data=torch.tensor(thx0, device=device), requires_grad=True)
#         self.thy0 = torch.nn.Parameter(data=torch.tensor(thy0, device=device), requires_grad=True)
#         self.q = torch.nn.Parameter(data=torch.tensor(q, device=device), requires_grad=True)
#         self.phi = torch.nn.Parameter(data=torch.tensor(phi, device=device), requires_grad=True)
#         self.index = torch.nn.Parameter(data=torch.tensor(index, device=device), requires_grad=True)
#         self.th_e = torch.nn.Parameter(data=torch.tensor(th_e, device=device), requires_grad=True)
#         self.I_e = torch.nn.Parameter(data=torch.tensor(I_e, device=device), requires_grad=True)
#
#     def brightness(self, thx, thy):
#         s = torch.tensor(0.0, device=self.device, dtype=self.thx0.dtype) #if s is None else s
#         thx, thy = translate_rotate(thx, thy, self.thx0, self.thy0, self.phi)
#         ex, ey = to_elliptical(thx, thy, self.q)
#         e = (ex**2 + ey**2).sqrt() + s
#
#         if self.lenstronomy_k_mode:
#             k = 1.9992 * self.index - 0.3271
#         else:
#             k = 2 * self.index - 1 / 3 + 4 / 405 / self.index + 46 / 25515 / self.index**2
#
#         exponent = -k * ((e / self.th_e) ** (1 / self.index) - 1)
#         return self.I_e * exponent.exp()
#
#
# if __name__ == '__main__':
#     import numpy as np
#     thx = torch.linspace(-1, 1, 100).float()
#     thx, thy = torch.meshgrid(thx, thx)
#     thx0 = np.array([[0., 0.]]).T
#     thy0 = np.array([[0., 0.]]).T
#     q = np.array([[0.5, 0.5]]).T
#     phi = np.array([[0., 0.]]).T
#     index = np.array([[1., 1.]]).T
#     th_e = np.array([[1., 1.]]).T
#     I_e = np.array([[1., 1.]]).T
#
#     print(thx0.shape)
#     sersic = Sersic(thx0, thy0, q, phi, index, th_e, I_e)
#     y = sersic.brightness(thx, thy)
