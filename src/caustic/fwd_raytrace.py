from typing import Callable, Tuple

import torch
from levmarq_torch import minimize_levmarq
from torch import Tensor

from .utils import get_cluster_means


def fwd_raytrace(
    beta_x: Tensor,
    beta_y: Tensor,
    get_beta_hat: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
    n_images: int,
    thx_range: Tuple[float, float],
    thy_range: Tuple[float, float],
    n_guesses: int = 15,
    lam: float = 1e-2,
    max_iters_guesses: int = 50,
    max_iters_final: int = 50,
) -> Tensor:
    bxy = torch.stack((beta_x, beta_y))
    thxy_min = torch.tensor((thx_range[0], thy_range[0]))
    thxy_max = torch.tensor((thx_range[1], thy_range[1]))
    fov = thxy_max - thxy_min

    while True:
        thxy_0s = fov[None, :] * torch.rand(n_guesses, 2) + thxy_min[None, :]
        thxys = minimize_levmarq(
            thxy_0s,
            bxy.repeat(n_guesses, 1),
            get_beta_hat,
            lam=lam,
            max_iters=max_iters_guesses,
        )
        # If the batch contains one point that optimizes poorly, the whole batch
        # can fail. If that happens, rerun with new guesses until it doesn't.
        if (thxys != thxy_0s).any():
            break

    # Pare down to number of images
    thxys = get_cluster_means(thxys, n_images)

    # Run final optimization
    return minimize_levmarq(
        thxys, bxy.repeat(n_images, 1), get_beta_hat, lam=lam, max_iters=max_iters_final
    )
