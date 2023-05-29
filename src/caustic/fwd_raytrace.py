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
    """
    Implements a forward ray tracing algorithm for a strong gravitational lensing system. 

    Args:
        beta_x (Tensor): The x coordinates of the source positions in the source plane.
        beta_y (Tensor): The y coordinates of the source positions in the source plane.
        get_beta_hat (Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]): A function that returns 
            the predicted source positions given the lensed image positions.
        n_images (int): The number of images to produce.
        thx_range (Tuple[float, float]): The range of x coordinates in the lens plane to consider for initial guesses.
        thy_range (Tuple[float, float]): The range of y coordinates in the lens plane to consider for initial guesses.
        n_guesses (int, optional): The number of initial guesses for the lensed image positions. Default is 15.
        lam (float, optional): The damping parameter for the Levenberg-Marquardt optimization. Default is 1e-2.
        max_iters_guesses (int, optional): The maximum number of iterations for the optimization of initial guesses. Default is 50.
        max_iters_final (int, optional): The maximum number of iterations for the final optimization. Default is 50.

    Returns:
        Tensor: The optimized lensed image positions in the lens plane.
    
    This function first generates a set of initial guesses for the lensed image positions. 
    These guesses are then optimized using the Levenberg-Marquardt algorithm to match the 
    observed source positions. If the optimization fails for any of the initial guesses, 
    new guesses are generated and the process is repeated until a successful optimization is achieved.

    Once the initial optimization is complete, the results are pared down to the desired number 
    of images using a clustering algorithm, and a final round of optimization is performed. 
    The function returns the final optimized lensed image positions.

    Note: If the number of images is greater than the number of observed source positions, 
    the function may not be able to find a solution.
    """

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
