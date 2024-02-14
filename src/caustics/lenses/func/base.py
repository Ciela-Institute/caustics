import torch

from ...utils import batch_lm


def forward_raytrace(bx, by, raytrace, epsilon, n_init, fov):
    """
    Perform a forward ray-tracing operation which maps from the source plane to the image plane.

    Parameters
    ----------
    bx: Tensor
        Tensor of x coordinate in the source plane (scalar).
    by: Tensor
        Tensor of y coordinate in the source plane (scalar).
    raytrace: callable
        function which takes in the image plane coordinates and returns the source plane coordinates.
    epsilon: Tensor
        maximum distance between two images (arcsec) before they are considered the same image.
    n_init: int
        number of random initialization points used to try and find image plane points.
    fov: float
        the field of view in which the initial random samples are taken.

    Returns
    -------
    tuple[Tensor, Tensor]
        Ray-traced coordinates in the x and y directions.
    """
    bxy = torch.stack((bx, by)).repeat(n_init, 1)  # has shape (n_init, Dout:2)

    # Random starting points in image plane
    guesses = torch.as_tensor(fov) * (
        torch.rand(n_init, 2) - 0.5
    )  # Has shape (n_init, Din:2)

    # Optimize guesses in image plane
    x, l, c = batch_lm(  # noqa: E741 Unused `l` variable
        guesses,
        bxy,
        lambda *a, **k: torch.stack(
            raytrace(a[0][..., 0], a[0][..., 1], *a[1:], **k), dim=-1
        ),
    )

    # Clip points that didn't converge
    x = x[c < 1e-2 * epsilon**2]

    # Cluster results into n-images
    res = []
    while len(x) > 0:
        res.append(x[0])
        d = torch.linalg.norm(x - x[0], dim=-1)
        x = x[d > epsilon]

    res = torch.stack(res, dim=0)
    return res[..., 0], res[..., 1]


def physical_from_reduced_deflection_angle(ax, ay, d_s, d_ls):
    """
    Compute the physical deflection angle from the reduced deflection angle.

    Parameters
    ----------
    ax: Tensor
        x component of the reduced deflection angle.
    ay: Tensor
        y component of the reduced deflection angle.
    d_s: float
        distance to the source (Mpc).
    d_ls: float
        distance from lens to source (Mpc).
    """
    return ((d_s / d_ls) * ax, (d_s / d_ls) * ay)


def reduced_from_physical_deflection_angle(ax, ay, d_s, d_ls):
    """
    Compute the reduced deflection angle from the physical deflection angle.

    Parameters
    ----------
    ax: Tensor
        x component of the physical deflection angle.
    ay: Tensor
        y component of the physical deflection angle.
    d_s: float
        distance to the source (Mpc).
    d_ls: float
        distance from lens to source (Mpc).
    """
    return ((d_ls / d_s) * ax, (d_ls / d_s) * ay)
