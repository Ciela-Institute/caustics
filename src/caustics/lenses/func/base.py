import torch

from ...utils import batch_lm


def forward_raytrace(bx, by, raytrace, epsilon, n_init, fov):
    """
    Perform a forward ray-tracing operation which maps from the source plane to the image plane.

    Parameters
    ----------
    bx: Tensor
        Tensor of x coordinate in the source plane.

        *Unit: arcsec*

    by: Tensor
        Tensor of y coordinate in the source plane.

        *Unit: arcsec*

    raytrace: function
        function that takes in the x and y coordinates in the image plane and returns the x and y coordinates in the source plane.

    epsilon: Tensor
        maximum distance between two images (arcsec) before they are considered the same image.

        *Unit: arcsec*

    n_init: int
        number of random initialization points used to try and find image plane points.

    fov: float
        the field of view in which the initial random samples are taken.

        *Unit: arcsec*

    Returns
    -------
    x_component: Tensor
        x-coordinate Tensor of the ray-traced light rays

        *Unit: arcsec*

    y_component: Tensor
        y-coordinate Tensor of the ray-traced light rays

        *Unit: arcsec*
    """
    bxy = torch.stack((bx, by)).repeat(n_init, 1)  # has shape (n_init, Dout:2)

    # Random starting points in image plane
    guesses = (torch.as_tensor(fov) * (torch.rand(n_init, 2) - 0.5)).to(
        device=bxy.device
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
    Computes the physical deflection angle of the given the reduced deflection angles [arcsec].

    Parameters
    ----------
    ax: Tensor
        Tensor of x axis reduced deflection angles in the lens plane.

        *Unit: arcsec*

    y: Tensor
        Tensor of y axis reduced deflection angles in the lens plane.

        *Unit: arcsec*

    d_s: float
        distance to the source.

        *Unit: Mpc*

    d_ls: float
        distance from lens to source.

        *Unit: Mpc*

    Returns
    --------
    x_component: Tensor
        Physical deflection Angle in the x-direction.

        *Unit: arcsec*

    y_component: Tensor
        Physical deflection Angle in the y-direction.

        *Unit: arcsec*

    """

    return (d_s / d_ls) * ax, (d_s / d_ls) * ay


def reduced_from_physical_deflection_angle(ax, ay, d_s, d_ls):
    """
    Computes the reduced deflection angle of the lens at given coordinates [arcsec].

    Parameters
    ----------
    ax: Tensor
        Tensor of x axis physical deflection angles in the lens plane.

        *Unit: arcsec*

    y: Tensor
        Tensor of y axis physical deflection angles in the lens plane.

        *Unit: arcsec*

    d_s: float
        distance to the source.

        *Unit: Mpc*

    d_ls: float
        distance from lens to source.

        *Unit: Mpc*

    Returns
    --------
    x_component: Tensor
        Reduced deflection Angle in the x-direction.

        *Unit: arcsec*

    y_component: Tensor
        Reduced deflection Angle in the y-direction.

        *Unit: arcsec*

    """

    return (d_ls / d_s) * ax, (d_ls / d_s) * ay
