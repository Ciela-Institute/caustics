import torch

from ...utils import batch_lm
from ...constants import arcsec_to_rad, c_Mpc_s, days_to_seconds


def inside_triangle(p, v):
    dp1p2 = p[1][0] * p[2][1] - p[1][1] * p[2][0]
    dp0p1 = p[0][0] * p[1][1] - p[0][1] * p[1][0]
    dp0p2 = p[0][0] * p[2][1] - p[0][1] * p[2][0]
    dpp2 = v[0] * p[2][1] - v[1] * p[2][0]
    dpp1 = v[0] * p[1][1] - v[1] * p[1][0]
    a = (dpp2 - dp0p2) / dp1p2
    b = -(dpp1 - dp0p1) / dp1p2
    if a < 0 or b < 0 or a + b > 1:
        return False
    return True


def triangle_area(p):
    dp1p2 = p[1][0] * p[2][1] - p[1][1] * p[2][0]
    dp0p2 = p[0][0] * p[2][1] - p[0][1] * p[2][0]
    dp0p1 = p[0][0] * p[1][1] - p[0][1] * p[1][0]
    return 0.5 * torch.abs(dp1p2 + dp0p2 + dp0p1)


def triangle_search(s, pimg, psrc, raytrace, epsilon):
    """
        Perform a triangle search to find the image plane points that map to the source plane point.

        Parameters
        ----------
        s: Tensor
            Tensor of x and y coordinates in the source plane.
        pimg: Tensor
    `       Tensor of x and y coordinates in the image plane for the three triangle points. Shape (3,2) for the three vertices of the triangle in the 2D image plane.
        psrc: Tensor
    `       Tensor of x and y coordinates in the source plane for the three triangle points. Shape (3,2) for the three vertices of the triangle in the 2D source plane.
        raytrace: function
            function that takes in the x and y coordinates in the image plane and returns the x and y coordinates in the source plane.
        epsilon: Tensor
            maximum distance between two images (arcsec) before they are considered the same image.
    """
    # Case 1: the point is outside the triangle
    if not inside_triangle(psrc, s):
        return torch.zeros((0, 2))  # end search, point is not in triangle

    pmid_img = pimg.sum(dim=0) / 3  # new point at the center of the triangle
    pmid_src = raytrace(pmid_img)

    # Case 2: size of triangle is within epsilon, optimize image plane position
    if triangle_area(pimg) < epsilon**2:
        res = forward_raytrace(
            s[0], s[1], pmid_img[0], pmid_img[1], raytrace, epsilon
        )  # fixme, optimize position
        if inside_triangle(pimg, res):
            return res.unsqueeze(0)

    # Case 3: divide triangle for recursive exploration
    pnew_img = pimg.repeat(3, 1, 1)
    pnew_img[0][0] = pmid_img
    pnew_img[1][1] = pmid_img
    pnew_img[2][2] = pmid_img
    pnew_src = psrc.repeat(3, 1, 1)
    pnew_src[0][0] = pmid_src
    pnew_src[1][1] = pmid_src
    pnew_src[2][2] = pmid_src
    res1 = triangle_search(s, pnew_img[0], pnew_src[0], raytrace, epsilon)
    res2 = triangle_search(s, pnew_img[1], pnew_src[1], raytrace, epsilon)
    res3 = triangle_search(s, pnew_img[2], pnew_src[2], raytrace, epsilon)
    return torch.cat((res1, res2, res3), dim=0)


def forward_raytrace(bx, by, raytrace, n_init, epsilon, fov):
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
    guesses = (
        torch.as_tensor(fov, dtype=bx.dtype)
        * (torch.rand(n_init, 2, dtype=bx.dtype) - 0.5)
    ).to(
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


def time_delay_arcsec2_to_days(d_l, d_s, d_ls, z_l):
    """
    Computes a scaling factor to use in time delay calculations which converts
    the time delay (i.e. potential and deflection angle squared terms) from
    arcsec^2 to units of days.
    """
    return (1 + z_l) / c_Mpc_s * d_s * d_l / d_ls * arcsec_to_rad**2 / days_to_seconds
