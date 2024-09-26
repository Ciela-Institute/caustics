import torch

from ...utils import batch_lm
from ...constants import arcsec_to_rad, c_Mpc_s, days_to_seconds


def triangle_contains(p, v):
    """
    determine if point v is inside triangle p. Where p is a (3,2) tensor, and v
    is a (2,) tensor.
    """
    p01 = p[1] - p[0]
    p02 = p[2] - p[0]
    dp0p02 = p[0][0] * p02[1] - p[0][1] * p02[0]
    dp0p01 = p[0][0] * p01[1] - p[0][1] * p01[0]
    dp01p02 = p01[0] * p02[1] - p01[1] * p02[0]
    dvp02 = v[0] * p02[1] - v[1] * p02[0]
    dvp01 = v[0] * p01[1] - v[1] * p01[0]
    a = (dvp02 - dp0p02) / dp01p02
    b = -(dvp01 - dp0p01) / dp01p02
    return (a >= 0) & (b >= 0) & (a + b <= 1)


def triangle_area(p):
    """
    Determine the area of triangle p where p is a (3,2) tensor.
    """
    return (
        0.5
        * (
            p[0][0] * (p[1][1] - p[2][1])
            + p[1][0] * (p[2][1] - p[0][1])
            + p[2][0] * (p[0][1] - p[1][1])
        ).abs()
    )


def triangle_neighbors(p):
    """
    Build a set of neighbors for triangle p where p is a (3,2) tensor. The
    neighbors all have the same shape as p, but are various translations and
    reflections of p that share a common edge or vertex.
    """
    p01 = p[1] - p[0]
    p02 = p[2] - p[0]
    p12 = p[2] - p[1]
    pref = -(p - p[0]) + p[0]
    return torch.stack(
        (
            p,
            p + p01,
            p - p01,
            p + p02,
            p - p02,
            p + p12,
            p - p12,
            pref,
            pref + p01,
            pref + 2 * p01,
            pref + p02,
            pref + 2 * p02,
            pref + p01 + p02,
        ),
        dim=0,
    )


def triangle_upsample(p):
    """
    Upsample triangle p where p is a (3,2) tensor. The upsampled triangles are
    all triangles internal to p built by taking the midpoints of the edges of p.
    """
    p01 = (p[1] + p[0]) / 2
    p02 = (p[2] + p[0]) / 2
    p12 = (p[2] + p[1]) / 2
    return torch.stack(
        (
            torch.stack((p[0], p01, p02), dim=0),
            torch.stack((p01, p[1], p12), dim=0),
            torch.stack((p02, p12, p[2]), dim=0),
            torch.stack((p01, p12, p02), dim=0),
        ),
        dim=0,
    )


def triangle_equals(p1, p2):
    """
    Determine if two triangles are equal. Where p1 and p2 are (3,2) tensors.
    """
    return torch.all((p1 - p2).abs() < 1e-6)


def remove_triangle_duplicates(p):
    unique_triangles = torch.zeros((0, 3, 2))
    B = p.shape[0]
    batch_triangle_equals = torch.vmap(triangle_equals, in_dims=(None, 0))
    for i in range(B):
        # Compare current triangle with all triangles in the unique list
        if i == 0 or not batch_triangle_equals(p[i], unique_triangles).any():
            unique_triangles = torch.cat((unique_triangles, p[i].unsqueeze(0)), dim=0)

    return unique_triangles


def forward_raytrace_rootfind(ix, iy, bx, by, raytrace):
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

    Returns
    -------
    x_component: Tensor
        x-coordinate Tensor of the ray-traced light rays

        *Unit: arcsec*

    y_component: Tensor
        y-coordinate Tensor of the ray-traced light rays

        *Unit: arcsec*
    """
    ixy = torch.stack((ix, iy), dim=1)  # has shape (B, Din:2)
    bxy = torch.stack((bx, by)).repeat(ix.shape[0], 1)  # has shape (B, Dout:2)
    # Optimize guesses in image plane
    x, l, c = batch_lm(  # noqa: E741 Unused `l` variable
        ixy,
        bxy,
        lambda *a, **k: torch.stack(
            raytrace(a[0][..., 0], a[0][..., 1], *a[1:], **k), dim=-1
        ),
    )
    return x


def forward_raytrace(s, raytrace, x0, y0, fov, n, epsilon):

    # Construct a tiling of the image plane (squares at this point)
    X, Y = torch.meshgrid(
        torch.linspace(x0 - fov / 2, x0 + fov / 2, n),
        torch.linspace(y0 - fov / 2, y0 + fov / 2, n),
        indexing="ij",
    )
    E = torch.stack((X, Y), dim=-1)
    # build the upper and lower triangles within the squares of the grid
    E = torch.cat(
        (
            torch.stack((E[:-1, :-1], E[:-1, 1:], E[1:, 1:]), dim=-2),
            torch.stack((E[:-1, :-1], E[1:, :-1], E[1:, 1:]), dim=-2),
        ),
        dim=0,
    ).reshape(-1, 3, 2)

    i = 0
    while triangle_area(E[0]) > epsilon**2:

        # Expand the search to neighboring triangles
        if i > 0:  # no need for neighbors in the first iteration
            E = torch.vmap(triangle_neighbors)(E)
            E = E.reshape(-1, 3, 2)
            E = remove_triangle_duplicates(E)
            # Upsample the triangles
            E = torch.vmap(triangle_upsample)(E)
            E = E.reshape(-1, 3, 2)

        S = raytrace(E[..., 0], E[..., 1])
        S = torch.stack(S, dim=-1)

        # Identify triangles that contain the source plane point
        locate = torch.vmap(triangle_contains, in_dims=(0, None))(S, s)
        E = E[locate]
        i += 1

    # Rootfind the source plane point in the triangle
    E = E.sum(dim=1) / 3
    E = forward_raytrace_rootfind(E[..., 0], E[..., 1], s[0], s[1], raytrace)
    return E[..., 0], E[..., 1]


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
