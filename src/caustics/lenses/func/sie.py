from ...utils import translate_rotate, derotate


def reduced_deflection_angle_sie(x0, y0, q, phi, b, x, y, s=0.0):
    x, y = translate_rotate(x, y, x0, y0, phi)
    psi = (q**2 * (x**2 + s**2) + y**2).sqrt()
    f = (1 - q**2).sqrt()
    ax = b * q.sqrt() / f * (f * x / (psi + s)).atan()  # fmt: skip
    ay = b * q.sqrt() / f * (f * y / (psi + q**2 * s)).atanh()  # fmt: skip

    return derotate(ax, ay, phi)


def potential_sie(x0, y0, q, phi, b, x, y, s=0.0):
    ax, ay = reduced_deflection_angle_sie(x0, y0, q, phi, b, x, y, s)
    ax, ay = derotate(ax, ay, -phi)
    x, y = translate_rotate(x, y, x0, y0, phi)
    return x * ax + y * ay


def convergence_sie(x0, y0, q, phi, b, x, y, s=0.0):
    x, y = translate_rotate(x, y, x0, y0, phi)
    psi = (q**2 * (x**2 + s**2) + y**2).sqrt()
    return 0.5 * q.sqrt() * b / psi
