import torch

from ...utils import translate_rotate


def reduced_deflection_angle_point(x0, y0, th_ein, x, y, s=0.0):
    x, y = translate_rotate(x, y, x0, y0)
    th = (x**2 + y**2).sqrt() + s
    ax = x / th**2 * th_ein**2
    ay = y / th**2 * th_ein**2
    return ax, ay


def potential_point(x0, y0, th_ein, x, y, s=0.0):
    ax, ay = reduced_deflection_angle_point(x0, y0, th_ein, x, y, s)
    x, y = translate_rotate(x, y, x0, y0)
    return x * ax + y * ay


def convergence_point(x0, y0, x, y):
    x, y = translate_rotate(x, y, x0, y0)
    return torch.where((x == 0) & (y == 0), torch.inf, 0.0)
