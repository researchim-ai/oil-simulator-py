"""Independent axial integration of inclined cuts; no runtime geometry kernels."""

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def disk_cut(h, d):
    h = min(d, max(0.0, h))
    r = d / 2
    if h > r:
        a, m = disk_cut(d - h, d)
        return math.pi * r * r - a, math.pi * r**3 - d * a + m
    if h < 1e-4 * d:
        a = quad(lambda t: 2 * math.sqrt(max(0, t * (d - t))), 0, h, epsabs=1e-30, epsrel=1e-11)[0]
        m = quad(
            lambda t: 2 * t * math.sqrt(max(0, t * (d - t))), 0, h, epsabs=1e-30, epsrel=1e-11
        )[0]
        return a, m
    angle = math.acos(1 - h / r)
    area = r * r * (angle - math.sin(angle) * math.cos(angle))
    return area, r * area - 2 / 3 * (h * (d - h)) ** 1.5


@lru_cache(maxsize=8192)
def cut(d, length, angle, h):
    a = abs(math.sin(angle))
    c = math.cos(angle)
    height = length * a + d * c
    h = min(height, max(0.0, h))
    if a == 0:
        v, m = disk_cut(h, d)
        return v * length, m * length
    if abs(angle) == math.pi / 2:
        area = math.pi * d * d / 4
        return area * h, area * h * h / 2
    points = sorted(
        {0.0, length, min(length, max(0.0, h / a)), min(length, max(0.0, (h - d * c) / a))}
    )

    def integrand(s, moment):
        area, m = disk_cut((h - a * s) / c, d)
        return a * s * area + c * m if moment else area

    out = []
    for moment in (False, True):
        out.append(
            sum(
                quad(
                    lambda s, moment=moment: integrand(s, moment),
                    lo,
                    hi,
                    epsabs=1e-14 * math.pi * d * d * length / 4,
                    epsrel=2e-11,
                )[0]
                for lo, hi in zip(points, points[1:])
                if hi > lo
            )
        )
    return tuple(out)


def grid(geometry, interfaces, order):
    d, length, angle = geometry.diameter_m, geometry.length_m, geometry.inclination_rad
    total = math.pi * d * d * length / 4
    height = length * abs(math.sin(angle)) + d * math.cos(angle)
    points, weights = np.polynomial.legendre.leggauss(order)
    heights = []
    volumes = []
    for lo, hi in zip([0.0, *interfaces], [*interfaces, height]):
        va, ma = cut(d, length, angle, float(lo))
        vb, mb = cut(d, length, angle, float(hi))
        volume = vb - va
        target = va + (points + 1) / 2 * volume
        hh = np.array(
            [
                brentq(
                    lambda z, v=v: cut(d, length, angle, z)[0] - v, 0, height, xtol=1e-14 * height
                )
                for v in target
            ]
        )
        ww = weights / 2 * volume
        mean = (ww * hh).sum() / volume
        variance = (ww * (hh - mean) ** 2).sum()
        ww = ww * (1 + ((mb - ma) - volume * mean) * (hh - mean) / variance)
        if np.any(ww <= 0) or abs(ww.sum() - volume) > 1e-10 * total:
            raise ValueError("Inclined quadrature is not a positive volume rule")
        heights.append(hh)
        volumes.append(ww)
    return heights, volumes
