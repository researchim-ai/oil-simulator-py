"""Scalar/NumPy wall free energy and independent quadrature of its pressure."""

import math

import numpy as np


def potential(parameters, sigma, height):
    h = np.asarray(height, dtype=float)
    if parameters is None or parameters["equilibrium_contact_angle_deg"] == 0:
        return np.zeros_like(h)
    scale = (
        10 / 3 * sigma * (1 - math.cos(math.radians(parameters["equilibrium_contact_angle_deg"])))
    )
    r = parameters["adsorption_thickness_m"] / h
    return scale * (-(r**2) / 2 + r**5 / 5)


def pressure(parameters, sigma, before, after):
    a, b = np.asarray(before), np.asarray(after)
    if parameters is None or parameters["equilibrium_contact_angle_deg"] == 0:
        return np.zeros_like(a + b)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    h = a[..., None] + (b - a)[..., None] * (nodes + 1) / 2
    length = parameters["adsorption_thickness_m"]
    scale = (
        10 / 3 * sigma * (1 - math.cos(math.radians(parameters["equilibrium_contact_angle_deg"])))
    )
    r = length / h
    return (scale / length * (r**3 - r**6) * (weights / 2)).sum(-1)
