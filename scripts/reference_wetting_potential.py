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
    length = parameters["adsorption_thickness_m"]
    dry = parameters.get("dry_extension") == "c2_half_adsorption"
    r = length / (np.maximum(h, length / 2) if dry else h)
    value = scale * (-(r**2) / 2 + r**5 / 5)
    if dry:
        x = np.minimum(h / (length / 2), 1)
        value = np.where(h < length / 2, scale * (246 * x**3 - 442 * x**4 + 200.4 * x**5), value)
    return value


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
    if parameters.get("dry_extension") == "c2_half_adsorption":

        def derivative(height):
            ratio = length / np.maximum(height, length / 2)
            original = scale / length * (ratio**3 - ratio**6)
            x = np.minimum(height / (length / 2), 1)
            polynomial = scale / (length / 2) * (738 * x**2 - 1768 * x**3 + 1002 * x**4)
            return np.where(height < length / 2, polynomial, original)

        low, high = np.minimum(a, b), np.maximum(a, b)
        cut = np.maximum(low, np.minimum(high, length / 2))
        integral = np.zeros_like(low + high)
        for left, right in ((low, cut), (cut, high)):
            hh = left[..., None] + (right - left)[..., None] * (nodes + 1) / 2
            integral += (right - left) * (derivative(hh) * weights / 2).sum(-1)
        return np.where(high > low, integral / np.where(high > low, high - low, 1), derivative(low))
    r = length / h
    return (scale / length * (r**3 - r**6) * (weights / 2)).sum(-1)
