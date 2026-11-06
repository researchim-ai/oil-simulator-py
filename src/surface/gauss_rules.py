"""Small immutable Gauss--Legendre rules; no NumPy/SciPy in physical kernels."""

import math
from functools import lru_cache


@lru_cache(maxsize=8)
def _rule(order):
    if type(order) is not int or not 2 <= order <= 64:
        raise ValueError("Gauss order must be 2..64")
    rows = []
    for i in range(1, order + 1):
        x = math.cos(math.pi * (i - 0.25) / (order + 0.5))
        for _ in range(30):
            a, b = 1.0, x
            for j in range(2, order + 1):
                a, b = b, ((2 * j - 1) * x * b - (j - 1) * a) / j
            derivative = order * (x * b - a) / (x * x - 1)
            step = b / derivative
            x -= step
            if abs(step) < 2e-16:
                break
        rows.append((x, 2 / ((1 - x * x) * derivative * derivative)))
    rows.sort()
    return tuple(x for x, _ in rows), tuple(w for _, w in rows)


def legendre_rule(order, like):
    nodes, weights = _rule(order)
    return like.new_tensor(nodes), like.new_tensor(weights)
