"""CPU geometry reference; independent of Torch layout/pickup/control code."""

import math

import numpy as np


def layout(body, vessel):
    if vessel.geometry is None:
        return None
    volume = np.asarray(body["n"]) * np.asarray(body["vm"])
    order = np.argsort(-np.asarray(body["rho"]), kind="stable")
    rho = np.asarray(body["rho"])[order]
    if len(rho) > 1 and np.any(rho[:-1] - rho[1:] <= 1e-10 * max(1, np.max(abs(rho)))):
        raise ValueError("Unresolved geometric density order")
    g = vessel.geometry
    upper = np.cumsum(volume[order])
    tops = np.array([g.level_m(min(g.volume_m3, max(0, float(v)))) for v in upper])
    bottoms = np.r_[0, tops[:-1]]
    inverse = np.argsort(order)
    return {
        "phase_order": order,
        "phase_tops_m": tops[inverse],
        "phase_bottoms_m": bottoms[inverse],
        "density_order_resolved": True,
    }


def _circle_fraction(h, diameter):
    if h <= 0:
        return 0.0
    if h >= diameter:
        return 1.0
    if h > diameter / 2:
        return 1 - _circle_fraction(diameter - h, diameter)
    theta = math.acos(1 - 2 * h / diameter)
    area = (
        theta**3 * (2 / 3 - 2 * theta**2 / 15 + 4 * theta**4 / 315)
        if theta < 1e-3
        else theta - math.sin(theta) * math.cos(theta)
    )
    return area / math.pi


def weights(body, vessel, height=None, diameter=None):
    if height is None:
        return np.asarray(body["n"]) * np.asarray(body["vm"]) / vessel.volume_m3
    layers = layout(body, vessel)
    bottom = height - diameter / 2
    return np.array(
        [
            _circle_fraction(float(b) - bottom, diameter)
            - _circle_fraction(float(a) - bottom, diameter)
            for a, b in zip(layers["phase_bottoms_m"], layers["phase_tops_m"])
        ]
    )


def feedback(control, body, vessel, names):
    column = np.asarray(body["x"])[:, names.index(control.component)]
    i = int(np.argmax(column))
    present = bool(column[i] > 0 and column[i] >= control.minimum_mole_fraction)
    level = float(layout(body, vessel)["phase_tops_m"][i]) if present else 0.0
    opening = (
        float(
            np.clip(
                control.opening_bias + control.opening_gain_per_m * (level - control.setpoint_m),
                0,
                1,
            )
        )
        if present
        else 0.0
    )
    return opening, level, present


def check_layout(logged, body, vessel):
    expected = layout(body, vessel)
    if expected is None:
        return
    if logged is None or set(logged) != set(expected):
        raise ValueError("Missing or invalid geometric journal")
    for key, value in expected.items():
        if key in ("phase_order", "density_order_resolved"):
            if np.any(np.asarray(logged[key]) != value):
                raise ValueError("Geometric phase order differs")
        elif np.max(abs(np.asarray(logged[key]) - value)) > 1e-9:
            raise ValueError("Geometric interface levels differ")
