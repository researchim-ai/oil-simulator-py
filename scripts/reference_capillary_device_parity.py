"""Unit-aware checkpoint comparison for the capillary functional matrix."""

import math


def audit(a, b):
    if (
        a["format"] != b["format"]
        or a["request"] != b["request"]
        or a["normalized_model"] != b["normalized_model"]
        or a["source_sha256"] != b["source_sha256"]
        or a["family"] != b["family"]
    ):
        raise ValueError("Capillary CPU/CUDA provenance differs")
    count = 0
    worst = (0.0, "")
    tolerances = {
        "thickness_m": (1e-17, 1e-9),
        "face_velocity_m_s": (1e-12, 1e-8),
        "time_s": (5e-15, 1e-12),
        "physical_heat_j": (1e-22, 1e-8),
        "gas_thermostat_energy_j": (1e-20, 1e-8),
        "numerical_heat_j": (1e-22, 1e-8),
        "component_moles": (1e-22, 1e-10),
        "volume_m3": (1e-24, 1e-10),
        "momentum_kg_m_s": (1e-22, 1e-8),
        "pressure_pa": (1e-4, 1e-10),
        "temperature_k": (1e-9, 0.0),
    }

    def compare(x, y, path, quantity=None):
        nonlocal count, worst
        if isinstance(x, dict):
            if not isinstance(y, dict) or x.keys() != y.keys():
                raise ValueError("Capillary state registry differs: " + path)
            for k in x:
                compare(x[k], y[k], path + "/" + k, k)
        elif isinstance(x, list):
            if not isinstance(y, list) or len(x) != len(y):
                raise ValueError("Capillary state shape differs: " + path)
            for i, (u, v) in enumerate(zip(x, y, strict=True)):
                compare(u, v, path + "/" + str(i), quantity)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            if (
                not isinstance(y, (int, float))
                or isinstance(y, bool)
                or not math.isfinite(x)
                or not math.isfinite(y)
            ):
                raise ValueError("Nonfinite capillary parity field")
            if quantity == "total_energy_j":
                tol = 64 * max(math.ulp(x), math.ulp(y)) + 1e-22
            elif quantity in tolerances:
                absolute, relative = tolerances[quantity]
                tol = absolute + relative * max(abs(x), abs(y))
            else:
                raise ValueError("Unknown physical parity unit: " + str(quantity))
            fraction = abs(x - y) / tol
            count += 1
            if fraction > worst[0]:
                worst = (fraction, path)
            if fraction > 1:
                raise ValueError(f"Capillary device parity failed: {path}: {x} vs {y}")
        elif x != y:
            raise ValueError("Capillary checkpoint metadata differs: " + path)

    sa, sb = a["result"]["steps"], b["result"]["steps"]
    if len(sa) != len(sb):
        raise ValueError("Different accepted capillary histories")
    compare(a["initial"], b["initial"], "initial")
    for i, (x, y) in enumerate(zip(sa, sb, strict=True)):
        compare(x["state"], y["state"], f"steps/{i}")
    compare(a["final"], b["final"], "final")
    return {
        "accepted": True,
        "checked_state_scalars": count,
        "maximum_tolerance_fraction": worst[0],
        "worst_path": worst[1],
        "tolerances": tolerances,
        "total_energy_tolerance": "64 ULP + 1e-22 J",
    }
