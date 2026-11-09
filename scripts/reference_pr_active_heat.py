"""Independent prescribed-heat audit using scalar PR phase packets only."""

import math

from reference_pr_active_multifluid import packets


def check_commands(report, rates, temperatures):
    thermal = report.get("thermal")
    if rates is None and temperatures is None:
        if thermal is not None:
            raise ValueError("Unrequested active pipe heat")
    elif (
        thermal is None
        or thermal["requested_heat_rates_w"] != rates
        or thermal["requested_reservoir_temperatures_k"] != temperatures
    ):
        raise ValueError("Active pipe heat differs from requested boundary")


def audit(pipe, before, after, report, close):
    if report["state"] != after.to_dict():
        from surface.operation import _digest

        if _digest(report["state"]) != _digest(after.to_dict()):
            raise ValueError("Published heat source state differs")
    a, b = packets(pipe, before.to_dict()), packets(pipe, after.to_dict())
    rates = report["requested_heat_rates_w"]
    temperatures = report["requested_reservoir_temperatures_k"]
    rates = [0.0] * len(before.cells) if rates is None else rates
    temperatures = [None] * len(before.cells) if temperatures is None else temperatures
    locations, powers, baths, gains = [], [], [], []
    for i, (ca, cb, power, bath) in enumerate(
        zip(before.cells, after.cells, rates, temperatures, strict=True)
    ):
        if tuple(n for n, _ in ca) != tuple(n for n, _ in cb):
            raise ValueError("Heat changed active phase registry")
        if not isinstance(power, dict):
            if len(ca) != 1 and power:
                raise ValueError("Ambiguous multiphase heat")
            power = {ca[0][0]: power} if len(ca) == 1 else {}
        if set(power) - set(pipe.names) or any(
            not math.isfinite(v) or (v != 0 and n not in dict(ca)) for n, v in power.items()
        ):
            raise ValueError("Invalid or absent phase heat")
        if isinstance(bath, dict) and (
            set(bath) - set(pipe.names)
            or any(not math.isfinite(t) or t <= 0 for t in bath.values())
        ):
            raise ValueError("Invalid phase utility temperature")
        for (name, p), (_, q) in zip(ca, cb):
            rate = power.get(name, 0.0)
            gain = report["dt_s"] * rate
            close(q.component_moles, p.component_moles, atol=0.0, rtol=0.0)
            close(q.volume_m3, p.volume_m3, atol=0.0, rtol=0.0)
            close(q.momentum_kg_m_s, p.momentum_kg_m_s, atol=0.0, rtol=0.0)
            close(q.total_energy_j, p.total_energy_j + gain, atol=1e-10, rtol=1e-12)
            if not rate:
                continue
            t = bath[name] if isinstance(bath, dict) else bath
            if (
                t is None
                or t <= 0
                or gain * (t - q.temperature_k) < -1e-11
                or gain * (t - p.temperature_k) < -1e-11
            ):
                raise ValueError("Nonpassive utility heat boundary")
            locations.append([i, name])
            powers.append(rate)
            baths.append(t)
            gains.append(gain)
    if [list(x) for x in report["active_locations"]] != locations:
        raise ValueError("Heat phase locations differ")
    for key, values in [
        ("heat_rates_w", powers),
        ("reservoir_temperatures_k", baths),
        ("phase_heat_j", gains),
    ]:
        close(report[key], values, atol=1e-12, rtol=1e-12)
    entropy_in = math.fsum(q / t for q, t in zip(gains, baths))
    production = b["entropy"] - a["entropy"] - entropy_in
    if production < -1e-7:
        raise ValueError("Independent pipe heat entropy decreased")
    close(report["external_heat_j"], math.fsum(gains), atol=1e-11, rtol=1e-12)
    close(report["incoming_entropy_j_k"], entropy_in, atol=1e-11, rtol=1e-12)
    close(report["entropy_production_j_k"], production, atol=1e-7, rtol=1e-9)
