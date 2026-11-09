"""Prescribed per-phase utility heat, fixed physical N/V/momentum, batched PR.

No heat is assigned to an absent phase or divided between phases implicitly.
This is a boundary power, not a heat-transfer coefficient or a finite wall.
Passive heat transfer must respect both endpoint temperatures and Clausius.
"""

import math
from dataclasses import replace

from .liquid import ConservationError, PropertyModelError, _number
from .pr_homogeneous_uv import recover_homogeneous_phases
from .pr_stock_balances import entropy_allowance


def apply_heat(pipe, state, dt_s, rates, temperatures):
    requested_rates, requested_temperatures = rates, temperatures
    dt = _number(dt_s, "heat step", positive=True)
    count = len(state.cells)
    rates = [0.0] * count if rates is None else list(rates)
    temperatures = [None] * count if temperatures is None else list(temperatures)
    if len(rates) != count or len(temperatures) != count:
        raise ValueError("One heat boundary per physical pipe cell required")
    before = pipe.inspect(state)
    flat, powers, baths = [], [], []
    cells = [list(cell) for cell in state.cells]
    for i, (cell, power, bath) in enumerate(zip(state.cells, rates, temperatures)):
        if not isinstance(power, dict):
            value = _number(power, "heat rate")
            if len(cell) != 1 and value != 0:
                raise ValueError("Multiphase heat requires explicit named phase rates")
            power = {cell[0][0]: value} if len(cell) == 1 else {}
        if set(power) - set(pipe.names) or (isinstance(bath, dict) and set(bath) - set(pipe.names)):
            raise ValueError("Unknown phase heat boundary")
        power = {n: _number(v, "phase heat rate") for n, v in power.items()}
        if any(v != 0 and n not in dict(cell) for n, v in power.items()):
            raise ValueError("Cannot heat an absent phase")
        if isinstance(bath, dict):
            bath = {
                n: _number(v, "heat reservoir temperature", positive=True) for n, v in bath.items()
            }
        elif bath is not None:
            bath = _number(bath, "heat reservoir temperature", positive=True)
        for j, (name, p) in enumerate(cell):
            q = power.get(name, 0.0)
            t = bath.get(name) if isinstance(bath, dict) else bath
            if q and t is None:
                raise ValueError("Heat input requires reservoir temperature")
            if not q:
                continue
            if q * (t - p.temperature_k) < -abs(q) * 32 * math.ulp(max(t, p.temperature_k)):
                raise PropertyModelError("Prescribed heat opposes reservoir temperature gradient")
            flat.append((i, j, name, p))
            powers.append(q)
            baths.append(t)
    f = pipe.fluid
    gains = [dt * q for q in powers]
    if flat:
        recovered = recover_homogeneous_phases(
            f,
            [p.component_moles for _, _, _, p in flat],
            [p.volume_m3 for _, _, _, p in flat],
            [p.momentum_kg_m_s for _, _, _, p in flat],
            [p.total_energy_j + q for (_, _, _, p), q in zip(flat, gains)],
            [pipe.slots[pipe.names.index(n)]["root"] for _, _, n, _ in flat],
            [p.temperature_k for _, _, _, p in flat],
        )
        for (i, j, n, _), p, q, t in zip(flat, recovered, powers, baths):
            if q * (t - p.temperature_k) < -abs(q) * 32 * math.ulp(max(t, p.temperature_k)):
                raise PropertyModelError("Prescribed heat crosses reservoir temperature")
            cells[i][j] = (n, p)
    new = replace(state, cells=tuple(tuple(c) for c in cells))
    after = pipe.inspect(new)
    heat = math.fsum(gains)
    entropy_in = math.fsum(q / t for q, t in zip(gains, baths))
    production = after["entropy"] - before["entropy"] - entropy_in
    if production < -entropy_allowance(after["entropy"], before["entropy"], entropy_in):
        raise ConservationError("Pipe heat decreased fluid/reservoir entropy")
    return new, {
        "dt_s": dt,
        "state": new.to_dict(),
        "external_heat_j": heat,
        "incoming_entropy_j_k": entropy_in,
        "entropy_production_j_k": production,
        "active_locations": [(i, n) for i, _, n, _ in flat],
        "requested_heat_rates_w": requested_rates,
        "requested_reservoir_temperatures_k": requested_temperatures,
        "heat_rates_w": powers,
        "reservoir_temperatures_k": baths,
        "phase_heat_j": gains,
        "conservation_residuals": pipe.ledger(state, new, external_heat_j=heat),
    }
