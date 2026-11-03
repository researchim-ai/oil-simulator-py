"""Explicit component/energy production process: transfers, mixer, PT/PH products.

Boundary pressures and duties are inputs; this is a thermodynamic process,
not the still-separate coupled pressure/flow network or thermal reservoir.
"""

import math

import torch

from .component_energy import (
    audit_component_energy,
    mix_component_energy_streams,
    separate_equilibrium_stream,
)
from .liquid import ConservationError, _number


def thermodynamic_report(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: thermodynamic_report(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [thermodynamic_report(v) for v in value]
    return value


def run_equilibrium_process(fluid, feeds, transfers, mixer):
    if not isinstance(feeds, dict) or not feeds or set(transfers) != set(feeds):
        raise ValueError("Transfers must match all component-energy feeds")
    if (
        set(mixer) - {"pressure_pa", "elevation_m", "velocity_m_s", "temperature_guess_k"}
        or "pressure_pa" not in mixer
    ):
        raise ValueError("Invalid equilibrium mixer fields")
    currents = {}
    reports = {}
    heat = []
    work = []
    entropy_heat = []
    initial_entropy = []
    for name, stream in feeds.items():
        spec = dict(transfers[name])
        if (
            set(spec)
            - {
                "pressure_pa",
                "elevation_m",
                "velocity_m_s",
                "heat_w",
                "shaft_power_w",
                "utility_temperature_k",
                "temperature_guess_k",
            }
            or "pressure_pa" not in spec
        ):
            raise ValueError("Invalid stream transfer fields")
        spec.setdefault("elevation_m", stream.elevation_m)
        spec.setdefault("velocity_m_s", stream.velocity_m_s)
        out = stream.transport(fluid, **spec)
        currents[name] = out
        guess = spec.get("temperature_guess_k")
        iq = stream.equilibrate(fluid, temperature_guess_k=guess)
        oq = out.equilibrate(fluid, temperature_guess_k=guess)
        initial_entropy.append(
            stream.molar_rate_mol_s * float(iq["molar_entropy_j_mol_k"].cpu()) if iq else 0.0
        )
        h = _number(spec.get("heat_w", 0.0), "heat_w")
        w = _number(spec.get("shaft_power_w", 0.0), "shaft_power_w")
        heat.append(h)
        work.append(w)
        entropy_heat.append(h / spec["utility_temperature_k"] if h else 0.0)
        reports[name] = {
            "incoming": stream.to_dict(),
            "outgoing": out.to_dict(),
            "incoming_equilibrium": thermodynamic_report(iq),
            "outgoing_equilibrium": thermodynamic_report(oq),
            "heat_w": h,
            "shaft_power_w": w,
        }
    combined = mix_component_energy_streams(tuple(currents.values()), fluid, **mixer)
    separated = separate_equilibrium_stream(
        combined, fluid, temperature_guess_k=mixer.get("temperature_guess_k")
    )
    final = separated["outputs"]
    audit = audit_component_energy(
        tuple(feeds.values()), final, fluid, heat_w=math.fsum(heat), shaft_power_w=math.fsum(work)
    )
    q = separated["equilibrium"]
    outgoing_entropy = (
        combined.molar_rate_mol_s * float(q["molar_entropy_j_mol_k"].cpu()) if q else 0.0
    )
    production = outgoing_entropy - math.fsum(initial_entropy) - math.fsum(entropy_heat)
    if production < -1e-6:
        raise ConservationError("Combined equilibrium process entropy audit failed")
    return {
        "model_id": fluid.model_id,
        "component_names": list(fluid.names),
        "transfers": reports,
        "mixed_stream": combined.to_dict(),
        "equilibrium": thermodynamic_report(q),
        "products": [s.to_dict() for s in final],
        "balances": {
            **audit,
            "heat_w": math.fsum(heat),
            "shaft_power_w": math.fsum(work),
            "entropy_production_w_k": production,
        },
    }
