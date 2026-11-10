"""Independent scalar EOS, finite-aperture and whole-graph balance audit.

Production graph stepping, column updates and port-transfer reconstruction
are not called. SciPy belongs solely to the separate PR-profile verifier.
"""

import math

import numpy as np
from check_pr_capillary_axial import audit as axial_audit
from check_pr_capillary_column import aperture_area
from check_pr_capillary_pressure import audit as normal_audit
from check_pr_capillary_pressure import close
from check_stratified_pr_flow import profile
from reference_pr_active_multifluid import packets
from reference_wetting_potential import potential

from surface.operation import _digest
from surface.pr_active_multifluid_pipe import PRActiveMultifluidState
from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_pressure import PRCapillaryPressure


def audit(
    system, before, after, report, *, normal_relaxation=True, heaters=None, phase_controls=None
):
    coupling = system.get("coupling", "imex")
    if report.get("coupling", "imex") != coupling or not report.get("coupling_converged", True):
        raise ValueError("Facility coupling declaration differs or has not converged")
    boundary_version = 0 if coupling == "imex" else 1
    if (
        before.model_id != _digest(system)
        or after.model_id != before.model_id
        or _digest(report["state"]) != _digest(after.to_dict())
    ):
        raise ValueError("Facility model/checkpoint identity differs")
    if report["normal_relaxation"] != normal_relaxation:
        raise ValueError("Facility split switch differs")
    dt = after.time_s - before.time_s
    if dt <= 0 or report["dt_s"] != dt:
        raise ValueError("Facility clocks differ")
    heaters = {} if heaters is None else dict(heaters)
    if set(heaters) - set(system["columns"]):
        raise ValueError("Unknown heater in audit request")
    oldp, newp = dict(before.pipes), dict(after.pipes)
    oldc, newc = dict(before.columns), dict(after.columns)
    olds, news = dict(before.layer_slots), dict(after.layer_slots)
    if (
        set(oldp) != set(newp)
        or set(oldp) != set(system["lines"])
        or set(oldc) != set(newc)
        or set(oldc) != set(system["columns"])
    ):
        raise ValueError("Facility inventory registry differs")
    references = [
        {n: profile(cs.to_dict(), system["columns"][n]) for n, cs in states.items()}
        for states in (oldc, newc)
    ]
    boundary_energy = [{}, {}]
    models, checked_ports, pipe_checks = {}, {}, {}
    entropy_terms = []
    energy_terms, number_terms = [], None
    for name, spec in system["lines"].items():
        model = PRCapillaryAxialPipe.from_dict(report["pipe_models"][name])
        models[name] = model
        p, source = model.pipe, model.source
        nc = len(p.fluid.names)
        if number_terms is None:
            number_terms = [[] for _ in range(nc)]
        if (
            _digest(model.pipe.to_dict()) != _digest(spec["pipe"])
            or model.boundary_geometry != "reservoir_cell"
        ):
            raise ValueError("Facility pipe or endpoint geometry differs")
        raw_source = {
            k: v
            for k, v in source.to_dict().items()
            if k not in ("boundary_heights_m", "boundary_distances_m")
        }
        expected = PRCapillaryPressure(p, **spec["parameters"]).to_dict()
        if _digest(raw_source) != _digest(expected):
            raise ValueError("Facility material or spatial source differs")
        body = report["pipes"][name]
        axial = body["axial"] if normal_relaxation else body
        transported = PRActiveMultifluidState.from_dict(axial["state"])
        pipe_checks[name] = axial_audit(model, oldp[name], transported, axial)
        checked_ports[name] = pipe_checks[name]["port_transfers"]
        if normal_relaxation:
            normal_audit(source, transported, newp[name], body["normal"])
        elif transported != newp[name]:
            raise ValueError("Unreported facility pipe update")
        for side, saved in enumerate(report["port_transfers"][name]):
            actual = checked_ports[name][side]
            if actual is None:
                if saved is not None:
                    raise ValueError("Wall acquired a material port")
            else:
                close(
                    saved["transfer_n_m_e"],
                    actual["transfer_n_m_e"],
                    "separate accepted port",
                    atol=1e-23,
                    rtol=1e-9,
                )
                close(
                    saved["capillary_work_j"],
                    actual["capillary_work_j"],
                    "separate capillary work",
                    atol=1e-23,
                    rtol=1e-9,
                )
        for version, states, bindings, ps in (
            (0, oldc, olds, oldp[name]),
            (1, newc, news, newp[name]),
        ):
            heights = []
            for side, port in enumerate(spec["ports"]):
                if port is None:
                    if model.boundaries[side] is not None:
                        raise ValueError("Unconnected endpoint acquired external stock")
                    heights.append(None)
                    boundary_energy[version][name, side] = 0.0
                    continue
                cn = port["column"]
                cs, slots = states[cn], bindings[cn]
                if len(slots) != len(cs.phase_roots) or len(set(slots)) != len(slots):
                    raise ValueError("Malformed active column registry")
                _f, geometry, sample, _amounts, _rho, _entropy = references[version][cn]
                bounds = (0.0, *cs.interface_heights_m, geometry.height_m)
                radius, center = p.diameter / 2, port["height_m"]
                fractions, active_names = {}, set()
                for i, slot in enumerate(slots):
                    low, high = max(bounds[i], center - radius), min(bounds[i + 1], center + radius)
                    fraction = aperture_area(low, high, center, p.diameter) / p.area
                    fractions[slot] = fraction
                    if fraction == 0:
                        continue
                    active_names.add(slot)
                    if version == boundary_version:
                        phase = model.boundaries[side][slot]
                        pressure, x, q = sample((low + high) / 2, i)
                        volume = fraction * p.area * p.lengths[0 if side == 0 else -1]
                        amount = volume / q["vm"]
                        close(
                            phase.volume_m3,
                            volume,
                            "finite aperture volume",
                            atol=1e-24,
                            rtol=1e-10,
                        )
                        close(
                            phase.component_moles,
                            amount * x,
                            "finite aperture components",
                            atol=1e-18,
                            rtol=2e-8,
                        )
                        close(
                            phase.pressure_pa,
                            pressure,
                            "finite aperture profile pressure",
                            atol=1e-4,
                            rtol=1e-9,
                        )
                        close(
                            phase.temperature_k,
                            cs.temperature_k,
                            "finite aperture profile temperature",
                            atol=1e-8,
                            rtol=0,
                        )
                        close(
                            phase.total_energy_j,
                            amount * q["u"],
                            "finite aperture calorics",
                            atol=1e-12,
                            rtol=2e-8,
                        )
                        if phase.momentum_kg_m_s != 0:
                            raise ValueError("Stationary finite column acquired a moving ghost")
                height = fractions.get(source.liquid, 0.0) * p.area / source.width
                heights.append(height)
                ci = 0 if side == 0 else -1
                # Explicit coupling changes exterior geometry AFTER the pipe;
                # implicit coupling changes it BEFORE the end-profile pipe
                # solve. Each declared first-order ordering has its own work.
                shape_state = newp[name] if coupling == "imex" else oldp[name]
                phase = dict(shape_state.cells[ci]).get(source.liquid)
                hfinal = (phase.volume_m3 if phase is not None else 0.0) / (
                    source.width * p.lengths[ci]
                )
                boundary_energy[version][name, side] = (
                    source.sigma * source.width / (2 * p.lengths[ci]) * (hfinal - height) ** 2
                )
                if version == boundary_version:
                    if set(model.boundaries[side]) != active_names:
                        raise ValueError("Absent aperture phase acquired ghost stock")
                    close(
                        source.boundary_heights[side],
                        height,
                        "physical boundary height",
                        atol=1e-18,
                        rtol=1e-10,
                    )
                    close(
                        source.boundary_distances[side],
                        p.lengths[ci],
                        "physical center distance",
                        atol=0,
                        rtol=0,
                    )
            h = np.array(
                [
                    dict(c)[source.liquid].volume_m3 if source.liquid in dict(c) else 0.0
                    for c in ps.cells
                ]
            ) / (source.width * np.array(p.lengths))
            surf = source.sigma * source.width / (2 * p.lengths[0]) * np.sum(np.diff(h) ** 2)
            for side, height in enumerate(heights):
                if height is not None:
                    ci = 0 if side == 0 else -1
                    surf += (
                        source.sigma * source.width / (2 * p.lengths[ci]) * (h[ci] - height) ** 2
                    )
            if source.wetting is not None:
                surf += np.sum(
                    source.width
                    * np.array(p.lengths)
                    * potential(source.wetting.to_dict(), source.sigma, h)
                )
            sign = -1 if version == 0 else 1
            energy_terms.append(sign * float(surf))
            entropy_terms.append(sign * packets(p, ps.to_dict())["entropy"])
            attached = [v for v in spec["ports"] if v is not None]
            gravity = system["columns"][attached[0]["column"]]["gravity_m_s2"] if attached else 0.0
            for cell in ps.cells:
                for _, phase in cell:
                    for j, amount in enumerate(phase.component_moles):
                        number_terms[j].append(sign * amount)
                    mass = math.fsum(
                        n * c.molar_mass_kg_mol
                        for n, c in zip(phase.component_moles, p.fluid.components)
                    )
                    energy_terms.append(
                        sign * (phase.total_energy_j + gravity * p.elevations[0] * mass)
                    )
    from dataclasses import asdict, is_dataclass

    commands = {} if phase_controls is None else phase_controls
    normalized = {
        name: {
            str(i): {
                **command,
                "inlet_phase": asdict(command["inlet_phase"])
                if is_dataclass(command["inlet_phase"])
                else command["inlet_phase"],
            }
            for i, command in cells.items()
        }
        for name, cells in commands.items()
    }
    if _digest(report.get("phase_controls", {})) != _digest(normalized):
        raise ValueError("Facility side-port request differs")
    for name, model in models.items():
        body = report["pipes"][name]
        axial = body["axial"] if normal_relaxation else body
        if _digest(axial.get("phase_controls", {})) != _digest(normalized.get(name, {})):
            raise ValueError("Pipe side-port request differs from facility command")
        side = axial.get("sideport_transfer_n_m_e", [0.0] * (nc + 2))
        for j, value in enumerate(side[:-2]):
            number_terms[j].append(-value)
        ports = [p for p in system["lines"][name]["ports"] if p is not None]
        gravity = system["columns"][ports[0]["column"]]["gravity_m_s2"] if ports else 0.0
        mass = math.fsum(
            v * c.molar_mass_kg_mol for v, c in zip(side[:-2], model.pipe.fluid.components)
        )
        energy_terms.extend(
            (
                -side[-1],
                -axial.get("sideport_actuator_work_j", 0.0),
                -mass * gravity * model.pipe.elevations[0],
            )
        )
        entropy_terms.append(-axial.get("sideport_incoming_entropy_j_k", 0.0))
    defects = []
    for name, column in system["columns"].items():
        old, new = oldc[name], newc[name]
        owned = [
            (line, side)
            for line, row in system["lines"].items()
            for side, port in enumerate(row["ports"])
            if port is not None and port["column"] == name
        ]
        saved = report["columns"][name]
        if _digest(saved["equilibrium"]["state"]) != _digest(new.to_dict()):
            raise ValueError("Column solution differs from accepted graph state")
        command = heaters.get(name, {"power_w": 0.0, "temperature_k": None})
        heat = dt * command["power_w"]
        bath = command["temperature_k"]
        if heat and (bath is None or bath <= 0):
            raise ValueError("Physical heat reservoir required")
        close(saved["heat_j"], heat, "prescribed graph heat", atol=1e-18, rtol=1e-14)
        if saved["reservoir_temperature_k"] != bath:
            raise ValueError("Graph heat reservoir differs")
        port_energy, shape = [], []
        for line, side in owned:
            tr = checked_ports[line][side]
            p = models[line].pipe
            mass = math.fsum(
                n * c.molar_mass_kg_mol
                for n, c in zip(tr["transfer_n_m_e"][:nc], p.fluid.components)
            )
            z = column["elevation_m"] + system["lines"][line]["ports"][side]["height_m"]
            port_energy.append(
                tr["transfer_n_m_e"][-1]
                + tr["capillary_work_j"]
                + mass * column["gravity_m_s2"] * z
            )
            shape.append(boundary_energy[1][line, side] - boundary_energy[0][line, side])
        close(
            saved["port_energy_to_pipes_j"],
            port_energy,
            "finite graph port energy",
            atol=1e-17,
            rtol=1e-9,
        )
        close(
            saved["boundary_shape_work_j"], shape, "finite graph shape work", atol=1e-20, rtol=1e-8
        )
        for j, (a, b) in enumerate(zip(old.component_moles, new.component_moles)):
            operands = [
                b,
                -a,
                *(checked_ports[line][side]["transfer_n_m_e"][j] for line, side in owned),
            ]
            close(
                math.fsum(operands),
                0,
                "shared finite column components",
                atol=128 * math.fsum(math.ulp(v) for v in operands) + 1e-20,
                rtol=0,
            )
            number_terms[j].extend((-a, b))
        operands = [new.total_energy_j, -old.total_energy_j, *port_energy, *shape, -heat]
        defect = math.fsum(operands)
        close(
            defect,
            0,
            "shared finite column energy",
            atol=128 * math.fsum(math.ulp(v) for v in operands) + 1e-18,
            rtol=0,
        )
        defects.append(abs(defect))
        energy_terms.extend((-old.total_energy_j, new.total_energy_j, -heat))
        entropy_terms.extend((-references[0][name][-1], references[1][name][-1]))
        if heat:
            entropy_terms.append(-heat / bath)
    terms = [*number_terms, energy_terms]
    residual = [math.fsum(t) for t in terms]
    allowances = [
        128 * math.fsum(math.ulp(v) for v in t) + (1e-18 if j == nc else 1e-20)
        for j, t in enumerate(terms)
    ]
    close(residual, np.zeros(nc + 1), "whole graph N/(E+F+PE)", atol=np.array(allowances), rtol=0)
    close(
        report["conservation_residuals"],
        residual,
        "reported graph residual",
        atol=np.array(allowances),
        rtol=0,
    )
    production = math.fsum(entropy_terms)
    entropy_allowance = 256 * math.fsum(math.ulp(v) for v in entropy_terms) + 1e-10
    if production < -entropy_allowance:
        raise ValueError("Independent whole graph entropy decreased")
    close(
        report["entropy_production_j_k"],
        production,
        "reported graph entropy",
        atol=entropy_allowance,
        rtol=0,
    )
    return {
        "accepted": True,
        "lines": len(models),
        "columns": len(oldc),
        "maximum_column_energy_defect_j": max(defects),
        "global_component_energy_residuals": residual,
        "entropy_production_j_k": production,
        "qualification": "Independent scalar PR profiles, finite aperture areas and every pipe/port/column/global balance with the declared initial/end-profile coupling and shape-work ordering; no empirical or monolithic-Newton claim",
    }
