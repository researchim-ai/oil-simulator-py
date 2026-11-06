"""Joint chemical-potential face reconstructions for single-phase pipe cells.

The algebraic traces are part of the same Newton as stocks and momentum.
They preserve isothermal hydrostatic states across arbitrary terrain and
across liquid/gas interfaces that coincide with faces. Mixed cells retain
the conservative homogeneous cell-centred discretization.
"""

import math
from dataclasses import dataclass, replace

import torch

from .liquid import HydraulicConvergenceError, PropertyModelError
from .peng_robinson import R


@dataclass(frozen=True)
class HydrostaticFace:
    cell: int
    port: str
    offset: int
    count: int
    active: object
    root: str
    height_difference_m: float


def equilibrium_face_guesses(model, state):
    """Warm-start an already equilibrated profile, without changing any stocks.

    Chemical constants are checked before sharing a face pressure. Non-steady
    initial data keep the ordinary initialisation. These are only algebraic
    starting values: the coupled residual always determines the final traces.
    """
    targets = [
        i
        for i, n in enumerate(model.nodes)
        if n.hydrostatic_reconstruction and n.elevation_change_m
    ]
    if (
        not targets
        or not model.gravity_m_s2
        or any(s.hydrostatic_face_guesses is not None for _, s in state.nodes)
    ):
        return state
    cells = [s for _, s in state.nodes]
    temperature = cells[0].temperature_k
    if any(
        len(s.phase_roots) != 1
        or s.momentum_kg_m_s != 0
        or abs(s.temperature_k - temperature) > 1e-10
        for s in cells
    ):
        return state
    f = model.fluid
    data = [model._state_data(i, s) for i, s in enumerate(cells)]
    active = (data[0]["n"] > 0).nonzero().flatten()
    if any(not torch.equal((d["n"] > 0).nonzero().flatten(), active) for d in data):
        return state
    constants = torch.stack(
        [
            d["x"][0, active].log()
            + d["q"]["log_fugacity_coefficients"][0, active]
            + (d["p"] / f.pref).log()
            + f.molar_masses[active] * model.gravity_m_s2 * node.elevation_m / (R * temperature)
            for node, d in zip(model.nodes, data)
        ]
    )
    if not bool(((constants - constants[0]).abs().max() < 1e-10).cpu()):
        return state
    from .pr_inclined_pipe import solve_chemical_nodes

    entries, initial, chemical = [], [], []
    for i in targets:
        node, d = model.nodes[i], data[i]
        x = d["x"][0, active]
        for port, normal in (("left", -1), ("right", 1)):
            dz = normal * node.elevation_change_m / 2
            height = round(node.elevation_m + dz, 12)
            p = (d["p"] - d["rho"] * model.gravity_m_s2 * dz).clamp(
                f.min_pressure_pa, f.max_pressure_pa
            )
            initial.append(torch.cat((p.log().reshape(1), (x[:-1] / x[-1]).log())))
            chemical.append(
                constants[0]
                - f.molar_masses[active] * model.gravity_m_s2 * height / (R * temperature)
            )
            entries.append((i, port, height))
    solved = []
    batch = max(1, 128 // len(active))
    try:
        for start in range(0, len(entries), batch):
            count = len(entries[start : start + batch])
            solved.extend(
                solve_chemical_nodes(
                    f,
                    torch.stack(initial[start : start + batch]),
                    f.tensor([temperature] * count),
                    torch.stack(chemical[start : start + batch]),
                    active,
                )
            )
    except (HydraulicConvergenceError, PropertyModelError):
        return state  # A failed optional starting guess does not change physics.
    pressures, guesses = {}, {}
    for (i, port, height), row in zip(entries, solved):
        pressure = pressures.setdefault(height, float(row[0].exp().cpu()))
        xa = torch.cat((row[1:], row.new_zeros(1))).softmax(0)
        x = row.new_zeros(len(f.names)).index_copy(0, active, xa)
        guesses.setdefault(i, []).append((port, pressure, tuple(x.cpu().tolist())))
    return replace(
        state,
        nodes=tuple(
            (name, replace(s, hydrostatic_face_guesses=tuple(guesses[i])) if i in guesses else s)
            for i, (name, s) in enumerate(state.nodes)
        ),
    )


class HydrostaticFaces:
    def __init__(self, model, state, seeds, olddata, active, offset):
        self.model = model
        self.entries, self.initial, self.lower, self.upper = [], [], [], []
        f = model.fluid
        for i, (node, (_, old), seed) in enumerate(zip(model.nodes, state.nodes, seeds)):
            if (
                not node.hydrostatic_reconstruction
                or not model.gravity_m_s2
                or not node.elevation_change_m
                or seed["phase_count"] != 1
            ):
                continue
            indices = active[i]
            count = len(indices)
            amounts = f.tensor(old.component_moles)[indices]
            amounts = torch.where(amounts > 0, amounts, sum(old.component_moles) * 1e-8)
            x = amounts / amounts.sum()
            phase = seed["phases"][0]
            ql = f.tensor_homogeneous(
                seed["pressure_pa"], seed["temperature_k"], phase["mole_fractions"], "liquid"
            )
            qg = f.tensor_homogeneous(
                seed["pressure_pa"], seed["temperature_k"], phase["mole_fractions"], "vapor"
            )
            z = phase["compressibility_factor"]
            root = (
                "liquid"
                if float((z - ql["compressibility_factor"]).abs().cpu())
                <= float((z - qg["compressibility_factor"]).abs().cpu())
                else "vapor"
            )
            guesses = {
                port: (pressure, x) for port, pressure, x in (old.hydrostatic_face_guesses or ())
            }
            for port, normal in (("left", -1.0), ("right", 1.0)):
                dz = node.elevation_change_m * normal / 2
                pressure = (
                    f.tensor(old.pressure_pa) - olddata[i]["rho"] * model.gravity_m_s2 * dz
                ).clamp(f.min_pressure_pa, f.max_pressure_pa)
                fractions = x
                if port in guesses:
                    pp, xx = guesses[port]
                    pressure = f.tensor(pp).clamp(f.min_pressure_pa, f.max_pressure_pa)
                    fractions = f.tensor(xx)[indices]
                    fractions = torch.where(fractions > 0, fractions, x * 1e-8)
                    fractions = fractions / fractions.sum()
                self.entries.append(HydrostaticFace(i, port, offset, count, indices, root, dz))
                self.initial.append(
                    torch.cat((pressure.log().reshape(1), (fractions[:-1] / fractions[-1]).log()))
                )
                self.lower.append(f.tensor([math.log(f.min_pressure_pa)] + [-100.0] * (count - 1)))
                self.upper.append(f.tensor([math.log(f.max_pressure_pa)] + [100.0] * (count - 1)))
                offset += count

    def evaluate(self, y, data):
        f = self.model.fluid
        rows, records = [], []
        valid = y.new_ones((), dtype=torch.bool)
        for entry in self.entries:
            d = data[entry.cell]
            local = y[entry.offset : entry.offset + entry.count]
            pressure = local[0].exp()
            logits = torch.cat((local[1:], local.new_zeros(1)))
            x = y.new_zeros(len(f.names)).index_copy(0, entry.active, logits.softmax(0))
            q = f.tensor_homogeneous(pressure, d["t"], x, entry.root)
            target = (
                d["x"][0, entry.active].clamp_min(1e-300).log()
                + d["q"]["log_fugacity_coefficients"][0, entry.active]
                + (d["p"] / f.pref).log()
            )
            mu = (
                x[entry.active].clamp_min(1e-300).log()
                + q["log_fugacity_coefficients"][entry.active]
                + (pressure / f.pref).log()
            )
            residual = (
                mu
                - target
                + (
                    f.molar_masses[entry.active]
                    * self.model.gravity_m_s2
                    * entry.height_difference_m
                    / (R * d["t"])
                )
            )
            rows.append(residual)
            point = {
                **{k: v for k, v in d.items() if k != "hydrostatic_faces"},
                "p": pressure,
                "x": x.unsqueeze(0),
                "alpha": pressure.new_ones(1),
                "q": {key: value.unsqueeze(0) for key, value in q.items()},
            }
            d.setdefault("hydrostatic_faces", {})[entry.port] = point
            records.append((entry, point, residual))
            valid = (
                valid
                & q["valid"]
                & (q["frozen_sound_speed_m_s"] + d["u"].abs() <= self.model.wave_speed_bound_m_s)
            )
        return rows, records, valid

    @staticmethod
    def pattern(model, offsets, entries, openings):
        pairs = set()
        for e in entries:
            aux = range(e.offset, e.offset + e.count)
            body = range(offsets[e.cell], offsets[e.cell + 1])
            for row in aux:
                pairs.update((row, col) for col in [*body, *aux])
            affected = {e.cell}
            for link, (a, b) in zip(model.links, model.ends):
                if openings[link.name] and e.cell in (a, b):
                    affected.update((a, b))
            for i in affected:
                pairs.update((row, col) for row in range(offsets[i], offsets[i + 1]) for col in aux)
        return pairs

    def validate(self, records):
        for _entry, point, residual in records:
            if not bool(
                (
                    point["q"]["valid"].all()
                    & torch.isfinite(residual).all()
                    & (residual.abs().max() < 1e-8)
                ).cpu()
            ):
                raise PropertyModelError("Hydrostatic face reconstruction rejected")

    def serialize(self, records):
        result = {}
        for entry, point, residual in records:
            result.setdefault(self.model.names[entry.cell], {})[entry.port] = {
                "pressure_pa": float(point["p"].cpu()),
                "mole_fractions": point["x"][0].cpu().tolist(),
                "compressibility_factor": float(point["q"]["compressibility_factor"][0].cpu()),
                "chemical_residuals": residual.cpu().tolist(),
            }
        return result
