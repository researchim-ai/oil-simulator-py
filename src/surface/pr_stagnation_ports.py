"""Monolithic single-phase reservoir traces for resolved 1-D pipe entries.

An outflow trace preserves reservoir entropy and stagnation enthalpy. On
inflow, the receiving mixed vessel imposes its static pressure/temperature;
upwind material and total energy are supplied by the pipe Riemann state.
The outflow port velocity is solved jointly with Riemann contact compatibility.
"""

import math
from dataclasses import dataclass

import torch

from .liquid import PropertyModelError
from .peng_robinson import R


@dataclass(frozen=True)
class ReservoirTraceEntry:
    link: str
    side: str
    vessel: int
    pipe: int
    pipe_normal: float
    offset: int
    energy_scale: float


class SinglePhaseReservoirTraces:
    def __init__(self, model, state, seeds, olddata, openings, offset):
        self.model = model
        self.entries = []
        self.initial = []
        self.lower = []
        self.upper = []
        f = model.fluid
        if model.reservoir_ports != "stagnation_single_phase":
            return
        for link, (a, b) in zip(model.links, model.ends):
            if (
                not openings[link.name]
                or seeds[a]["phase_count"] != 1
                or seeds[b]["phase_count"] != 1
            ):
                continue
            if model.nodes[a].kind == "vessel" and model.nodes[b].kind == "pipe_cell":
                vi, pi, side, port = a, b, "source", link.target_port
            elif model.nodes[b].kind == "vessel" and model.nodes[a].kind == "pipe_cell":
                vi, pi, side, port = b, a, "target", link.source_port
            else:
                continue
            if port.startswith("side:"):
                # Axial stagnation traces cannot represent a perpendicular
                # jet with the body's tangential momentum.
                continue
            if model.nodes[vi].geometry is not None:
                continue
            s = state.nodes[vi][1]
            q = olddata[vi]["q"]
            scale = max(1.0, abs(float(q["molar_enthalpy_j_mol"][0].cpu())), R * s.temperature_k)
            self.entries.append(
                ReservoirTraceEntry(
                    link.name, side, vi, pi, 1.0 if port == "right" else -1.0, offset, scale
                )
            )
            offset += 3
            velocity = -self.entries[-1].pipe_normal * float(olddata[pi]["u"].cpu())
            self.initial.append(
                f.tensor(
                    [
                        math.log(s.pressure_pa),
                        s.temperature_k / f.tref,
                        velocity / model.maximum_velocity_m_s,
                    ]
                )
            )
            self.lower.append(
                f.tensor([math.log(f.min_pressure_pa), f.min_temperature_k / f.tref, -1.0])
            )
            self.upper.append(
                f.tensor([math.log(f.max_pressure_pa), f.max_temperature_k / f.tref, 1.0])
            )

    def evaluate(self, y, data):
        f = self.model.fluid
        points = {}
        residual = []
        details = []
        for e in self.entries:
            source = data[e.vessel]
            p = y[e.offset].exp()
            t = y[e.offset + 1] * f.tref
            x = source["x"][0]
            q = f.tensor_homogeneous(p, t, x, source["tags"][0])
            vout = y[e.offset + 2] * self.model.maximum_velocity_m_s
            total_h = q["molar_enthalpy_j_mol"] + 0.5 * (x * f.molar_masses).sum() * vout.square()
            outgoing = torch.stack(
                (
                    (q["molar_entropy_j_mol_k"] - source["q"]["molar_entropy_j_mol_k"][0]) / R,
                    (total_h - source["q"]["molar_enthalpy_j_mol"][0]) / e.energy_scale,
                )
            )
            incoming = torch.stack(((p / source["p"]).log(), (t - source["t"]) / f.tref))
            row = torch.where(vout >= 0, outgoing, incoming)
            residual.append(row)
            actual_vout = torch.where(vout >= 0, vout, vout * 0)
            point = {
                **source,
                "p": p,
                "t": t,
                "q": {k: v.unsqueeze(0) for k, v in q.items()},
                "port_velocity": actual_vout if e.side == "source" else -actual_vout,
                "alpha": p.new_ones(1),
                "drift": p.new_zeros(1),
                "drift_stress": p * 0,
            }
            points[(e.link, e.side)] = point
            actual_h = (
                q["molar_enthalpy_j_mol"] + 0.5 * (x * f.molar_masses).sum() * actual_vout.square()
            )
            details.append((e, point, source, vout, actual_h, row))
        return points, residual, details

    def complete(self, details, records):
        contacts = {r["name"]: r["contact_velocity"] for r in records}
        rows = []
        complete = []
        for e, point, source, vout, total_h, row in details:
            velocity = contacts[e.link] * (1.0 if e.side == "source" else -1.0)
            full = torch.cat(
                (row, ((vout - velocity) / self.model.maximum_velocity_m_s).reshape(1))
            )
            rows.append(full)
            complete.append((e, point, source, vout, total_h, full))
        return rows, complete

    def validate(self, details):
        f = self.model.fluid
        for e, point, _source, vout, _h, _row in details:
            if not bool(point["q"]["valid"].all().cpu()):
                raise PropertyModelError("Stagnation port has no stable homogeneous EOS root")
            sound = float(point["q"]["frozen_sound_speed_m_s"][0].cpu())
            if sound + abs(float(vout.cpu())) > self.model.wave_speed_bound_m_s:
                raise PropertyModelError("Stagnation port exceeds acoustic bound")
            if float(vout.cpu()) >= 0:
                if float(vout.cpu()) > sound * (1 + 1e-8):
                    raise PropertyModelError(
                        "Reservoir entry requires a resolved sonic throat upstream"
                    )
                stable = f.flash_pt(point["p"], point["t"], point["x"][0])
                if stable["phase_count"] != 1:
                    raise PropertyModelError(
                        "Single-phase stagnation trace crosses a phase boundary"
                    )

    def serialize(self, details):
        result = {}
        for e, point, source, vout, total_h, row in details:
            result.setdefault(e.link, {})[e.side] = {
                "pressure_pa": float(point["p"].cpu()),
                "temperature_k": float(point["t"].cpu()),
                "mole_fractions": point["x"][0].cpu().tolist(),
                "contact_outward_velocity_m_s": float(vout.cpu()),
                "outward_velocity_m_s": float(point["port_velocity"].cpu())
                * (1.0 if e.side == "source" else -1.0),
                "mode": "reversible_outflow" if float(vout.cpu()) >= 0 else "mixed_inflow",
                "constraint_residuals": row.cpu().tolist(),
                "reservoir_entropy_j_mol_k": float(source["q"]["molar_entropy_j_mol_k"][0].cpu()),
                "trace_entropy_j_mol_k": float(point["q"]["molar_entropy_j_mol_k"][0].cpu()),
                "reservoir_total_enthalpy_j_mol": float(
                    source["q"]["molar_enthalpy_j_mol"][0].cpu()
                ),
                "trace_total_enthalpy_j_mol": float(total_h.cpu()),
            }
        return result
