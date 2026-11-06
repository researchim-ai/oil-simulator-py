"""One monolithic PR plant with an inertial, mixed-profile pipe component.

Profile ends bind actual finite vessels of the ordinary arbitrary graph. No
sequential boundary-pressure iteration, artificial buffer, or second flow solve.
"""

import math
from dataclasses import dataclass, replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_equilibrium_transport import PREquilibriumTransportNetwork, PREquilibriumTransportState
from .pr_layered_transport import LayeredTransportState, PRLayeredTransportNetwork
from .time_interval import interval_step_s, split_step_s


@dataclass(frozen=True)
class LayeredPlantState:
    transport: PREquilibriumTransportState
    layered: LayeredTransportState
    model_id: str

    @property
    def time_s(self):
        return self.transport.time_s

    def to_dict(self):
        return {
            "transport": self.transport.to_dict(),
            "layered": self.layered.to_dict(),
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"transport", "layered", "model_id"}:
            raise ValueError("Unknown mixed-profile plant state fields")
        return cls(
            PREquilibriumTransportState.from_dict(data["transport"]),
            LayeredTransportState.from_dict(data["layered"]),
            data["model_id"],
        )


class PRLayeredPlantNetwork:
    def __init__(self, transport, layered, boundary_nodes, *, maximum_head_fraction=1e-3):
        if not isinstance(transport, PREquilibriumTransportNetwork) or not isinstance(
            layered, PRLayeredTransportNetwork
        ):
            raise TypeError("Ordinary PR graph and layered pipe component required")
        self.transport, self.layered = transport, layered
        self.boundary_nodes = tuple(boundary_nodes)
        self.maximum_head_fraction = _number(
            maximum_head_fraction, "transverse head bound", positive=True
        )
        if len(self.boundary_nodes) != 2 or self.maximum_head_fraction > 0.01:
            raise ValueError("Two end bindings and head bound <=.01 required")
        if (
            transport.fluid.model_id != layered.fluid.model_id
            or transport.fluid.device != layered.fluid.device
            or transport.gravity_m_s2 != layered.gravity
        ):
            raise ValueError("Common EOS, device and gravity required")
        for side, name in zip((-1, 1), self.boundary_nodes):
            if name is None:
                continue
            if name not in transport.names:
                raise ValueError("Unknown profile boundary node")
            node = transport.nodes[transport.names.index(name)]
            cell = 0 if side == -1 else len(layered.columns) - 1
            center = layered._disk_bottom(cell, side) + layered.diameter * layered.cosine / 2
            if (
                node.kind != "vessel"
                or node.drift_flux is not None
                or abs(node.elevation_m - center) > 1e-8
            ):
                raise ValueError(
                    "Profile ends require real finite vessels at the physical disk centres"
                )
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_layered_plant_v1",
            "transport": self.transport.to_dict(),
            "layered": self.layered.to_dict(),
            "boundary_nodes": self.boundary_nodes,
            "maximum_head_fraction": self.maximum_head_fraction,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        raw = dict(data)
        if raw.pop("model", None) != "pr_layered_plant_v1":
            raise ValueError("Unknown layered plant model")
        raw["transport"] = PREquilibriumTransportNetwork.from_dict(raw["transport"], device=device)
        raw["layered"] = PRLayeredTransportNetwork.from_dict(raw["layered"], device=device)
        return cls(**raw)

    def initialize(self, transport_state, layered_state):
        state = LayeredPlantState(transport_state, layered_state, self.model_id)
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, LayeredPlantState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Layered plant state/model differs")
        if state.transport.time_s != state.layered.time_s:
            raise PropertyModelError("Plant component clocks differ")
        self.transport.inspect(state.transport)
        self.layered.inspect(state.layered)

    def _stocks(self, state):
        f = self.transport.fluid
        n = f.tensor([0.0] * len(f.names))
        energy = 0.0
        for node, (_, stock) in zip(self.transport.nodes, state.transport.nodes):
            n = n + f.tensor(stock.component_moles)
            energy += stock.total_energy_j
            if node.wall is not None:
                energy += node.wall.heat_capacity_j_k * stock.wall_temperature_k
        energy += self.transport.rotor_energy(state.transport)
        for cell in state.layered.cells:
            nn = f.tensor(cell.profile.component_moles)
            n = n + nn
            energy += (
                cell.total_energy_j
                - float((nn * f.molar_masses).sum().cpu())
                * self.layered.gravity
                * self.transport.reference_elevation_m
            )
        return n, energy

    def _ledger(self, old, new, external_energy):
        na, ea = self._stocks(old)
        nb, eb = self._stocks(new)
        residual = [*(nb - na).cpu().tolist(), eb - ea - external_energy]
        if max(map(abs, residual[:-1])) > 1e-8 or abs(residual[-1]) > 1e-5 + 1e-9 * max(
            abs(ea), abs(eb), 1.0
        ):
            raise ConservationError("Common graph/profile N/E ledger rejected")
        return residual

    def step(
        self,
        state,
        dt_s,
        *,
        openings=None,
        heat_w=None,
        utility_temperature_k=None,
        loss_coefficients=None,
        drive_torque_nm=None,
        restriction_openings=None,
        profile_heat_w=None,
        profile_utility_temperature_k=None,
    ):
        self.inspect(state)
        dt = _number(dt_s, "time step", positive=True)
        if state.time_s + dt <= state.time_s:
            raise ValueError("Step below clock resolution")
        t = self.transport
        opened = dict(t.default_openings)
        if openings is not None:
            if set(openings) - set(opened) or any(type(v) is not bool for v in openings.values()):
                raise ValueError("Unknown/nonboolean machine or link opening")
            opened.update(openings)
        heat = dict(heat_w or {})
        utility = dict(utility_temperature_k or {})
        losses = dict(loss_coefficients or {})
        if (
            set(heat) - set(t.names)
            or set(utility) - set(t.names)
            or set(losses) - {n.name for n in t.nodes if n.kind == "pipe_cell"}
        ):
            raise ValueError("Unknown heat, utility or resistance node")
        hh = [_number(heat.get(n, 0.0), "heat rate") for n in t.names]
        uu = [
            _number(utility.get(n, s.temperature_k), "utility temperature", positive=True)
            for n, (_, s) in zip(t.names, state.transport.nodes)
        ]
        if any(q and n not in utility for n, q in zip(t.names, hh)):
            raise ValueError("Nonzero heat requires explicit utility temperature")
        losses = {n: _number(v, "loss coefficient", nonnegative=True) for n, v in losses.items()}
        ph = self.layered._heat(profile_heat_w)
        pu = (
            None
            if profile_utility_temperature_k is None
            else tuple(
                _number(v, "profile utility temperature", positive=True)
                for v in profile_utility_temperature_k
            )
        )
        if (any(ph) and pu is None) or (pu is not None and len(pu) != len(ph)):
            raise ValueError("Profile heat requires one explicit utility temperature per cell")
        seeds = None
        for attempt in range(3):
            coupling = _LayeredCoupling(self, state, dt, opened, ph, pu, seeds)
            try:
                ordinary, report = t._step(
                    state.transport,
                    dt,
                    opened,
                    hh,
                    uu,
                    losses,
                    _coupling=coupling,
                    drive_torque_nm=drive_torque_nm,
                    restriction_openings=restriction_openings,
                )
                break
            except (HydraulicConvergenceError, PropertyModelError):
                if attempt == 2 or coupling.last_data is None:
                    raise
                seeds = self.layered.predict_seeds(state.layered, dt, coupling.last_data, heat_w=ph)
        linked = report["coupling"]
        profile = LayeredTransportState.from_dict(linked["state"])
        new = LayeredPlantState(ordinary, profile, self.model_id)
        self.inspect(new)
        heat_j = report["ambient_heat_j"] + report["external_heat_j"] + dt * math.fsum(ph)
        work_j = report["external_work_j"]
        ledger = self._ledger(state, new, heat_j + work_j)
        entropy = report["entropy_production_j_k"] + linked["entropy_production_j_k"]
        if entropy < -1e-6:
            raise ConservationError("Common graph/profile entropy ledger rejected")
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "transport": report,
            "profile": linked["report"],
            "vessel_support_force_n": linked["vessel_support_force_n"],
            "total_heat_j": heat_j,
            "total_work_j": work_j,
            "conservation_residuals": ledger,
            "entropy_production_j_k": entropy,
            "topology_retries": attempt,
        }

    def _suggested_step(self, state):
        return min(state.transport.next_step_s, state.layered.next_step_s)

    def _temporal_error(self, coarse, fine, relative_tolerance, interface_tolerance_m):
        return max(
            self.transport._error(coarse.transport, fine.transport),
            self.layered._error(
                coarse.layered,
                fine.layered,
                relative_tolerance=relative_tolerance,
                interface_tolerance_m=interface_tolerance_m,
            ),
        )

    def _adaptive_state(self, fine, nextdt):
        return replace(
            fine,
            transport=replace(fine.transport, next_step_s=nextdt),
            layered=replace(fine.layered, next_step_s=nextdt),
        )

    def advance(
        self,
        state,
        duration_s,
        *,
        relative_tolerance=1e-4,
        interface_tolerance_m=1e-5,
        maximum_attempts=1000,
        **commands,
    ):
        self.inspect(state)
        relative_tolerance = _number(relative_tolerance, "relative tolerance", positive=True)
        interface_tolerance_m = _number(interface_tolerance_m, "interface tolerance", positive=True)
        if type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Positive integer interval budget required")
        duration = _number(duration_s, "duration", positive=True)
        end = state.time_s + duration
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Interval below clock resolution")
        old = state
        steps = []
        dt = min(self._suggested_step(state), duration)
        for attempt in range(maximum_attempts):
            dt = interval_step_s(state.time_s, end, dt)
            one, two = split_step_s(state.time_s, dt)
            try:
                coarse, _ = self.step(state, dt, **commands)
                half, a = self.step(state, one, **commands)
                fine, b = self.step(half, two, **commands)
                error = self._temporal_error(
                    coarse, fine, relative_tolerance, interface_tolerance_m
                )
                if error > 1:
                    raise HydraulicConvergenceError("Common graph/profile temporal error exceeded")
            except (HydraulicConvergenceError, PropertyModelError, ConservationError):
                dt /= 2
                if dt < self.transport.minimum_step_s:
                    raise
                continue
            nextdt = min(
                self.transport.maximum_step_s,
                max(self.transport.minimum_step_s, dt * (1.5 if error < 0.25 else 1.0)),
            )
            state = self._adaptive_state(fine, nextdt)
            steps.extend((a, b))
            dt = nextdt
            if state.time_s >= end:
                break
        else:
            raise HydraulicConvergenceError("Common graph/profile interval budget exhausted")
        heat = math.fsum(s["total_heat_j"] for s in steps)
        work = math.fsum(s["total_work_j"] for s in steps)
        return {
            "state": state.to_dict(),
            "steps": steps,
            "total_heat_j": heat,
            "total_work_j": work,
            "conservation_residuals": self._ledger(old, state, heat + work),
            "attempts": attempt + 1,
        }


class _LayeredCoupling:
    def __init__(
        self,
        model,
        state,
        dt,
        openings,
        heat,
        utility,
        seeds,
        *,
        shared_active=None,
        shared_guesses=None,
    ):
        self.model, self.state, self.dt = model, state, dt
        self.heat, self.utility, self.seeds = heat, utility, seeds
        self.openings = openings
        self.last_data = None
        t = model.transport
        f = t.fluid
        self.external_ports = tuple(
            (t.names.index(n), None) for n in model.boundary_nodes if n is not None
        )
        if shared_active is not None:
            self.active = shared_active
            self.ordinary_guesses = shared_guesses
            return
        supports = [set(v.cpu().tolist()) for v in t._active(state.transport, openings)]
        native = {i for i, v in enumerate(state.layered.cells[0].profile.component_moles) if v > 0}
        for index, _ in self.external_ports:
            if supports[index] - native:
                raise PropertyModelError(
                    "Profile boundary cannot introduce a new component into the layered registry"
                )
            supports[index] |= native
        edges = [(a, b, True) for link, (a, b) in zip(t.links, t.ends) if openings[link.name]]
        edges += [
            (
                t.names.index(u.source),
                t.names.index(u.target),
                getattr(u, "signed_map", None) is not None,
            )
            for u in (*t.compressors, *t.pump_connections)
            if openings[u.name]
        ]
        edges += [
            (t.names.index(u.source), t.names.index(u.target), True)
            for u in t.restrictions
            if openings[u.name]
        ]
        for _ in t.nodes:
            for a, b, bidirectional in edges:
                union = supports[b] | supports[a]
                supports[b] = union
                if bidirectional:
                    supports[a] = set(union)
        for index, _ in self.external_ports:
            if supports[index] - native:
                raise PropertyModelError(
                    "Reachable profile inlet contains unsupported new components"
                )
        self.active = [torch.tensor(sorted(s), device=f.device, dtype=torch.long) for s in supports]
        self.ordinary_guesses = [f.tensor(s.component_moles) for _, s in state.transport.nodes]

    def prepare(self, y, lo, hi, base, offsets, dt, pattern):
        m = self.model
        t = m.transport
        base_pattern = t._sparsity(
            offsets,
            self.openings,
            base.reservoir_traces.entries,
            base.compressors.entries,
            base.hydrostatic_faces.entries,
            base.droplet_populations.entries,
        )
        pairs = sorted(set(zip(*base_pattern)) | base.additional_sparsity)
        base.sparsity = ([a for a, _ in pairs], [b for _, b in pairs])
        yy, ll, hh, joint = m.layered.joint_vessel_problem(
            t,
            (y, lo, hi, base, offsets),
            self.state.layered,
            dt,
            boundary_nodes=m.boundary_nodes,
            heat_w=self.heat,
            seed_overrides=self.seeds,
            maximum_head_fraction=m.maximum_head_fraction,
        )
        self.joint = joint
        self.cut = joint.ordinary_size

        def evaluate(z):
            residual, data = joint(z)
            result = data["ordinary"]
            result["layered_data"] = data["profile"]
            result["vessel_support_force_n"] = data["vessel_support_force_n"]
            return residual, result

        evaluate.linear_blocks = joint.linear_blocks

        evaluate.maximum_update = torch.cat(
            (base.maximum_update, yy.new_full((len(yy) - self.cut,), math.inf))
        )
        with torch.no_grad():
            self.last_data = joint(yy)[1]["profile"]
        return yy, ll, hh, evaluate, joint.sparsity if len(yy) > t.dense_limit else None

    def accept(self, y, result):
        m = self.model
        self.last_data = result["layered_data"]
        new, report = m.layered.accept_solved(
            self.state.layered,
            self.dt,
            y[self.cut :],
            self.last_data,
            heat_w=self.heat,
            external_ends=tuple(n is not None for n in m.boundary_nodes),
        )
        production = report["entropy_production_j_k"]
        if any(self.heat):
            production = (
                report["entropy_change_j_k"]
                - report["incoming_entropy_j_k"]
                - self.dt * math.fsum(q / t for q, t in zip(self.heat, self.utility))
            )
            if any(
                q * (t - c.profile.temperature_k) < -1e-10
                for q, t, c in zip(self.heat, self.utility, new.cells)
            ):
                raise PropertyModelError("Profile heat violates passive utility direction")
            if production < -1e-6:
                raise ConservationError("Profile heat entropy ledger rejected")
            report = {
                **report,
                "entropy_production_j_k": production,
                "heat_entropy_validated": True,
            }
        return {
            "state": new.to_dict(),
            "report": report,
            "entropy_production_j_k": production,
            "vessel_support_force_n": [float(v.cpu()) for v in result["vessel_support_force_n"]],
        }
