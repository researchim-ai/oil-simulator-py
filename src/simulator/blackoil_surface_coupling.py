"""Implicit black-oil FIM/production-tree coupling for strictly producing wells.

Reservoir/Schur on CPU or CUDA; network reference on CPU or PVT/quadrature/AD
on the reservoir device. Host tree/root orchestration remains. One PVT region,
no VFP/MSW/groups, no perf backflow or zero-flow holdup.
Network source coordinates are conserved stock-component masses in kg/s.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .blackoil_fim import I_P, BlackOilWells
from .device_exchange import HostReturnPacket
from .fim_wells import DAY
from .full_well_schur import FullWellSchur

try:
    from ..surface import (
        BlackOilPressureControl,
        BlackOilProductionNetwork,
        ComponentStream,
        PropertyModelError,
    )
    from ..surface.liquid import HydraulicConvergenceError
except ImportError:
    from surface import (
        BlackOilPressureControl,
        BlackOilProductionNetwork,
        ComponentStream,
        PropertyModelError,
    )
    from surface.liquid import HydraulicConvergenceError


class SurfaceCoupledBlackOilWells(FullWellSchur, BlackOilWells):
    def __init__(self, well_manager, reservoir, fluid, dims, device, spec):
        super().__init__(well_manager, reservoir, fluid, dims, device)
        if set(spec) - {"network", "well_nodes", "network_backend", "controls", "pressure_control"}:
            raise ValueError("Unknown physical_surface_network fields")
        self.network_backend = spec.get("network_backend", "cpu_reference")
        if self.network_backend not in ("cpu_reference", "torch_device"):
            raise ValueError("network_backend must be cpu_reference or torch_device")
        self.surface_network = BlackOilProductionNetwork.from_dict(spec["network"])
        self.surface_pressure_control = None
        self.set_surface_controls(**spec.get("controls", {}))
        self.set_surface_pressure_control(spec.get("pressure_control"))
        if (
            self.surface_pressure_control is not None
            and self.surface_pressure_control.valve
            in spec.get("controls", {}).get("valve_openings", {})
        ):
            raise ValueError("Manual opening conflicts with active pressure regulator")
        if self.network_backend == "torch_device":
            self.surface_network.model = self.surface_network.model.to(self.device)
        if getattr(fluid, "pvt", None) is None or getattr(fluid, "blackoil_pvt", None) is not None:
            raise PropertyModelError(
                "Physical black-oil coupling requires one explicit PVTTable region"
            )
        model = self.surface_network.model
        for key in ("P", "Bo", "Bw", "Bg", "mu_o", "mu_w", "mu_g", "Rs", "Rv", "oil_rs"):
            if not np.array_equal(getattr(model.pvt, key), getattr(fluid.pvt, key)):
                raise PropertyModelError("Reservoir and physical surface must use identical PVT")
        if set(model.pvt.oil_tables) != set(fluid.pvt.oil_tables):
            raise PropertyModelError("Reservoir and surface PVTO differ")
        for rs, table in model.pvt.oil_tables.items():
            if any(
                not np.array_equal(v, fluid.pvt.oil_tables[rs].get(k)) for k, v in table.items()
            ):
                raise PropertyModelError("Reservoir and surface PVTO differ")
        if not np.array_equal(model.rho.detach().cpu().numpy(), self.rho_sc.detach().cpu().numpy()):
            raise PropertyModelError("Reservoir and surface stock densities differ")
        names = {w.name: i for i, w in enumerate(self.wells)}
        bindings = spec["well_nodes"]
        if not isinstance(bindings, dict) or set(bindings) != set(names) or not names:
            raise ValueError("First black-oil physical coupling requires all wells bound producers")
        self.surface_nodes = tuple(bindings[w.name] for w in self.wells)
        self.surface_well_indices = tuple(range(self.n_w))
        if self.n_w > 128 or self.conn_cell.numel() > 10000:
            raise ValueError("Dense physical coupling supports <=128 wells and <=10000 connections")
        for i, (w, node) in enumerate(zip(self.wells, self.surface_nodes)):
            if w.type != "producer" or w.control_type != "rate" or getattr(w, "stopped", False):
                raise PropertyModelError(
                    "Physical black-oil binding requires active rate-controlled producers"
                )
            if node not in self.surface_network.by_name or node == self.surface_network.root:
                raise ValueError("Bound producer must connect to a free production node")
            if bool((self.conn_dz[self.conn_well == i].abs() > 1e-6).any()):
                raise PropertyModelError(
                    "Bound black-oil perforations must lie at BHP reference elevation; distributed wellbore hydrostatics are not yet implicit"
                )
            if not math.isclose(
                self.surface_network.by_name[node].elevation_m,
                -self.ref_depth[i],
                rel_tol=0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "Production node elevation must equal negative BHP reference depth"
                )
            if (
                getattr(w, "msw", None) is not None
                or getattr(w, "vfp_table", None) is not None
                or getattr(w, "thp_limit", None) is not None
            ):
                raise PropertyModelError(
                    "Physical black-oil pipes cannot be combined with MSW/VFP/THP"
                )
        self.physical_surface = True
        self.surface_report = None
        self._surface_flow_linearization = None
        self._surface_return_packet = HostReturnPacket()
        self._surface_eye = torch.eye(self.n_w, dtype=torch.float64, device=self.device)
        self._surface_connection_index = torch.arange(self.conn_cell.numel(), device=self.device)

    def set_surface_controls(self, valve_openings=None):
        """Set branch openings between steps; validation is atomic.

        Zero opening is rejected until a static multiphase holdup model exists.
        """
        control = self.surface_pressure_control
        if control is not None and valve_openings and control.valve in valve_openings:
            raise ValueError("Manual opening conflicts with active pressure regulator")
        effective = self.surface_network.with_controls(valve_openings)
        self.surface_network = effective

    def set_surface_pressure_control(self, control):
        if isinstance(control, dict):
            control = BlackOilPressureControl(**control)
        if control is not None:
            if not isinstance(control, BlackOilPressureControl):
                raise TypeError("Expected BlackOilPressureControl or None")
            control.validate_network(self.surface_network)
        self.surface_pressure_control = control

    def set_surface_boundary_pressures(self, pressures_pa):
        root = self.surface_network.root
        if set(pressures_pa) - {root}:
            raise ValueError("Only the existing gathering receiver pressure may be changed")
        spec = self.surface_network.to_dict()
        for node in spec["nodes"]:
            if node["name"] == root:
                node["pressure_pa"] = pressures_pa.get(root, node["pressure_pa"])
        self.surface_network = BlackOilProductionNetwork.from_dict(spec)
        if self.network_backend == "torch_device":
            self.surface_network.model = self.surface_network.model.to(self.device)

    def _conn_flows(self, v, d, p_bh):
        self._mix = None
        self._producer_mix = {}
        q, dq, cv, _dpp, _wi, _pwc = self._production_connection_flows(v, d, p_bh)
        result = (q, dq, cv)
        self._surface_flow_linearization = result
        return result

    def begin_step(self, v, d, t_days):
        if self.network is not None or self.groups is not None or self.liftopt is not None:
            raise PropertyModelError(
                "Physical gathering cannot be combined with VFP network/groups/gas lift"
            )
        if any(w.type != "producer" for w in self.wells):
            raise PropertyModelError(
                "Physical black-oil binding requires active rate-controlled producers"
            )
        previous = self.p_bh.clone() if self.surface_report is not None else None
        if any(getattr(w, "eff", 1.0) not in (None, 1.0) for w in self.wells):
            raise PropertyModelError("Physical black-oil coupling currently requires WEFAC=1")
        super().begin_step(v, d, t_days)
        if any(
            ctrl != "rate" or not math.isfinite(target) or target <= 0
            for ctrl, target in self.step_targets
        ):
            raise PropertyModelError(
                "Physical black-oil coupling requires positive rate upper bounds"
            )
        for i in self.surface_well_indices:
            ceiling = (
                v[I_P][self.conn_cell[self.conn_well == i]] - self.conn_head[self.conn_well == i]
            ).min()
            # Start in the producing domain; no artificial source is supplied
            # to a zero-flow network to invent its static phase distribution.
            self.p_bh[i] = (
                0.95 * float(ceiling)
                if previous is None
                else min(float(previous[i]), 0.95 * float(ceiling))
            )

    def update_controls(self, p_bh, q_surf):
        # The exact network/rate/BHP active set is assembled simultaneously.
        return []

    def assemble(self, R, J_vals, diag_map, v, d, p_bh):
        lower, upper = self.surface_network.model.pressure_bounds_pa
        bad_pressure = ((v[I_P : I_P + 3] < lower) | (v[I_P : I_P + 3] > upper)).any()
        q, dq, Cv, jd = self.assemble_connections(R, J_vals, diag_map, v, d, p_bh)
        cw = self.conn_well
        rates = q.new_zeros((self.n_w, 3)).index_add_(0, cw, q)
        q_surf = q.new_zeros(self.n_w).index_add_(0, cw, (self.conn_a * q).sum(1))
        flags = torch.stack(
            [bad_pressure, (q < 0).any(), (rates.sum(1) <= 0).any(), self.is_inj.any()]
        ).to(q.dtype)
        # All admission statuses and well values cross the device boundary
        # together. No per-flag item() or separate per-quantity D2H copies.
        host = torch.cat([flags, rates.reshape(-1), p_bh, q_surf]).detach().cpu().numpy()
        if host[0]:
            raise PropertyModelError("Coupled reservoir phase pressure outside shared PVT coverage")
        if host[3]:
            raise PropertyModelError(
                "Physical black-oil binding requires active rate-controlled producers"
            )
        if host[1]:
            # Invalid-domain slow path retains the previous crossflow/zero
            # flow error classification. The valid producing path needs no
            # separate injection/backflow GPU scalar decisions.
            legacy, _, _ = super()._conn_flows(v, d, p_bh)
            if self._producer_mix or self._mix is not None:
                raise PropertyModelError(
                    "Physical black-oil binding does not yet support perforation crossflow"
                )
            if bool((legacy < 0).any()):
                raise HydraulicConvergenceError(
                    "Black-oil physical Newton iterate leaves producing-flow domain"
                )
            raise PropertyModelError("Zero-flow black-oil branches need a static holdup model")
        if host[2]:
            raise PropertyModelError("Zero-flow black-oil branches need a static holdup model")
        host_rates = host[4 : 4 + 3 * self.n_w].reshape(self.n_w, 3)
        host_bhp = host[4 + 3 * self.n_w : 4 + 4 * self.n_w]
        host_control = host[4 + 4 * self.n_w :]
        self._Q = rates / self.rho_sc
        self._lin = (q, Cv, p_bh.clone())
        sys = {
            "Cv": Cv * self.conn_eff[:, None],
            "q": q,
            "q_o": q[:, 0],
            "q_w": q[:, 1],
            "q_g": q[:, 2],
            "q_surf": q_surf,
            "component_rates": self._Q,
            "diag_pos": jd,
            "is_thp": torch.zeros(self.n_w, dtype=torch.bool, device=q.device),
            "is_stopped": torch.zeros(self.n_w, dtype=torch.bool, device=q.device),
        }
        sources = {}
        for node, rate in zip(self.surface_nodes, host_rates):
            stream = ComponentStream.from_mass_rates(*rate, self.surface_network.model.densities)
            sources[node] = ComponentStream.mix([sources.get(node, ComponentStream()), stream])
        control = self.surface_pressure_control
        result = (
            self.surface_network.solve(sources)
            if control is None
            else control.solve(self.surface_network, sources)
        )
        host_K = np.array(
            [
                [
                    list(result["pressure_derivatives_per_kg_s"][node][source].values())
                    for source in self.surface_nodes
                ]
                for node in self.surface_nodes
            ]
        )
        required = []
        upper = []
        is_rate = []
        is_surface = []
        states = {}
        for i, node in enumerate(self.surface_nodes):
            network_p = result["node_pressures_pa"][node]
            floor = self.bhp_limit(self.wells[i])
            target = max(network_p, floor)
            cap = self.step_targets[i][1]
            rate = host_control[i]
            trial = rate + max(abs(rate), 0.01 / DAY) / max(target, 1e5) * (host_bhp[i] - target)
            active = bool(trial >= cap)
            mode = "rate" if active else ("surface" if network_p >= floor else "bhp_limit")
            self.step_mode[i] = mode
            required.append(target)
            upper.append(cap)
            is_rate.append(active)
            is_surface.append(mode == "surface")
            states[self.wells[i].name] = {
                "node": node,
                "mode": mode,
                "bhp_pa": float(host_bhp[i]),
                "network_reference_pressure_pa": network_p,
                "component_mass_rates_kg_s": host_rates[i].tolist(),
                "standard_control_rate_m3_s": float(rate),
            }
        # One H2D packet for the dense derivative and all active-row data.
        exchange = getattr(self, "_surface_return_packet", None)
        if exchange is None:
            exchange = self._surface_return_packet = HostReturnPacket()
        packed = exchange.upload(
            np.concatenate([host_K.reshape(-1), required, upper, is_rate, is_surface]), q
        )
        offset = 3 * self.n_w * self.n_w
        K = packed[:offset].view(self.n_w, self.n_w, 3)
        targets = packed[offset : offset + self.n_w]
        caps = packed[offset + self.n_w : offset + 2 * self.n_w]
        rate_mask = packed[offset + 2 * self.n_w : offset + 3 * self.n_w].bool()
        surface_mask = packed[offset + 3 * self.n_w :].bool()
        grad_conn = K[:, cw, :]
        Bnet = -(grad_conn[:, :, :, None] * dq[None, :, :, :]).sum(2)
        eye = self._surface_eye
        Dnet = eye.clone().index_add_(1, cw, -(grad_conn * Cv[None, :, :]).sum(2))
        Brate = torch.zeros_like(Bnet)
        B_rate = (self.conn_a[:, :, None] * dq).sum(1)
        Brate[cw, self._surface_connection_index] = B_rate
        Drate = torch.diag(q.new_zeros(self.n_w).index_add_(0, cw, (self.conn_a * Cv).sum(1)))
        sys["Bv"] = torch.where(rate_mask[cw, None], B_rate, 0.0)
        B = torch.where(
            rate_mask[:, None, None], Brate, torch.where(surface_mask[:, None, None], Bnet, 0.0)
        )
        D = torch.where(rate_mask[:, None], Drate, torch.where(surface_mask[:, None], Dnet, eye))
        sys["r_w"] = torch.where(rate_mask, sys["q_surf"] - caps, p_bh - targets)
        sys["is_rate"] = rate_mask
        sys["target"] = torch.where(rate_mask, caps, targets)
        sys["B_full"] = B
        sys["D_full"] = D
        sys["D"] = torch.diagonal(D)
        # Its status joins the convergence packet. Only a requested linear
        # solve checks it, preserving converged-step acceptance semantics.
        self.factor_well_block(sys)
        self.surface_report = {
            "network": result,
            "wells": states,
            "execution": {
                "reservoir_device": str(q.device),
                "network_device": str(self.surface_network.model.rho.device),
                "backend": self.network_backend,
            },
        }
        return sys

    def commit(self, p_bh):
        super().commit(p_bh)
        for well in self.wells:
            state = self.surface_report["wells"][well.name]
            well.surface_mode = state["mode"]
            well.surface_reference_pressure_mpa = state["network_reference_pressure_pa"] / 1e6
