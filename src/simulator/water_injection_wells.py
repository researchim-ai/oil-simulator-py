"""Water injector rows coupled to a distribution tree, stock-mass coordinates."""

import math

import torch

try:
    from ..surface.liquid import HydraulicConvergenceError, PropertyModelError
    from ..surface.water_injection import WaterInjectionTree
except ImportError:
    from surface.liquid import HydraulicConvergenceError, PropertyModelError
    from surface.water_injection import WaterInjectionTree

from .blackoil_fim import I_P, BlackOilWells
from .full_well_schur import FullWellSchur
from .surface_well_blocks import assemble_surface_blocks


class WaterInjectionWells(FullWellSchur, BlackOilWells):
    def __init__(self, manager, reservoir, fluid, dims, device, spec):
        super().__init__(manager, reservoir, fluid, dims, device)
        if set(spec) - {"network", "well_nodes", "completion_model"}:
            raise ValueError("Unknown injection coupling fields")
        if spec.get("completion_model") != "nonreturn":
            raise PropertyModelError(
                "First injection coupling requires explicit completion_model=nonreturn (check valves at perforations)"
            )
        self.tree = WaterInjectionTree(spec["network"])
        bindings = spec["well_nodes"]
        if (
            set(bindings) != {w.name for w in self.wells}
            or set(bindings.values()) != set(self.tree.leaves)
            or len(bindings) != len(set(bindings.values()))
        ):
            raise ValueError("Bind each water injector to one unique injection-tree leaf")
        if not math.isclose(self.tree.fluid.density_kg_m3, fluid.rho_water_ref, rel_tol=1e-12):
            raise PropertyModelError("Injection water stock density differs from reservoir")
        if any(abs(float(x) - 1) > 1e-12 for x in fluid.pvt.Bw):
            raise PropertyModelError(
                "First water injection tree requires incompressible PVT water Bw=1"
            )
        self.nodes = tuple(bindings[w.name] for w in self.wells)
        for i, w in enumerate(self.wells):
            if (
                w.type != "injector"
                or w.injected_phase != "water"
                or w.control_type not in ("rate", "shut")
                or getattr(w, "msw", None) is not None
                or getattr(w, "vfp_table", None) is not None
            ):
                raise PropertyModelError(
                    "Water tree supports water-injection rate caps/shut and BHP maxima, without MSW/VFP"
                )
            if bool((self.conn_dz[self.conn_well == i].abs() > 1e-6).any()):
                raise PropertyModelError(
                    "Injection perforations must be at the BHP reference elevation"
                )
            if not math.isclose(
                self.tree.by_name[self.nodes[i]].elevation_m, -self.ref_depth[i], abs_tol=1e-6
            ):
                raise PropertyModelError(
                    "Injection terminal elevation must equal negative BHP reference depth"
                )
        self.enabled = True
        self.speed_ratios = {}
        self.allow_infeasible_trial = False
        self._nominal_schedule = self.conn_sched.clone()
        self.injection_report = None
        # Each completion is explicitly non-return: there is no incoming
        # reservoir stream to mix back into other perforations.
        self.wellbore_mixing = False

    def _conn_flows(self, v, d, p_bh):
        # Explicit perforation check-valve law in the Newton residual.
        # Blocked valves carry zero component flux AND zero derivatives.
        q, dq, cv = super()._conn_flows(v, d, p_bh)
        opened = (
            (v[I_P][self.conn_cell] <= p_bh[self.conn_well] + self.conn_head)
            & (self.conn_sched > 0)
            & (self.conn_open > 0)
        )
        self._injection_valve_open = opened
        self._mix = None
        self._producer_mix = {}
        return (
            torch.where(opened[:, None], q, 0.0),
            torch.where(opened[:, None, None], dq, 0.0),
            torch.where(opened[:, None], cv, 0.0),
        )

    def begin_step(self, v, d, t_days):
        self.conn_sched = self._nominal_schedule.clone()
        self.conn_open = self.conn_sched.new_tensor([w.control_type != "shut" for w in self.wells])[
            self.conn_well
        ]
        super().begin_step(v, d, t_days)
        self.closed = [
            not self.enabled or ctrl == "shut" or (ctrl == "rate" and value == 0)
            for ctrl, value in self.step_targets
        ]
        for i, closed in enumerate(self.closed):
            if closed:
                self.step_targets[i] = ("shut", 0.0)
        self.conn_sched = (
            self._nominal_schedule
            * self.p_bh.new_tensor([not c for c in self.closed])[self.conn_well]
        )
        self.closed_bhp = self.p_bh.clone()

    def assemble(self, R, J_vals, diag_map, v, d, p_bh):
        q, dq, Cv, jd = self.assemble_connections(R, J_vals, diag_map, v, d, p_bh)
        if self._mix is not None or self._producer_mix:
            raise PropertyModelError("Injection-tree coupling does not admit perforation crossflow")
        rates = q.new_zeros((self.n_w, 3)).index_add_(0, self.conn_well, q)
        host = rates.detach().cpu().tolist()
        if any(abs(a) > 1e-10 or abs(c) > 1e-10 or b > 1e-10 for a, b, c in host):
            raise HydraulicConvergenceError(
                "Injection Newton trial leaves pure-water forward domain"
            )
        amounts = {node: -row[1] for node, row in zip(self.nodes, host)}
        result = self.tree.evaluate(amounts, self.device, speed_ratios=self.speed_ratios)
        if not self.enabled:
            result.update(shaft_power_w=0.0, feasible=True, violations=[], isolated=True)
            result["node_pressures_pa"] = {
                n: (p if n == self.tree.root else None)
                for n, p in result["node_pressures_pa"].items()
            }
            result["branches"] = {
                name: {"flow_m3_s": 0.0, "mass_rate_kg_s": 0.0, "isolated": True}
                for name in result["branches"]
            }
        q_control = (self.conn_a * q).sum(1)
        qs = q.new_zeros(self.n_w).index_add_(0, self.conn_well, q_control)
        targets = []
        caps = []
        is_rate = []
        is_surface = []
        states = {}
        p_host = p_bh.detach().cpu().tolist()
        qr = qs.detach().cpu().tolist()
        for i, w in enumerate(self.wells):
            cap = self.step_targets[i][1]
            network_p = result["node_pressures_pa"][self.nodes[i]]
            maximum = self.bhp_limit(w)
            if self.closed[i]:
                target = float(self.closed_bhp[i])
                cap = 0.0
                mode = "shut"
                active = False
            else:
                target = min(network_p, maximum)
                shadow = qr[i] - max(abs(qr[i]), 0.01 / 86400) / max(target, 1e5) * (
                    p_host[i] - target
                )
                active = shadow >= cap
                blocked = not bool(self._injection_valve_open[self.conn_well == i].any())
                if blocked:
                    active, mode = False, "no_head"
                else:
                    mode = (
                        "rate" if active else ("surface" if network_p <= maximum else "bhp_limit")
                    )
            self.step_mode[i] = mode
            targets.append(target)
            caps.append(cap)
            is_rate.append(active)
            is_surface.append(mode == "surface" or (mode == "no_head" and network_p <= maximum))
            states[w.name] = {
                "node": self.nodes[i],
                "mode": mode,
                "bhp_pa": p_host[i],
                "bhp_status": "isolated_gauge" if self.closed[i] else "connected_water_column",
                "network_reference_pressure_pa": network_p,
                "component_mass_rates_kg_s": host[i],
                "standard_control_rate_m3_s": qr[i],
                "throttle_dissipation_w": max(
                    0.0, ((network_p or p_host[i]) - p_host[i]) * max(qr[i], 0.0)
                ),
            }
        # q_water is signed reservoir withdrawal (negative for injection).
        leaf_index = [result["leaves"].index(n) for n in self.nodes]
        K = q.new_zeros((self.n_w, self.n_w, 3))
        K[:, :, 1] = result["pressure_derivatives_pa_per_kg_s"][leaf_index][:, leaf_index] * -1
        # Tree tangent is with respect to positive DELIVERED mass; convert
        # to signed reservoir withdrawal before the existing Schur assembler.
        rm = torch.tensor(is_rate, device=q.device)
        sm = torch.tensor(is_surface, device=q.device)
        B, D, Brate = assemble_surface_blocks(K, self.conn_well, dq, Cv, self.conn_a, rm, sm)
        targets = q.new_tensor(targets)
        caps = q.new_tensor(caps)
        self._Q = rates / self.rho_sc
        self._lin = (q, Cv, p_bh.clone())
        self.injection_report = {
            k: value for k, value in result.items() if k != "pressure_derivatives_pa_per_kg_s"
        }
        self.injection_report["wells"] = states
        self.injection_report["perforation_valves_open"] = (
            self._injection_valve_open.detach().cpu().tolist()
        )
        self.injection_report["well_boundary_model"] = (
            "Ideal rate choke and explicitly configured non-return perforation valves"
        )
        return {
            "Cv": Cv,
            "Bv": torch.where(rm[self.conn_well, None], Brate, 0.0),
            "B_full": B,
            "D_full": D,
            "D": D.diagonal(),
            "r_w": torch.where(rm, qs - caps, p_bh - targets),
            "q_surf": qs,
            "is_rate": rm,
            "is_thp": torch.zeros_like(rm),
            "is_stopped": q.new_tensor(self.closed, dtype=torch.bool),
            "component_rates": self._Q,
            "target": torch.where(rm, caps, targets),
            "q": q,
            "q_o": q[:, 0],
            "q_w": q[:, 1],
            "q_g": q[:, 2],
            "diag_pos": jd,
        }

    def commit(self, p_bh):
        if not self.allow_infeasible_trial and not self.injection_report["feasible"]:
            raise HydraulicConvergenceError("Injection equipment envelope rejected")
        super().commit(p_bh)
