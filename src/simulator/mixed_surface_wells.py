"""Compose existing producer and water-injector blocks in one FIM Schur system."""

from copy import copy

import torch

try:
    from ..surface.liquid import PropertyModelError
except ImportError:
    from surface.liquid import PropertyModelError

from .blackoil_fim import BlackOilWells
from .blackoil_surface_coupling import SurfaceCoupledBlackOilWells
from .full_well_schur import FullWellSchur
from .water_injection_wells import WaterInjectionWells


class MixedSurfaceBlackOilWells(FullWellSchur, BlackOilWells):
    def __init__(self, manager, reservoir, fluid, dims, device, spec):
        super().__init__(manager, reservoir, fluid, dims, device)
        if self.n_w > 128 or self.conn_cell.numel() > 10000:
            raise ValueError(
                "Mixed dense surface block supports <=128 wells and <=10000 connections"
            )
        if spec.get("well_block_backend", "dense") != "dense":
            raise PropertyModelError(
                "Mixed injection/production currently requires dense well blocks, not a dense reservoir"
            )
        prod = [i for i, w in enumerate(self.wells) if w.type == "producer"]
        inj = [i for i, w in enumerate(self.wells) if w.type == "injector"]
        if not prod or not inj or len(prod) + len(inj) != self.n_w:
            raise ValueError("Mixed surface needs both producers and water injectors")
        if any(
            getattr(w, "ramping_days", 0) or getattr(w, "eff", 1.0) not in (None, 1.0)
            for w in self.wells
        ):
            raise PropertyModelError("Mixed surface requires unramped WEFAC=1 wells")
        production_spec = {k: v for k, v in spec.items() if k != "injection_network"}
        if set(production_spec["well_nodes"]) != {self.wells[i].name for i in prod}:
            raise ValueError("Production bindings must contain producers only")
        self.subwell_models = []
        self.partitions = []
        for indices, kind, config in [
            (prod, SurfaceCoupledBlackOilWells, production_spec),
            (inj, WaterInjectionWells, spec["injection_network"]),
        ]:
            subset = copy(manager)
            subset.wells = [self.wells[i] for i in indices]
            child = kind(subset, reservoir, fluid, dims, device, config)
            wi = torch.tensor(indices, dtype=torch.long, device=device)
            connection_map = torch.cat([torch.where(self.conn_well == i)[0] for i in indices])
            if not torch.equal(child.conn_cell, self.conn_cell[connection_map]):
                raise ValueError("Inconsistent mixed-well connection ordering")
            pair_lookup = {
                (int(c), int(d)): i for i, (c, d) in enumerate(zip(self.pair_c, self.pair_d))
            }
            pair_map = torch.tensor(
                [
                    pair_lookup[(int(connection_map[c]), int(connection_map[d]))]
                    for c, d in zip(child.pair_c, child.pair_d)
                ],
                dtype=torch.long,
                device=device,
            )
            self.subwell_models.append(child)
            self.partitions.append((wi, connection_map, pair_map))
        self.production, self.injection = self.subwell_models
        self.surface_well_indices = tuple(prod)
        self.injection_well_indices = tuple(inj)
        self.surface_network = self.production.surface_network
        self.network_backend = self.production.network_backend
        self.physical_surface = True
        self.surface_report = None

    def begin_step(self, v, d, t_days):
        self.step_targets = [None] * self.n_w
        self.step_mode = [None] * self.n_w
        self.p_bh = v.new_zeros(self.n_w)
        self.switches = 0
        self.well_switches = [0] * self.n_w
        for child, (wi, ci, pi) in zip(self.subwell_models, self.partitions):
            child.pvt = self.pvt
            child.flux_terms = self.flux_terms
            child.wellbore_density = self.wellbore_density
            child.pair_pos = self.pair_pos[pi]
            child.begin_step(v, d, t_days)
            self.p_bh[wi] = child.p_bh
            self.conn_head[ci] = child.conn_head
            self.conn_a[ci] = child.conn_a
            self.conn_eff[ci] = child.conn_eff
            for j, i in enumerate(wi.tolist()):
                self.step_targets[i] = child.step_targets[j]
                self.step_mode[i] = child.step_mode[j]

    def update_controls(self, p_bh, q_surf):
        # Each child assembles its exact rate/pressure/check-valve active set.
        return []

    def set_type(self, *args, **kwargs):
        raise PropertyModelError(
            "Mixed surface bindings have fixed well types; rebuild the topology to change them"
        )

    def assemble(self, R, J_vals, diag_map, v, d, p_bh):
        pieces = [
            child.assemble(R, J_vals, diag_map, v, d, p_bh[wi])
            for child, (wi, _, _) in zip(self.subwell_models, self.partitions)
        ]
        sys = {}
        for key in ["Cv", "Bv", "q", "q_o", "q_w", "q_g", "diag_pos"]:
            shape = (self.conn_cell.numel(), *pieces[0][key].shape[1:])
            sys[key] = pieces[0][key].new_zeros(shape)
            for part, (_, ci, _) in zip(pieces, self.partitions):
                sys[key][ci] = part[key]
        for key in [
            "D",
            "r_w",
            "q_surf",
            "is_rate",
            "is_thp",
            "is_stopped",
            "component_rates",
            "target",
        ]:
            sys[key] = pieces[0][key].new_zeros((self.n_w, *pieces[0][key].shape[1:]))
            for part, (wi, _, _) in zip(pieces, self.partitions):
                sys[key][wi] = part[key]
        sys["B_full"] = p_bh.new_zeros((self.n_w, self.conn_cell.numel(), 3))
        sys["D_full"] = p_bh.new_zeros((self.n_w, self.n_w))
        for child, part, (wi, ci, _) in zip(self.subwell_models, pieces, self.partitions):
            sys["B_full"][wi[:, None], ci[None, :]] = part["B_full"]
            sys["D_full"][wi[:, None], wi[None, :]] = part["D_full"]
            for j, i in enumerate(wi.tolist()):
                self.step_mode[i] = child.step_mode[j]
        self._Q = sys["component_rates"]
        self._lin = (sys["q"], sys["Cv"], p_bh.clone())
        self.factor_well_block(sys)
        self.surface_report = {
            **self.production.surface_report,
            "wells": {
                **self.production.surface_report["wells"],
                **self.injection.injection_report["wells"],
            },
            "injection_network": self.injection.injection_report,
        }
        return sys

    def commit(self, p_bh):
        for child, (wi, _, _) in zip(self.subwell_models, self.partitions):
            child.commit(p_bh[wi])
        self.p_bh = p_bh.detach().clone()
        self.mode = list(self.step_mode)

    def set_surface_controls(self, **controls):
        self.production.set_surface_controls(**controls)
        self.surface_network = self.production.surface_network

    def set_surface_boundary_pressures(self, pressures_pa):
        self.production.set_surface_boundary_pressures(pressures_pa)
        self.surface_network = self.production.surface_network
