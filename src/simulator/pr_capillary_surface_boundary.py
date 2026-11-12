"""Accepted FIM packets into existing capillary/gravity surface suboperators."""

from dataclasses import replace

from .pr_gravity_transport_boundary import PRGravityTransportBoundary

try:
    from ..surface.pr_capillary_surface_plant import PRCapillarySurfacePlant
except ImportError:
    from surface.pr_capillary_surface_plant import PRCapillarySurfacePlant


class PRCapillarySurfaceBoundary(PRGravityTransportBoundary):
    model_tag = "dynamic_pr_capillary_surface_fim_v1"
    scope = (
        "Producer-only two-way FIM coupling to existing capillary lines and an open "
        "PR gravitational pipe/equipment graph sharing finite vessels. First-order "
        "Lie splitting with whole-system full/two-half error control and endpoint "
        "surface-work closure; not a monolithic all-physics Newton. Explicit stock "
        "assay and well thermal boundaries; no thermal reservoir, dynamic injection "
        "coupling or experimental calibration inferred."
    )

    def transaction(self):
        return self.model.capillary.cached_inspections()

    def totals(self, state):
        return self.model.totals(state)

    def build(self, streams, height, outlet_openings, connection_openings):
        surface = super().build(streams, height, outlet_openings, connection_openings)
        parameters = self.model.to_dict()
        return PRCapillarySurfacePlant(
            surface,
            self.model.capillary,
            **{k: v for k, v in parameters.items() if k not in ("model", "surface", "capillary")},
        )

    def rebind(self, state, target):
        # Accepted streams change plant identity, never any physical submodel.
        # Reject cross-graph rebinding instead of concealing mismatched stocks.
        if (
            target.graph.model_id != self.model.graph.model_id
            or target.capillary.model_id != self.model.capillary.model_id
        ):
            raise ValueError("Cannot rebind a different physical capillary plant")
        rebound = replace(state, model_id=target.model_id)
        target.inspect(rebound)
        return rebound

    def step(self, model, state, dt, connection_openings):
        end, raw = model.step(state, dt, openings=connection_openings)
        flow = self.flow_report(model.surface, dt, raw["surface"])
        # Keep complete accepted suboperator histories for independent replay.
        flow.pop("gravity_model")
        flow.pop("gravity_report")
        flow.update(
            plant_model=model.to_dict(),
            plant_report=raw,
            external_energy_j=raw["total_work_j"],
            entropy_production_j_k=raw["entropy_production_j_k"],
        )
        return end, flow

    def temporal_error(self, coarse, fine, control, height):
        return max(
            super().temporal_error(coarse, fine, control, height),
            self.model.error(coarse, fine),
        )
