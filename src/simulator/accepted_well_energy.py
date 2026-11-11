"""Audited caloric processing of ACTUAL accepted FIM well material.

The reservoir remains isothermal. An explicit assay and thermal boundary
supply energy to this downstream PR process; no reservoir energy equation or
PR capacity feedback is implied. Failures belong to the owning FIM transaction.
"""

import math
from copy import deepcopy

from .thermal_surface_boundary import characterize_well_thermal_boundaries

try:
    from ..surface.equilibrium_process import run_equilibrium_process
    from ..surface.liquid import _number
    from ..surface.operation import _digest
except ImportError:
    from surface.equilibrium_process import run_equilibrium_process
    from surface.liquid import _number
    from surface.operation import _digest


class AcceptedWellEnergyProcess:
    def __init__(self, fluid, assay, thermal_conditions, transfers, mixer):
        self.fluid, self.assay = fluid, assay
        self.thermal_conditions = deepcopy(thermal_conditions)
        self.transfers, self.mixer = deepcopy(transfers), deepcopy(mixer)
        if not self.thermal_conditions or set(self.thermal_conditions) != set(self.transfers):
            raise ValueError("Thermal boundaries and transfers must bind the same wells")
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        from dataclasses import asdict

        return {
            "model": "accepted_well_energy_process_v1",
            "fluid": self.fluid.to_dict(),
            "assay": asdict(self.assay),
            "thermal_conditions": self.thermal_conditions,
            "transfers": self.transfers,
            "mixer": self.mixer,
        }

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        try:
            from ..surface.component_energy import StockComponentAssay
            from ..surface.equilibrium_properties import EquilibriumFluid
        except ImportError:
            from surface.component_energy import StockComponentAssay
            from surface.equilibrium_properties import EquilibriumFluid
        if (
            set(raw) != {"model", "fluid", "assay", "thermal_conditions", "transfers", "mixer"}
            or raw["model"] != "accepted_well_energy_process_v1"
        ):
            raise ValueError("Unknown accepted-well energy model")
        fluid = EquilibriumFluid.from_dict(raw["fluid"], device=device)
        assay = StockComponentAssay(**raw["assay"])
        if assay.fluid_id != fluid.model_id or assay.component_names != fluid.names:
            raise ValueError("Caloric assay and EOS differ")
        return cls(fluid, assay, raw["thermal_conditions"], raw["transfers"], raw["mixer"])

    def run(self, well_reports, duration_s, previous=None, *, controls=None):
        if _digest(self.to_dict()) != self.model_id:
            raise ValueError("Changed accepted-well energy model")
        dt = _number(duration_s, "accepted caloric interval", positive=True)
        if previous is not None and previous["model_id"] != self.model_id:
            raise ValueError("Energy continuation model differs")
        controls = {} if controls is None else deepcopy(controls)
        if set(controls) - {"thermal_conditions", "transfers", "mixer"}:
            raise ValueError("Unknown caloric process controls")
        boundary = characterize_well_thermal_boundaries(
            well_reports,
            controls.get("thermal_conditions", self.thermal_conditions),
            self.fluid,
            self.assay,
        )
        process = run_equilibrium_process(
            self.fluid,
            boundary["streams"],
            controls.get("transfers", self.transfers),
            controls.get("mixer", self.mixer),
        )
        nc = len(self.fluid.names)
        inputs = [
            dt
            * math.fsum(
                stream.component_molar_rates_mol_s[j] for stream in boundary["streams"].values()
            )
            for j in range(nc)
        ]
        outputs = [
            dt
            * math.fsum(stream["component_molar_rates_mol_s"][j] for stream in process["products"])
            for j in range(nc)
        ]
        ein = dt * boundary["total_energy_rate_w"]
        eout = dt * math.fsum(stream["total_energy_rate_w"] for stream in process["products"])
        utility = dt * (process["balances"]["heat_w"] + process["balances"]["shaft_power_w"])
        old = {} if previous is None else previous["cumulative"]
        cumulative = {
            "elapsed_s": math.fsum((old.get("elapsed_s", 0.0), dt)),
            "inlet_component_moles": [
                math.fsum((old.get("inlet_component_moles", [0.0] * nc)[j], inputs[j]))
                for j in range(nc)
            ],
            "export_component_moles": [
                math.fsum((old.get("export_component_moles", [0.0] * nc)[j], outputs[j]))
                for j in range(nc)
            ],
            "thermal_boundary_energy_j": math.fsum(
                (old.get("thermal_boundary_energy_j", 0.0), ein)
            ),
            "utility_energy_j": math.fsum((old.get("utility_energy_j", 0.0), utility)),
            "export_energy_j": math.fsum((old.get("export_energy_j", 0.0), eout)),
        }
        cumulative["component_residuals_mol"] = [
            a - b
            for a, b in zip(
                cumulative["inlet_component_moles"], cumulative["export_component_moles"]
            )
        ]
        cumulative["energy_residual_j"] = math.fsum(
            (
                cumulative["thermal_boundary_energy_j"],
                cumulative["utility_energy_j"],
                -cumulative["export_energy_j"],
            )
        )
        return {
            "model_id": self.model_id,
            "dt_s": dt,
            "applied_controls": controls,
            "boundaries": boundary["boundaries"],
            "process": process,
            "cumulative": cumulative,
            "reservoir_energy_equation_solved": False,
            "pvt_eos_characterization_calibrated": False,
            "pressure_feedback_from_caloric_process": False,
            "scope": "Accepted FIM well material mapped by explicit assay into a downstream PR caloric process; energy supplied by declared boundaries. PVT capacity feedback remains in the existing facility; this caloric process has no separate PR capacity feedback.",
        }
