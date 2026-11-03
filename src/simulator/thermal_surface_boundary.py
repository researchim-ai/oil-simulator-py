"""Explicit thermal characterization of accepted black-oil well reports.

Black-oil FIM is isothermal: it does not produce a reservoir energy balance.
An assay and a temperature/energy boundary are mandatory, recorded inputs.
This adapter does not alter well rates, reservoir PVT or FIM equations.
"""

import math

try:
    from ..surface.component_energy import StockComponentAssay
    from ..surface.liquid import ConservationError, _number
except ImportError:
    from surface.component_energy import StockComponentAssay
    from surface.liquid import ConservationError, _number


def characterize_well_thermal_boundaries(well_reports, thermal_conditions, fluid, assay):
    """Map accepted SurfaceCoupledBlackOilWells surface_report['wells'].

    Every condition supplies elevation_m and exactly one of temperature_k or
    total_energy_rate_w; no guessed temperature/stock-to-species assignment.
    Input rates are stock-component kg/s, never reservoir phase rates.
    """
    if not isinstance(assay, StockComponentAssay):
        raise TypeError("Explicit StockComponentAssay required")
    if (
        not isinstance(well_reports, dict)
        or not well_reports
        or set(well_reports) != set(thermal_conditions)
    ):
        raise ValueError("Thermal conditions must match all accepted well reports")
    streams = {}
    boundaries = {}
    for name, report in well_reports.items():
        condition = dict(thermal_conditions[name])
        if not {"component_mass_rates_kg_s", "bhp_pa"} <= set(report):
            raise ValueError("Accepted well stock rates and BHP required")
        if (
            "elevation_m" not in condition
            or set(condition)
            - {
                "elevation_m",
                "velocity_m_s",
                "temperature_k",
                "total_energy_rate_w",
                "temperature_guess_k",
            }
            or ("temperature_k" in condition) == ("total_energy_rate_w" in condition)
        ):
            raise ValueError("Explicit elevation and exactly one thermal boundary required")
        pressure = _number(report["bhp_pa"], "well_bhp_pa", positive=True)
        stock = report["component_mass_rates_kg_s"]
        if "temperature_k" in condition:
            if "temperature_guess_k" in condition:
                raise ValueError("Temperature guess applies only to an energy boundary")
            boundary = assay.from_isothermal_boundary(stock, fluid, pressure, **condition)
            stream = boundary.pop("stream")
        else:
            stream = assay.from_energy_boundary(stock, fluid, pressure, **condition)
            boundary = {
                "boundary_type": "prescribed_total_energy",
                "reservoir_energy_balance_available": False,
                "energy_supplied_by_thermal_boundary_w": stream.total_energy_rate_w,
                "assay_id": assay.model_id,
            }
        error = stream.mass_rate_kg_s(fluid) - math.fsum(stock)
        if abs(error) > 1e-10 * max(1.0, math.fsum(stock)):
            raise ConservationError("Well thermal characterization changed mass")
        streams[name] = stream
        boundaries[name] = {**boundary, "mass_residual_kg_s": error, "stream": stream.to_dict()}
    return {
        "streams": streams,
        "boundaries": boundaries,
        "total_energy_rate_w": math.fsum(s.total_energy_rate_w for s in streams.values()),
        "reservoir_energy_equation_solved": False,
        "pvt_eos_characterization_calibrated": False,
    }
