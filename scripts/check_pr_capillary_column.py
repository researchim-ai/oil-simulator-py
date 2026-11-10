"""Independent PR/profile, finite-area and shared-transfer coupling audit.

SciPy is used only by the existing independent chemical-profile verifier.
The production coupled integrator and its linear algebra use PyTorch.
"""

import math

import numpy as np
from check_pr_capillary_axial import audit as axial_audit
from check_pr_capillary_pressure import audit as normal_audit
from check_pr_capillary_pressure import close
from check_stratified_pr_flow import profile
from reference_pr_active_multifluid import packets
from reference_wetting_potential import potential

from surface.operation import _digest
from surface.pr_active_multifluid_pipe import PRActiveMultifluidState
from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_pressure import PRCapillaryPressure


def aperture_area(low, high, height, diameter):
    """Independent 64-point positive integrals, including vanishing caps."""
    if high <= low:
        return 0.0
    if low >= height:
        low, high = height + diameter / 2 - high, height + diameter / 2 - low
    else:
        low, high = low - height + diameter / 2, high - height + diameter / 2
    lo, hi = np.clip([low, high], 0, diameter)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    t, w = (nodes + 1) / 2, weights / 2

    def cap(h):
        return (
            4 * math.sqrt(diameter) * h**1.5 * np.sum(w * t * t * np.sqrt(1 - h / diameter * t * t))
        )

    def band(a, b):
        if b <= a:
            return 0.0
        if a > 0 and b - a < a / 2:
            h = a + (b - a) * t
            return 2 * (b - a) * np.sum(w * np.sqrt(h * (diameter - h)))
        return cap(b) - cap(a)

    radius = diameter / 2
    return band(min(lo, radius), min(hi, radius)) + band(
        min(diameter - hi, radius), min(diameter - lo, radius)
    )


def audit(
    system,
    before,
    after,
    record,
    *,
    normal_relaxation=True,
    column_heat_w=0.0,
    heat_temperature_k=None,
):
    if _digest(record["state"]) != _digest(after.to_dict()) or before.model_id != _digest(system):
        raise ValueError("Coupled model/history identity differs")
    if _digest(record["column"]["state"]) != _digest(after.column.to_dict()):
        raise ValueError("Reported column state differs")
    for suffix, state in (("before", before), ("after", after)):
        if (
            record["column_phase_count_" + suffix] != len(state.column.phase_roots)
            or tuple(record["column_layer_slots_" + suffix]) != state.layer_slots
        ):
            raise ValueError("Reported active layer registry differs")
    start = PRCapillaryAxialPipe.from_dict(record["pipe_model"])
    end = PRCapillaryAxialPipe.from_dict(record["end_pipe_model"])
    expected = PRCapillaryPressure(start.pipe, **system["parameters"]).to_dict()
    for model in (start, end):
        if _digest(model.pipe.to_dict()) != _digest(system["pipe"]):
            raise ValueError("Coupled physical pipe changed")
        actual = {
            k: v
            for k, v in model.source.to_dict().items()
            if k not in ("boundary_heights_m", "boundary_distances_m")
        }
        if _digest(actual) != _digest(expected):
            raise ValueError("Coupled capillary material or geometry changed")
    nc = len(start.pipe.fluid.names)
    side, height = system["port_side"], system["port_height_m"]
    column = system["column"]
    old_column, new_column = before.column.to_dict(), after.column.to_dict()
    references = []
    for state, coupled, model in ((old_column, before, start), (new_column, after, end)):
        _f, geometry, sample, _amounts, _rho, entropy = profile(state, column)
        bounds = [0, *state["interface_heights_m"], geometry.height_m]
        if len(bounds) not in (2, 3) or len(coupled.layer_slots) != len(bounds) - 1:
            raise ValueError("Explicit active column-layer registry required")
        diameter, area = model.pipe.diameter, model.pipe.area
        active_names = set()
        liquid_fraction = 0.0
        for i, name in enumerate(coupled.layer_slots):
            low, high = (
                max(bounds[i], height - diameter / 2),
                min(bounds[i + 1], height + diameter / 2),
            )
            fraction = aperture_area(low, high, height, diameter) / area
            if name == model.source.liquid:
                liquid_fraction += fraction
            if model is end:
                close(
                    record["end_aperture"][i]["area_fraction"],
                    fraction,
                    "reported aperture area",
                    atol=1e-12,
                    rtol=1e-10,
                )
            if fraction == 0:
                if name in model.boundaries[side]:
                    raise ValueError("Empty aperture phase acquired a ghost stock")
                continue
            active_names.add(name)
            phase = model.boundaries[side][name]
            pressure, x, q = sample((low + high) / 2, i)
            volume = fraction * area * model.pipe.lengths[0 if side == 0 else -1]
            close(
                phase.volume_m3, volume, "independent aperture stock volume", atol=1e-24, rtol=1e-10
            )
            close(
                phase.pressure_pa, pressure, "independent aperture pressure", atol=1e-4, rtol=1e-9
            )
            close(
                phase.temperature_k,
                state["temperature_k"],
                "aperture temperature",
                atol=1e-8,
                rtol=0,
            )
            expected = volume / q["vm"] * x
            close(
                phase.component_moles, expected, "aperture component state", atol=1e-18, rtol=2e-8
            )
            close(
                phase.total_energy_j,
                float(expected.sum() * q["u"]),
                "aperture calorics",
                atol=1e-12,
                rtol=2e-8,
            )
            if phase.momentum_kg_m_s != 0:
                raise ValueError("Stationary column acquired a ghost boundary momentum")
            head = q["rho"] * column["gravity_m_s2"] * diameter / pressure
            if head > system["maximum_transverse_head_fraction"]:
                raise ValueError("Unqualified transverse head")
        if set(model.boundaries[side]) != active_names or model.boundaries[1 - side] is not None:
            raise ValueError("Active aperture boundary registry differs")
        if model.boundary_geometry != "reservoir_cell":
            raise ValueError("Finite column requires the declared reservoir-cell boundary")
        close(
            model.source.boundary_heights[side],
            liquid_fraction * area / model.source.width,
            "boundary height",
            atol=1e-18,
            rtol=1e-10,
        )
        close(
            model.source.boundary_distances[side],
            model.pipe.lengths[0 if side == 0 else -1],
            "physical boundary distance",
            atol=0,
            rtol=0,
        )
        references.append(entropy)
    pipe_record = record["pipe"]
    axial = pipe_record["axial"] if normal_relaxation else pipe_record
    transported = PRActiveMultifluidState.from_dict(axial["state"])
    axial_audit(start, before.pipe, transported, axial)
    if normal_relaxation:
        normal_audit(start.source, transported, after.pipe, pipe_record["normal"])
    elif transported != after.pipe:
        raise ValueError("Unreported coupled pipe update")
    transfer = np.array(axial["external_transfer_n_m_e"])
    residual_n = (
        np.array(new_column["component_moles"]) - old_column["component_moles"] + transfer[:nc]
    )
    n_allowance = np.array(
        [
            128
            * math.fsum(
                (
                    math.ulp(a),
                    math.ulp(b),
                    *(
                        math.ulp(p.component_moles[j])
                        for s in (before.pipe, after.pipe)
                        for c in s.cells
                        for _, p in c
                    ),
                )
            )
            + 1e-20
            for j, (a, b) in enumerate(
                zip(old_column["component_moles"], new_column["component_moles"])
            )
        ]
    )
    close(residual_n, np.zeros(nc), "finite column shared components", atol=n_allowance, rtol=0)

    def surface(source, pipe_state=after.pipe):
        raw = source.to_dict()
        h = np.array([dict(c)[source.liquid].volume_m3 for c in pipe_state.cells]) / (
            source.width * np.array(source.pipe.lengths)
        )
        value = source.sigma * source.width / (2 * source.pipe.lengths[0]) * np.sum(np.diff(h) ** 2)
        for i, hf in enumerate(raw.get("boundary_heights_m", (None, None))):
            if hf is not None:
                value += (
                    source.sigma
                    * source.width
                    / (2 * source.boundary_distances[i])
                    * (h[0 if i == 0 else -1] - hf) ** 2
                )
        if "wetting" in raw:
            value += np.sum(
                source.width
                * np.array(source.pipe.lengths)
                * potential(raw["wetting"], source.sigma, h)
            )
        return float(value)

    shape_work = surface(end.source) - surface(start.source)
    mw = np.array([c.molar_mass_kg_mol for c in start.pipe.fluid.components])
    energy_transfer = (
        transfer[-1]
        + axial["external_capillary_work_j"]
        + float(transfer[:nc] @ mw) * column["gravity_m_s2"] * (column["elevation_m"] + height)
    )
    allowance = (
        128
        * math.fsum(
            math.ulp(v)
            for v in (
                old_column["total_energy_j"],
                new_column["total_energy_j"],
                *(
                    p.total_energy_j
                    for s in (before.pipe, after.pipe)
                    for c in s.cells
                    for _, p in c
                ),
            )
        )
        + 1e-18
    )
    heat = column_heat_w * (after.time_s - before.time_s)
    if heat and (heat_temperature_k is None or heat_temperature_k <= 0):
        raise ValueError("Prescribed heat needs its reservoir temperature")
    close(record["column_heat_j"], heat, "prescribed heat", atol=1e-18, rtol=1e-14)
    if record["heat_reservoir_temperature_k"] != heat_temperature_k:
        raise ValueError("Heat reservoir differs")
    defect = math.fsum(
        (
            new_column["total_energy_j"],
            -old_column["total_energy_j"],
            energy_transfer,
            shape_work,
            -heat,
        )
    )
    close(defect, 0, "column energy including moving boundary work", atol=allowance, rtol=0)
    close(
        record["boundary_shape_work_from_column_j"],
        shape_work,
        "reported shape work",
        atol=1e-20,
        rtol=1e-8,
    )
    close(
        record["port_energy_from_column_j"],
        energy_transfer,
        "reported port energy",
        atol=1e-18,
        rtol=1e-10,
    )
    before_s = references[0] + packets(start.pipe, before.pipe.to_dict())["entropy"]
    after_s = references[1] + packets(start.pipe, after.pipe.to_dict())["entropy"]
    entropy = (
        after_s - before_s - (heat / heat_temperature_k if heat_temperature_k is not None else 0.0)
    )
    entropy_allowance = 256 * (math.ulp(before_s) + math.ulp(after_s)) + 1e-10
    if entropy < -entropy_allowance:
        raise ValueError("Independent coupled entropy decreased")
    close(
        record["entropy_production_j_k"], entropy, "coupled entropy", atol=entropy_allowance, rtol=0
    )
    return {
        "accepted": True,
        "column_energy_defect_j": defect,
        "entropy_production_j_k": entropy,
        "qualification": "Independent scalar PR/profile and balances of declared first-order midpoint-aperture model; not a resolved aperture convergence or field qualification",
    }
