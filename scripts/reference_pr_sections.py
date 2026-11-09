"""Independent NumPy fixed-pivot/tail audit; no runtime population methods."""

import bisect
import math

import numpy as np
from check_stratified_pr_flow import close


def properties(node, state, x, pn, vm, z, rho):
    from reference_pr_population import identify_roles

    law, spec = dict(node["drift_flux"]), node["droplet_population"]
    coords = np.asarray(state.get("droplet_sections"), dtype=float)
    pivots = np.array(spec["reference_volumes_m3"])
    if coords.shape != (len(pivots) + 1,) or np.any(coords < 0) or not np.isfinite(coords).all():
        raise ValueError("Invalid sectional checkpoint")
    counts, tail = coords[:-1], coords[-1]
    close(state["droplet_number"], counts.sum(), atol=1e-6, rtol=1e-10)
    roles = identify_roles(law, spec, x, z, rho)
    chosen = 1 if law["dispersed_liquid"] == "oil" else 2
    carrier = 0 if law.get("model") == "dispersed_gas_liquid" else (2 if chosen == 1 else 1)
    mask = roles == chosen
    present = bool(np.any(mask) and np.any(roles == carrier))
    volume = float((pn * vm)[mask].sum()) if present else 0.0
    if counts.sum() > spec["maximum_number_density_m3"] * node["volume_m3"]:
        raise ValueError("Sectional count outside domain")
    if counts[-1] == 0 and tail != 0:
        raise ValueError("Upper-tail volume without particles")
    upper = tail / counts[-1] if counts[-1] else pivots[-1]
    if upper < pivots[-1] * (1 - 1e-10) or upper > pivots[-1] * spec["maximum_tail_volume_ratio"]:
        raise ValueError("Upper-tail volume outside domain")
    representatives = np.r_[pivots[:-1], upper]
    parts = np.r_[counts[:-1] * pivots[:-1], tail]
    reference = parts.sum()
    if present:
        if counts.sum() <= 0 or reference <= 0:
            raise ValueError("Dispersed liquid has no size spectrum")
        fractions = parts / reference
        sizes = np.cbrt(6 * volume * representatives / (math.pi * reference))
        surfaces = law["surface_tension_n_m"] * math.pi * counts * sizes**2
        effective = math.sqrt(fractions @ sizes**2)
    else:
        if np.any(coords != 0):
            raise ValueError("Particles without a dispersed/carrier pair")
        fractions = np.zeros_like(counts)
        sizes = np.full_like(counts, law["droplet_diameter_m"])
        surfaces = np.zeros_like(counts)
        effective = law["droplet_diameter_m"]
    law["droplet_diameter_m"] = effective
    return (
        {
            "number": counts.sum(),
            "diameter_m": effective,
            "surface_energy_j": surfaces.sum(),
            "drop_volume_m3": volume,
            "drop_mask": mask,
            "present": present,
            "phase_roles": roles,
            "coordinates": coords,
            "section_numbers": counts,
            "section_volume_fractions": fractions,
            "section_diameters_m": sizes,
            "section_surface_energy_j": surfaces,
            "reference_representatives_m3": representatives,
            "reference_volume_m3": reference,
            "tail_reference_volume_m3": tail,
        },
        law,
        surfaces.sum(),
    )


def velocities(
    node,
    p,
    rho,
    phase_velocity,
    axial,
    gravity,
    maximum,
    pressure,
    temperature=None,
    gas_molar_mass=None,
):
    roles, law = p["phase_roles"], node["drift_flux"]
    ro = float(np.sum(rho[roles == 1]))
    rw = float(np.sum(rho[roles == 2]))
    carrier_mu = (
        law["water_viscosity_pa_s"]
        if law["dispersed_liquid"] == "oil"
        else law["oil_viscosity_pa_s"]
    )
    mist = law.get("model") == "dispersed_gas_liquid"
    if mist:
        carrier_mu = law["gas_viscosity_pa_s"]
    drop_mu = (
        law["oil_viscosity_pa_s"]
        if law["dispersed_liquid"] == "oil"
        else law["water_viscosity_pa_s"]
    )
    density = rw if law["dispersed_liquid"] == "oil" else ro
    if mist:
        density = float(np.sum(rho[roles == 0]))
    diameter = p["section_diameters_m"]
    mobility = diameter**2 / (18 * carrier_mu)
    if law["interface"] == "clean":
        mobility *= 3 * (carrier_mu + drop_mu) / (2 * carrier_mu + 3 * drop_mu)
    slip = (rw - ro) * axial * mobility * (1 if law["dispersed_liquid"] == "oil" else -1)
    contrast = abs(rw - ro)
    if mist:
        drop_density = ro if law["dispersed_liquid"] == "oil" else rw
        contrast = drop_density - density
        slip = -contrast * axial * mobility
    deviation = slip - p["section_volume_fractions"] @ slip
    mean = float(phase_velocity @ p["drop_mask"])
    p["section_velocities_m_s"] = mean + deviation if p["present"] else np.full_like(slip, mean)
    if p["present"] and (
        np.max(density * abs(contrast) * gravity * mobility * diameter / carrier_mu)
        > law["maximum_reynolds_number"]
        or np.max(abs(contrast) * gravity * diameter**2 / law["surface_tension_n_m"])
        > law["maximum_eotvos_number"]
        or np.max(diameter / node["diameter_m"]) > law["maximum_confinement_ratio"]
        or np.max(4 * law["surface_tension_n_m"] / diameter)
        > law["maximum_capillary_pressure_fraction"] * pressure
        or np.max(np.abs(p["section_velocities_m_s"])) > maximum
    ):
        raise ValueError("Size-resolved drift envelope rejected")
    if mist and p["present"]:
        if temperature is None or gas_molar_mass is None:
            raise ValueError("Gas mist spectrum needs local continuum properties")
        path = (
            carrier_mu
            / density
            * math.sqrt(math.pi * gas_molar_mass / (2 * 8.31446261815324 * temperature))
        )
        if np.any(path / diameter > law["maximum_knudsen_number"]):
            raise ValueError("Rarefied droplet section outside continuum regime")
    return float(p["section_volume_fractions"] @ deviation**2)


def endpoint_data(data):
    pop = data.get("population")
    if pop is None or "section_velocities_m_s" not in pop:
        return data
    count, n = len(pop["section_numbers"]), len(data["x"])
    chosen = int(np.argmax(pop["drop_mask"]))
    index = np.r_[np.arange(n), np.full(count, chosen)]
    alpha = float(data["alpha"] @ pop["drop_mask"])
    velocity = np.r_[data["velocities"], pop["section_velocities_m_s"]]
    result = {
        **data,
        "x": data["x"][index],
        "alpha": np.r_[
            data["alpha"] * (1 - pop["drop_mask"]), alpha * pop["section_volume_fractions"]
        ],
        "velocities": velocity,
        "drift": velocity - data["u"],
        "population": {
            **pop,
            "drop_mask": np.r_[np.zeros(n), np.ones(count)],
            "surface_density_phase": np.r_[
                np.zeros(n),
                alpha * pop["section_surface_energy_j"] / max(pop["drop_volume_m3"], 1e-100),
            ],
        },
    }
    for key in ("vm", "rho_phase", "phase_z", "h", "s_phase"):
        result[key] = data[key][index]
    return result


def sources(spec, coords, volume, population=None, temperature_k=None, law=None):
    pivots = np.asarray(spec["reference_volumes_m3"])
    n = np.asarray(coords[:-1])
    tail = coords[-1]
    rep = np.r_[pivots[:-1], tail / n[-1] if n[-1] else pivots[-1]]
    coal = np.zeros_like(coords)
    split = np.zeros_like(coords)

    def deposit(result, size, rate):
        if size >= pivots[-1]:
            result[-2] += rate
            result[-1] += rate * size
        else:
            hi = bisect.bisect_right(pivots, size)
            hi = max(1, hi)
            lo = hi - 1
            weight = (size - pivots[lo]) / (pivots[hi] - pivots[lo])
            result[lo] += rate * (1 - weight)
            result[hi] += rate * weight
            if hi == len(pivots) - 1:
                result[-1] += rate * weight * pivots[-1]

    for i in range(len(n)):
        for j in range(i, len(n)):
            if spec.get("coalescence_model", "constant") == "constant":
                kernel = spec["coalescence_kernel_m3_s"]
            else:
                mu = (
                    law["water_viscosity_pa_s"]
                    if law["dispersed_liquid"] == "oil"
                    else law["oil_viscosity_pa_s"]
                )
                if law.get("model") == "dispersed_gas_liquid":
                    mu = law["gas_viscosity_pa_s"]
                mud = (
                    law["oil_viscosity_pa_s"]
                    if law["dispersed_liquid"] == "oil"
                    else law["water_viscosity_pa_s"]
                )
                mobility = (
                    3 * (mu + mud) / (2 * mu + 3 * mud) if law["interface"] == "clean" else 1.0
                )
                di, dj = population["section_diameters_m"][[i, j]]
                ui, uj = population["section_velocities_m_s"][[i, j]]
                kernel = spec["collision_efficiency"] * (
                    2
                    * 1.380649e-23
                    * temperature_k
                    * mobility
                    / (3 * mu)
                    * (di + dj) ** 2
                    / (di * dj)
                    + math.pi * (di + dj) ** 2 * abs(ui - uj) / 4
                )
            event = kernel * n[i] * n[j] / volume * (0.5 if i == j else 1.0)
            coal[i] -= event
            coal[j] -= event
            if i == len(n) - 1:
                coal[-1] -= event * rep[i]
            if j == len(n) - 1:
                coal[-1] -= event * rep[j]
            deposit(coal, rep[i] + rep[j], event)
        event = spec["section_breakup_rates_s"][i] * n[i]
        statistical = spec.get("daughter_distribution", "deterministic") == "symmetric_beta"
        fractions = spec.get("daughter_volume_fractions", (0.5, 0.5))
        if statistical:
            if rep[i] < 2 * pivots[0]:
                event = 0.0
        elif rep[i] * min(fractions) < pivots[0] * (1 - 1e-12):
            event = 0.0
        if event:
            split[i] -= event
            if i == len(n) - 1:
                split[-1] -= event * rep[i]
            if not statistical or rep[i] == 2 * pivots[0]:
                for fraction in (0.5, 0.5) if statistical else fractions:
                    deposit(split, rep[i] * fraction, event)
            else:
                from scipy.integrate import quad, quad_vec

                lower = pivots[0] / rep[i]
                shape = spec.get("daughter_beta_shape", 1)

                def density(x, shape=shape):
                    return (x * (1 - x)) ** (shape - 1)

                normalizer = quad(density, lower, 1 - lower, epsabs=1e-25)[0]

                def integrand(
                    x, parent=rep[i], event=event, density=density, normalizer=normalizer
                ):
                    birth = np.zeros_like(coords)
                    deposit(birth, parent * x, 2 * event * density(x) / normalizer)
                    return birth

                points = [v / rep[i] for v in pivots if lower < v / rep[i] < 1 - lower]
                split += quad_vec(
                    integrand, lower, 1 - lower, points=points, epsabs=1e-20, epsrel=1e-12
                )[0]
    return coal, split


def surface(spec, coords, volume, sigma):
    n = np.asarray(coords[:-1])
    tail = coords[-1]
    if np.any(n < 0) or tail < 0:
        raise ValueError("Negative counterfactual spectrum")
    pivots = np.asarray(spec["reference_volumes_m3"])
    total = n[:-1] @ pivots[:-1] + tail
    if not total:
        return 0.0
    scale = volume / total
    return (
        sigma
        * (36 * math.pi) ** (1 / 3)
        * (n[:-1] @ (scale * pivots[:-1]) ** (2 / 3) + n[-1] ** (1 / 3) * (scale * tail) ** (2 / 3))
    )


def audit_balance(node, old, p, net, record, dt, temperature_k):
    spec = node["droplet_population"]
    c = p["coordinates"]
    net = np.zeros_like(c) + net
    prior = np.asarray(old["droplet_sections"])
    coag, split = sources(spec, c, node["volume_m3"], p, temperature_k, node["drift_flux"])
    if not p["present"]:
        expected = np.zeros_like(c)
    elif prior[:-1].sum() > 0 or np.asarray(net)[:-1].sum() > 0:
        expected = prior + dt * (net + coag + split)
    else:
        frac = np.array(spec["initial_volume_fractions"])
        pivot = np.array(spec["reference_volumes_m3"])
        pivot[-1] *= spec["tail_initial_volume_ratio"]
        expected = np.r_[p["drop_volume_m3"] * frac / pivot, p["drop_volume_m3"] * frac[-1]]
    close(c[:-1], expected[:-1], atol=1e-5, rtol=2e-8)
    close(c[-1], expected[-1], atol=1e-20, rtol=2e-8)
    for key in (
        "number",
        "diameter_m",
        "surface_energy_j",
        "drop_volume_m3",
        "coordinates",
        "section_numbers",
        "section_volume_fractions",
        "section_diameters_m",
        "section_surface_energy_j",
        "section_velocities_m_s",
        "reference_volume_m3",
        "tail_reference_volume_m3",
    ):
        close(record[key], p[key], atol=1e-15, rtol=2e-8)
    for key, value in (
        ("net_section_rates", net),
        ("coalescence_section_rates", coag),
        ("breakup_section_rates", split),
    ):
        close(record[key][:-1], value[:-1], atol=1e-5, rtol=2e-7)
        close(record[key][-1], value[-1], atol=1e-20, rtol=2e-7)
    required = (
        max(
            0.0,
            (
                p["surface_energy_j"]
                - surface(
                    spec,
                    c - dt * split,
                    p["drop_volume_m3"],
                    node["drift_flux"]["surface_tension_n_m"],
                )
            )
            / dt,
        )
        if any(spec["section_breakup_rates_s"])
        else 0.0
    )
    if required > spec["agitation_power_w"] * (1 + 1e-8) + 1e-12:
        raise ValueError("Sectional breakup exceeds mechanical power")
    close(record["breakup_surface_power_w"], required, atol=1e-10, rtol=2e-8)
    if max(map(abs, record["section_balance_residuals"])) > 1e-8:
        raise ValueError("Sectional residual rejected")


def number_flux(base, contact, left, right, law, a, b, spec):
    count = len(spec["reference_volumes_m3"])

    def packet(d, p):
        if p is None:
            from reference_pr_population import identify_roles

            roles = identify_roles(law, spec, d["phase_x"], d["phase_z"], d["phase_densities"])
            mask = roles == (1 if law["dispersed_liquid"] == "oil" else 2)
            fraction = d["phase_weights"] @ mask
            pivots = np.array(spec["reference_volumes_m3"])
            pivots[-1] *= spec["tail_initial_volume_ratio"]
            distribution = np.array(spec["initial_volume_fractions"])
            return fraction * np.r_[distribution / pivots, distribution[-1]], np.full(
                count + 1, d["velocity"] @ mask
            )
        density = (
            d["phase_weights"][-count:].sum() / max(p["drop_volume_m3"], 1e-100) * p["coordinates"]
        )
        velocity = d["velocity"][-count:]
        return density, np.r_[velocity, velocity[-1]]

    nl, vl = packet(left, a)
    nr, vr = packet(right, b)
    donor = left if contact >= 0 else right
    bulk = (
        float(np.sum(base[:-2]))
        / float(np.sum(donor["phase_concentration"]))
        * (nl if contact >= 0 else nr)
    )
    return (
        bulk
        + nl * (np.maximum(vl, 0) - max(left["u"], 0))
        + nr * (np.minimum(vr, 0) - min(right["u"], 0))
    )
