"""Reproduce the liquid ring check with a locally built EPANET 2.2 library.

Usage: python scripts/check_surface_epanet.py PATH_TO_EPANET_BUILD_ROOT
The root must contain build/lib/libepanet2.so and source_manifest.json.
Writes case input, report and comparison JSON into that root.
With --config, compare another single-liquid pipe/pump network.
"""

import ctypes as c
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse

from surface import LiquidNetwork

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference_root", type=Path)
parser.add_argument("--config", type=Path)
parser.add_argument("--max-flow-error-kg-s", type=float, default=1e-5)
parser.add_argument("--max-head-error-m", type=float, default=0.02)
parser.add_argument("--max-power-relative-error", type=float, default=0.002)
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--match-si-local-loss", action="store_true",
                    help="Convert SI K to EPANET's rounded 0.02517 internal head coefficient")
args = parser.parse_args()
root = args.reference_root
output = args.output_dir or root
output.mkdir(parents=True, exist_ok=True)
data = json.loads(
    (
        args.config or Path(__file__).resolve().parents[1] / "configs/surface_liquid_ring.json"
    ).read_text()
)
# Explicitly match EPANET's default kinematic viscosity in SI.
if args.config is None:
    data["fluid"]["viscosity_pa_s"] = 1000 * 1.1e-5 * 0.3048**2
net = LiquidNetwork.from_dict(data)
ours = net.solve()
rho = net.fluid.density_kg_m3
g = 9.80665
# hydcoeffs.c uses h(ft)=0.02517*K*Q(ft3/s)^2/D(ft)^4.
# Match h=K*v^2/(2*g), converting lengths by the exact 0.3048 m/ft.
local_loss_scale = 8 * 0.3048 / (math.pi**2 * g * 0.02517) if args.match_si_local_loss else 1.0
lines = ["[JUNCTIONS]"]
for n in net.nodes:
    if n.pressure_pa is None:
        lines.append(f"{n.name} {n.elevation_m} {-n.source_mass_rate_kg_s / rho * 1000}")
lines += ["[RESERVOIRS]"]
for n in net.nodes:
    if n.pressure_pa is not None:
        lines.append(f"{n.name} {n.elevation_m + n.pressure_pa / (rho * g):.16g}")
lines += ["[PIPES]"]
for p in net.pipes:
    lines.append(
        f"{p.name} {p.start} {p.end} {p.length_m} {p.diameter_m * 1000} {p.roughness_m * 1000} {p.minor_loss_coefficient * local_loss_scale} {'CV' if p.open and p.check_valve else 'OPEN' if p.open else 'CLOSED'}"
    )
lines += ["[PUMPS]"]
for pump in net.pumps:
    lines.append(
        f"{pump.name} {pump.start} {pump.end} HEAD curve_{pump.name} SPEED {pump.speed_ratio}"
    )
lines += ["[VALVES]"]
for valve in net.valves:
    if valve.check_valve:
        raise ValueError("EPANET reference exporter does not combine TCV with a check valve")
    setting = local_loss_scale / (valve.discharge_coefficient * valve.opening) ** 2 if valve.opening else 1.0
    lines.append(
        f"{valve.name} {valve.start} {valve.end} {valve.diameter_m * 1000} TCV {setting:.16g} 0"
    )
lines += ["[STATUS]"]
for valve in net.valves:
    if not valve.open or valve.opening == 0:
        lines.append(f"{valve.name} CLOSED")
for pump in net.pumps:
    if not pump.open:
        lines.append(f"{pump.name} CLOSED")
lines += ["[CURVES]"]
for pump in net.pumps:
    if pump.head_curve is None:
        points = [
            (q, pump.shutoff_head_m - pump.curve_coefficient_s2_m5 * q**2)
            for q in (0, 0.5 * pump.max_flow_m3_s, pump.max_flow_m3_s)
        ]
    else:
        # >3 points prevents EPANET's 3-point power-law fit. Extra collinear
        # points retain the supplied piecewise-linear equipment map exactly.
        axis = pump.head_curve.flow_m3_s
        queries = sorted(
            {
                a + (b - a) * fraction
                for a, b in zip(axis, axis[1:])
                for fraction in (0, 1 / 3, 2 / 3, 1)
            }
        )
        points = [(q, pump.head_curve.evaluate(q)[0]) for q in queries]
    for q, head in points:
        lines.append(f"curve_{pump.name} {q * 1000:.16g} {head:.16g}")
    efficiency_points = (
        zip(pump.efficiency_curve.flow_m3_s, pump.efficiency_curve.values)
        if pump.efficiency_curve is not None
        else [(0, pump.efficiency), (pump.max_flow_m3_s, pump.efficiency)]
    )
    for q, eff in efficiency_points:
        lines.append(f"eff_{pump.name} {q * 1000:.16g} {eff * 100:.16g}")
lines += ["[ENERGY]"]
for pump in net.pumps:
    lines.append(f"PUMP {pump.name} EFFIC eff_{pump.name}")
viscosity = net.fluid.viscosity_pa_s / rho / (1.1e-5 * 0.3048**2)
lines += [
    "[OPTIONS]",
    "UNITS LPS",
    "HEADLOSS D-W",
    f"VISCOSITY {viscosity:.16g}",
    f"SPECIFIC GRAVITY {rho / 1000:.16g}",
    "ACCURACY 0.00000001",
    "TRIALS 100",
    "[END]",
]
case_name = args.config.stem if args.config else "ring"
inp = output / f"{case_name}.inp"
inp.write_text("\n".join(lines) + "\n")
lib = c.CDLL(str((root / "build/lib/libepanet2.so").resolve()))
h = c.c_void_p()


reference_warnings = []


def call(name, args, types):
    fn = getattr(lib, name)
    fn.argtypes = types
    fn.restype = c.c_int
    code = fn(*args)
    if code == 4 and name == "EN_solveH":
        # EPANET warns when a pump cannot deliver against backpressure.
        # Preserve the warning and export link status for independent review.
        reference_warnings.append({"function": name, "code": code})
    elif code:
        raise RuntimeError((name, code))


P = c.c_void_p
S = c.c_char_p
I = c.c_int
D = c.c_double
call("EN_createproject", [c.byref(h)], [c.POINTER(P)])
try:
    call(
        "EN_open",
        [h, str(inp).encode(), str(output / f"{case_name}.rpt").encode(), b""],
        [P, S, S, S],
    )
    call("EN_solveH", [h], [P])
    heads = {}
    flows = {}
    for items, prefix, prop, out in [
        (net.nodes, "node", 10, heads),
        (net.pipes + net.pumps + net.valves, "link", 8, flows),
    ]:
        for item in items:
            index = I()
            value = D()
            call(
                "EN_get" + prefix + "index",
                [h, item.name.encode(), c.byref(index)],
                [P, S, c.POINTER(I)],
            )
            call(
                "EN_get" + prefix + "value",
                [h, index, prop, c.byref(value)],
                [P, I, I, c.POINTER(D)],
            )
            out[item.name] = value.value
    pump_power = {}
    for pump in net.pumps:
        index, value = I(), D()
        call("EN_getlinkindex", [h, pump.name.encode(), c.byref(index)], [P, S, c.POINTER(I)])
        call("EN_getlinkvalue", [h, index, 13, c.byref(value)], [P, I, I, c.POINTER(D)])
        pump_power[pump.name] = value.value * 1000
    statuses = {}
    for item in net.pipes + net.pumps + net.valves:
        index = I()
        value = D()
        call("EN_getlinkindex", [h, item.name.encode(), c.byref(index)], [P, S, c.POINTER(I)])
        call("EN_getlinkvalue", [h, index, 11, c.byref(value)], [P, I, I, c.POINTER(D)])
        statuses[item.name] = value.value
finally:
    call("EN_close", [h], [P])
    call("EN_deleteproject", [h], [P])
mass_rates = dict(
    ours.pipe_mass_rates_kg_s, **ours.pump_mass_rates_kg_s, **ours.valve_mass_rates_kg_s
)
flowerr = max(abs(mass_rates[k] - v * rho / 1000) for k, v in flows.items())
headerr = max(
    abs(ours.pressures_pa[n.name] / (rho * g) + n.elevation_m - heads[n.name]) for n in net.nodes
)
our_power = {
    pump.name: (ours.pump_shaft_power_w[pump.name]
                if ours.branch_states[pump.name] == "open" else 0.0)
    for pump in net.pumps
}
powererr = max(
    (abs(our_power[k] - v) / max(abs(v), 1.0)
     for k, v in pump_power.items()), default=0.0
)
status_mismatches = [
    k for k, v in statuses.items()
    if (ours.branch_states[k] == "open") != bool(v)
]
accepted = (
    all(math.isfinite(v) for values in
        (mass_rates, heads, flows, pump_power, our_power, ours.pressures_pa, statuses)
        for v in values.values())
    and all(math.isfinite(v) for v in (flowerr, headerr, powererr))
    and flowerr <= args.max_flow_error_kg_s
    and headerr <= args.max_head_error_m
    and powererr <= args.max_power_relative_error
    and not status_mismatches
)
result = {
    "reference": "OpenWaterAnalytics EPANET v2.2",
    "source": json.loads((root / "source_manifest.json").read_text()),
    "input": data,
    "epanet_heads_m": heads,
    "epanet_shaft_power_w": pump_power,
    "epanet_flows_lps": flows,
    "epanet_link_status": statuses,
    "epanet_warnings": reference_warnings,
    "ours": ours.to_dict(),
    "max_flow_difference_kg_s": flowerr,
    "max_head_difference_m": headerr,
    "max_power_scaled_difference": powererr,
    "status_mismatches": status_mismatches,
    "si_local_loss_coefficient_matched": args.match_si_local_loss,
    "local_loss_export_scale": local_loss_scale,
    "accepted": accepted,
    "budgets": {"flow_kg_s": args.max_flow_error_kg_s,
                "head_m": args.max_head_error_m,
                "power_relative_with_1w_floor": args.max_power_relative_error},
    "note": "EPANET uses g=32.2 ft/s2 and rounded local-loss coefficient; production uses standard g=9.80665 m/s2. Viscosity explicitly matched.",
}
(output / (f"{case_name}_comparison.json" if args.config else "comparison.json")).write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps({"flow_error_kg_s": flowerr, "head_error_m": headerr}))

if not accepted:
    raise SystemExit("EPANET comparison exceeded tolerances")
