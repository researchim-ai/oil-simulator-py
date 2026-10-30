"""Reproduce the liquid ring check with a locally built EPANET 2.2 library.

Usage: python scripts/check_surface_epanet.py PATH_TO_EPANET_BUILD_ROOT
The root must contain build/lib/libepanet2.so and source_manifest.json.
Writes ring.inp, ring.rpt and comparison.json into that root.
"""

import ctypes as c
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse

from surface import LiquidNetwork

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference_root", type=Path)
args = parser.parse_args()
root = args.reference_root
data = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/surface_liquid_ring.json").read_text()
)
# Explicitly match EPANET's default kinematic viscosity in SI.
data["fluid"]["viscosity_pa_s"] = 1000 * 1.1e-5 * 0.3048**2
net = LiquidNetwork.from_dict(data)
ours = net.solve()
rho = 1000
g = 9.80665
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
        f"{p.name} {p.start} {p.end} {p.length_m} {p.diameter_m * 1000} {p.roughness_m * 1000} {p.minor_loss_coefficient} OPEN"
    )
lines += ["[OPTIONS]", "UNITS LPS", "HEADLOSS D-W", "ACCURACY 0.00000001", "TRIALS 100", "[END]"]
inp = root / "ring.inp"
inp.write_text("\n".join(lines) + "\n")
lib = c.CDLL(str((root / "build/lib/libepanet2.so").resolve()))
h = c.c_void_p()


def call(name, args, types):
    fn = getattr(lib, name)
    fn.argtypes = types
    fn.restype = c.c_int
    code = fn(*args)
    if code:
        raise RuntimeError((name, code))


P = c.c_void_p
S = c.c_char_p
I = c.c_int
D = c.c_double
call("EN_createproject", [c.byref(h)], [c.POINTER(P)])
try:
    call("EN_open", [h, str(inp).encode(), str(root / "ring.rpt").encode(), b""], [P, S, S, S])
    call("EN_solveH", [h], [P])
    heads = {}
    flows = {}
    for items, prefix, prop, out in [(net.nodes, "node", 10, heads), (net.pipes, "link", 8, flows)]:
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
finally:
    call("EN_close", [h], [P])
    call("EN_deleteproject", [h], [P])
flowerr = max(abs(ours.pipe_mass_rates_kg_s[k] - v) for k, v in flows.items())
headerr = max(
    abs(ours.pressures_pa[n.name] / (rho * g) + n.elevation_m - heads[n.name]) for n in net.nodes
)
result = {
    "reference": "OpenWaterAnalytics EPANET v2.2",
    "source": json.loads((root / "source_manifest.json").read_text()),
    "input": data,
    "epanet_heads_m": heads,
    "epanet_flows_lps": flows,
    "ours": ours.to_dict(),
    "max_flow_difference_kg_s": flowerr,
    "max_head_difference_m": headerr,
    "note": "EPANET uses g=32.2 ft/s2 and rounded local-loss coefficient; production uses standard g=9.80665 m/s2. Viscosity explicitly matched.",
}
(root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"flow_error_kg_s": flowerr, "head_error_m": headerr}))

if flowerr > 1e-5 or headerr > 0.02:
    raise SystemExit("EPANET comparison exceeded tolerances")
