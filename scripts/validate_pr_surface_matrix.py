"""Reproducible functional CPU/CUDA surface matrix with independent audits.

Timing includes initialization, schedule and subprocess overhead. This is not
a large-network throughput benchmark or experimental equipment validation.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CASES = {
    "thermochemical_evaporation": ("surface_pr_thermochemical_evaporation", "thermochemical"),
    "thermochemical_condensation": ("surface_pr_thermochemical_condensation", "thermochemical"),
    "thermochemical_instant": ("surface_pr_thermochemical_instant", "thermochemical"),
    "thermochemical_components": ("surface_pr_thermochemical_components", "thermochemical"),
    "active_multifluid_static": ("surface_pr_active_multifluid_static", "active_multifluid"),
    "active_multifluid_contact": ("surface_pr_active_multifluid_contact", "active_multifluid"),
    "active_multifluid_birth": ("surface_pr_active_multifluid_birth", "active_multifluid"),
    "active_multifluid_gow": ("surface_pr_active_multifluid_gow", "active_multifluid"),
    "native_cross_block_heat": ("surface_pr_native_cross_block_heat", "gravity"),
    "native_surveyed_slip": ("surface_pr_native_surveyed_slip", "gravity"),
    "native_column_reverse": ("surface_pr_native_column_reverse", "gravity"),
    "native_column_parallel": ("surface_pr_native_column_parallel", "gravity"),
    "equilibrium_euler_plant_cycle": ("surface_pr_equilibrium_euler_plant_cycle", "multifluid"),
    "equilibrium_euler_plant_pump": ("surface_pr_equilibrium_euler_plant_pump", "multifluid"),
    "equilibrium_euler_plant_compressor": (
        "surface_pr_equilibrium_euler_plant_compressor",
        "multifluid",
    ),
    "equilibrium_euler_cycle": ("surface_pr_equilibrium_euler_cycle", "multifluid"),
    "equilibrium_euler_transport": ("surface_pr_equilibrium_euler_transport", "multifluid"),
    "equilibrium_euler_walls": ("surface_pr_equilibrium_euler_walls", "multifluid"),
    "equilibrium_euler_three_phase": ("surface_pr_equilibrium_euler_three_phase", "multifluid"),
    "multifluid_hydrostatic": ("surface_pr_multifluid_hydrostatic", "multifluid"),
    "multifluid_hydrostatic_three_phase": (
        "surface_pr_multifluid_hydrostatic_three_phase",
        "multifluid",
    ),
    "multifluid_pump": ("surface_pr_multifluid_pump", "multifluid"),
    "multifluid_network": ("surface_pr_multifluid_network", "multifluid"),
    "multifluid_gravity": ("surface_pr_multifluid_gravity", "multifluid"),
    "multifluid_gravity_three_phase": ("surface_pr_multifluid_gravity_three_phase", "multifluid"),
    "multifluid_counterflow": ("surface_pr_multifluid_counterflow", "multifluid"),
    "multifluid_three_phase": ("surface_pr_multifluid_three_phase", "multifluid"),
    "multifluid_walls": ("surface_pr_multifluid_walls", "multifluid"),
    "interphase_relaxation": ("surface_pr_interphase_relaxation", "relaxation"),
    "interphase_three_phase": ("surface_pr_interphase_three_phase", "relaxation"),
    "interphase_stiff": ("surface_pr_interphase_stiff", "relaxation"),
    "controlled_profile_graph": ("surface_pr_controlled_profile_graph", "controlled"),
    "profile_graph": ("surface_pr_profile_graph", "profile_graph"),
    "controlled_signed_pump": ("surface_pr_controlled_signed_pump", "controlled"),
    "layered_transport": ("surface_pr_layered_transport", "layered"),
    "layered_plant": ("surface_pr_layered_plant", "layered_plant"),
    "signed_pump_connection": ("surface_pr_signed_pump_connection", "equilibrium"),
    "signed_compressor": ("surface_pr_signed_compressor", "equilibrium"),
    "signed_pump_reversal": ("surface_pr_signed_pump_reversal", "equilibrium"),
    "inclined_inventory": ("surface_pr_inclined_inventory", "gravity"),
    "stratified_drift": ("surface_pr_stratified_drift", "equilibrium"),
    "sectional_beta": ("surface_pr_sectional_beta_mixer", "equilibrium"),
    "controlled_motor": ("surface_pr_controlled_motor", "controlled"),
    "controlled_minimum_flow": ("surface_pr_controlled_minimum_flow", "controlled"),
    "controlled_antisurge": ("surface_pr_controlled_antisurge", "controlled"),
    "sectional_physical": ("surface_pr_sectional_physical", "equilibrium"),
    "sectional_mixer": ("surface_pr_sectional_mixer", "equilibrium"),
    "sectional_transport": ("surface_pr_sectional_transport", "equilibrium"),
    "holmes_inclined": ("surface_pr_holmes_inclined", "equilibrium"),
    "hem_gas": ("surface_pr_hem_gas", "equilibrium"),
    "hem_two_phase": ("surface_pr_hem_two_phase", "equilibrium"),
    "hem_flashing": ("surface_pr_hem_flashing", "equilibrium"),
    "hem_phase_boundary": ("surface_pr_hem_phase_boundary", "equilibrium"),
    "hem_three_phase": ("surface_pr_hem_three_phase", "equilibrium"),
    "controlled_hem_throat": ("surface_pr_controlled_hem_throat", "controlled"),
    "dense_emulsion": ("surface_pr_dense_emulsion", "equilibrium"),
    "powered_breakup": ("surface_pr_powered_breakup", "equilibrium"),
    "driven_pump": ("surface_pr_driven_pump", "equilibrium"),
    "pump_shutoff": ("surface_pr_pump_shutoff", "equilibrium"),
    "driven_compressor": ("surface_pr_driven_compressor", "equilibrium"),
    "driven_pump_column": ("surface_pr_driven_pump_column", "gravity"),
    "terrain_three_phase": ("surface_pr_terrain_three_phase", "equilibrium"),
    "coalescing_separator": ("surface_pr_coalescing_separator", "equilibrium"),
    "terrain_trap": ("surface_pr_terrain_trap", "equilibrium"),
    "vertical_transition": ("surface_pr_vertical_transition", "equilibrium"),
    "counterflow": ("surface_pr_counterflow_heat_exchanger", "equilibrium"),
    "bubbly": ("surface_pr_vertical_dispersed_bubbly", "equilibrium"),
    "cap_slug": ("surface_pr_vertical_cap_slug", "equilibrium"),
    "reboiler": ("surface_pr_finite_reboiler", "gravity"),
    "nozzle": ("surface_pr_resolved_nozzle", "equilibrium"),
    "transonic": ("surface_pr_transonic_nozzle", "equilibrium"),
    "oil_water_clean": ("surface_pr_oil_water_clean", "equilibrium"),
    "oil_water_immobile": ("surface_pr_oil_water_immobile", "equilibrium"),
    "water_in_oil": ("surface_pr_water_in_oil", "equilibrium"),
    "pump": ("surface_pr_pump_unit", "equilibrium"),
    "pump_column": ("surface_pr_pump_column", "gravity"),
    "controlled_pump": ("surface_pr_controlled_pump", "controlled"),
    "compressor": ("surface_pr_compressor", "equilibrium"),
    "compressor_choke": ("surface_pr_compressor_choke", "equilibrium"),
    "compressor_column": ("surface_pr_compressor_column", "gravity"),
    "compressor_recycle": ("surface_pr_compressor_recycle", "equilibrium"),
    "resolved_separator": ("surface_pr_resolved_separator", "equilibrium"),
    "retained_liquid": ("surface_pr_retained_liquid", "equilibrium"),
    "controlled_resolved_separator": ("surface_pr_controlled_resolved_separator", "controlled"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.timeout_s <= 3600:
        parser.error("Timeout must be in (0, 3600] seconds")
    if not 1 <= args.jobs <= 8 or (args.device == "cuda" and args.jobs != 1):
        parser.error("Use 1..8 CPU jobs; CUDA verification uses one job per selected device")
    if args.device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"):
        parser.error("Explicitly select the CUDA device with CUDA_VISIBLE_DEVICES")
    root = Path(__file__).resolve().parents[1]
    selected = args.case or list(CASES)
    paths = [
        *sorted((root / "src/surface").glob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
        *(root / "configs" / (CASES[name][0] + ".json") for name in selected),
    ]

    def hashes():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    before = hashes()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    def run_case(name):
        config_name, family = CASES[name]
        config = root / "configs" / (config_name + ".json")
        result = output / (name + ".json")
        stem = {
            "equilibrium": "pr_equilibrium_transport",
            "gravity": "pr_gravity_transport",
            "controlled": "pr_controlled_surface",
            "layered": "pr_layered_transport",
            "layered_plant": "pr_layered_plant",
            "profile_graph": "pr_profile_graph",
            "relaxation": "pr_interphase_relaxation",
            "thermochemical": "pr_thermochemical_relaxation",
            "active_multifluid": "pr_active_multifluid_pipe",
            "multifluid": "pr_multifluid_pipe",
        }[family]
        start = time.monotonic()
        print(f"Running {name} on {args.device}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / f"run_{stem}.py"),
                str(config),
                "--device",
                args.device,
                "--output",
                str(result),
            ],
            cwd=root,
            env=env,
            check=True,
            timeout=args.timeout_s,
        )
        checked = subprocess.run(
            [sys.executable, str(root / "scripts" / f"check_{stem}.py"), str(config), str(result)],
            cwd=root,
            env={**env, "CUDA_VISIBLE_DEVICES": ""},
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_s,
        )
        if checked.returncode:
            raise RuntimeError(f"Independent audit failed for {name}: {checked.stderr[-4000:]}")
        audit = json.loads(checked.stdout)
        if audit.get("accepted") is not True:
            raise RuntimeError(f"Independent audit rejected {name}")
        result_record = {
            "audit": audit,
            "elapsed_s": time.monotonic() - start,
            "report": result.name,
        }
        print(json.dumps({"case": name, **result_record}), flush=True)
        return name, result_record

    # Independent physical scenarios use separate subprocesses and outputs.
    # CUDA remains serial so verification does not oversubscribe a device.
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = dict(executor.map(run_case, selected))
    if hashes() != before:
        raise RuntimeError("Surface sources, verification scripts or configs changed during matrix")
    report = {
        "device": args.device,
        "cases": results,
        "source_sha256": before,
        "qualification": "Functional numerical verification; timings are not throughput benchmarks",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
