"""Diagnostic NeqSim pipe/finite receiver handoff, not our full-facility benchmark.

This deliberately tests ordinary outlet-stream wiring plus staggered pressure
feedback. Shared-boundary mass and energy must close before its runtime can be
used as conservation-qualified evidence. No external library joins our runtime.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path


def setup(cls, args):
    fluid = cls("neqsim.thermo.system.SystemPrEos")(300.0, 50.0)
    fluid.addComponent("methane", 0.8)
    fluid.addComponent("n-pentane", 0.2)
    fluid.setMixingRule("classic")
    fluid.setMultiPhaseCheck(True)
    stream = cls("neqsim.process.equipment.stream.Stream")("feed", fluid)
    stream.setFlowRate(1e-8, "kg/sec")
    stream.run()
    pipe = cls("neqsim.process.equipment.pipeline.TwoFluidPipe")("pipe", stream)
    boundary = cls("neqsim.process.equipment.pipeline.TwoFluidPipe$BoundaryCondition")
    pipe.setInletBoundaryCondition(boundary.CONSTANT_PRESSURE)
    pipe.setOutletBoundaryCondition(boundary.CONSTANT_PRESSURE)
    settings = {
        "Length": 10.0,
        "Diameter": 0.1,
        "NumberOfSections": args.cells,
        "Roughness": 0.0,
        "CflNumber": 0.5,
        "InletPressure": 5e6,
        "OutletPressure": 5e6,
        "AllowOutletPhaseBackflow": True,
        "IncludeEnergyEquation": True,
        "HeatTransferCoefficient": 0.0,
        "ThermodynamicUpdateInterval": 1,
        "MaximumTransientSubsteps": 100000,
        "EnableInterfacialPressure": True,
        "EnableCoupledPressureMomentum": True,
        "ComponentTransportEnabled": True,
        "IncludeMassTransfer": True,
        "EnableSlugTracking": False,
        "UseConservativeEnergyRecovery": True,
    }
    pipe.setOutletBoundaryFluid(fluid)
    for name, value in settings.items():
        getattr(pipe, "set" + name)(value)
    pipe.setTimeIntegrationMethod(
        cls("neqsim.process.equipment.pipeline.twophasepipe.numerics.TimeIntegrator$Method").EULER
    )
    pipe.run()
    if not pipe.isSteadyStateConverged():
        raise ValueError("Pipe steady initialization did not converge")
    receiver = cls("neqsim.process.equipment.separator.Separator")("receiver", pipe.getOutletStream())
    receiver.setInternalDiameter(args.receiver_diameter_m)
    receiver.setSeparatorLength(args.receiver_length_m)
    receiver.setOrientation("vertical")
    receiver.setLiquidLevel(0.5)
    receiver.setCalculateSteadyState(False)
    receiver.setConservativeTransientEnergyIntegrationEnabled(True)
    receiver.setTransientEnergyBalanceTrackingEnabled(True)
    receiver.run()
    return pipe, receiver


def mass(receiver):
    fluid = receiver.getThermoSystem()
    return fluid.getTotalNumberOfMoles() * fluid.getMolarMass()


def trial(cls, args):
    started = time.perf_counter()
    pipe, receiver = setup(cls, args)
    initial_mass = mass(receiver)
    initial_energy = receiver.getThermoSystem().getInternalEnergy()
    initialization = time.perf_counter() - started
    initial_phases = receiver.getThermoSystem().getNumberOfPhases()
    total = cls("neqsim.process.equipment.pipeline.TwoFluidMassBalanceReport$Phase").TOTAL
    clock = pipe.getSimulationTime()
    pipe.setInletPressure(5.001e6)
    pipe_mass = pipe_energy = None
    external_mass = external_energy = 0.0
    mass_gap = energy_gap = 0.0
    absolute_boundary_mass = absolute_boundary_energy = 0.0
    reverse_calls = 0
    rows = []
    pipe_wall = receiver_wall = 0.0
    started = time.perf_counter()
    failure = None
    try:
        while clock < args.duration_s - 8 * math.ulp(max(1.0, args.duration_s)):
            dt = min(args.communication_step_s, args.duration_s - clock)
            token = cls("java.util.UUID").randomUUID()
            pipe.setOutletPressure(receiver.getPressure() * 1e5)
            pipe.setOutletBoundaryFluid(receiver.getThermoSystem())
            before_mass = mass(receiver)
            receiver_out = dt * (
                receiver.getGasOutStream().getFlowRate("kg/sec")
                + receiver.getLiquidOutStream().getFlowRate("kg/sec")
            )
            before = time.perf_counter()
            pipe.runTransient(dt, token)
            pipe_wall += time.perf_counter() - before
            ledger = pipe.getLastMassBalanceReport()
            energy = pipe.getLastConservativeEnergyReport()
            if pipe_mass is None:
                pipe_mass, pipe_energy = ledger.getInitialMassKg(total), energy.getInitialEnergyJ()
            transfer = ledger.getOutletMassKg(total)
            reverse_calls += transfer < 0
            before = time.perf_counter()
            receiver.runTransient(dt, token)
            receiver_wall += time.perf_counter() - before
            se = receiver.getLastTransientEnergyBalanceReport()
            external_mass += ledger.getInletMassKg(total) + ledger.getSourceMassKg(total) - receiver_out
            external_energy += (
                energy.getInletAdvectedEnergyJ() + energy.getWallHeatToFluidJ()
                + energy.getGravitationalWorkJ() - se.getGasOutletEnergyJ() - se.getLiquidOutletEnergyJ()
            )
            absolute_boundary_mass += abs(ledger.getInletMassKg(total)) + abs(receiver_out)
            absolute_boundary_energy += (
                abs(energy.getInletAdvectedEnergyJ()) + abs(se.getGasOutletEnergyJ())
                + abs(se.getLiquidOutletEnergyJ())
            )
            after_mass = mass(receiver)
            mass_gap += after_mass - before_mass - transfer + receiver_out
            energy_gap += se.getInletEnergyJ() - energy.getOutletAdvectedEnergyJ()
            clock = pipe.getSimulationTime()
            rows.append({
                "time_s": clock, "native_receiver_clock_s": receiver.getTime(),
                "receiver_tracked_elapsed_s": se.getElapsedTimeSeconds(),
                "receiver_pressure_pa": receiver.getPressure() * 1e5,
                "receiver_level_fraction": receiver.getLiquidLevel(),
                "pipe_substeps": ledger.getAcceptedSubsteps(),
                "outlet_transfer_kg": transfer,
                "global_mass_residual_kg": ledger.getFinalMassKg(total) + after_mass
                    - pipe_mass - initial_mass - external_mass,
                "global_energy_residual_j": energy.getFinalEnergyJ()
                    + receiver.getThermoSystem().getInternalEnergy()
                    - pipe_energy - initial_energy - external_energy,
                "pipe_mass_residual_kg": ledger.getResidualKg(total),
                "pipe_energy_residual_j": energy.getResidualJ(),
                "receiver_relative_energy_residual": se.getRelativeCumulativeResidual(),
                "cumulative_mass_handoff_gap_kg": mass_gap,
                "cumulative_energy_handoff_gap_j": energy_gap,
            })
            if not all(math.isfinite(v) for v in rows[-1].values()):
                raise ValueError("Nonfinite connected diagnostic")
    except (ValueError, RuntimeError, cls("java.lang.Exception")) as exc:
        failure = str(exc)
    elapsed = time.perf_counter() - started
    complete = math.isclose(clock, args.duration_s, rel_tol=1e-10, abs_tol=1e-12) and failure is None
    maximum_mass = max((abs(r["global_mass_residual_kg"]) for r in rows), default=None)
    maximum_energy = max((abs(r["global_energy_residual_j"]) for r in rows), default=None)
    # Public diagnostic gates use exchanged quantities, not a large initial
    # inventory to hide an interface-transfer error. These are not universal
    # experimental tolerances or equivalence to the full project model.
    mass_budget = 1e-8 + 1e-8 * absolute_boundary_mass
    energy_budget = 1e-4 + 1e-8 * absolute_boundary_energy
    return {
        "complete_requested_interval": complete, "failure": failure,
        "initialization_wall_s": initialization, "connected_loop_wall_s": elapsed,
        "pipe_calls_wall_s": pipe_wall, "receiver_calls_wall_s": receiver_wall,
        "simulated_s": clock, "initial_receiver_phases": initial_phases,
        "reverse_transfer_calls": reverse_calls,
        "maximum_global_mass_residual_kg": maximum_mass,
        "maximum_global_energy_residual_j": maximum_energy,
        "diagnostic_mass_budget_kg": mass_budget, "diagnostic_energy_budget_j": energy_budget,
        "coupling_checks_passed": complete and maximum_mass <= mass_budget and maximum_energy <= energy_budget,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--communication-step-s", type=float, default=0.01)
    parser.add_argument("--receiver-diameter-m", type=float, default=1.0)
    parser.add_argument("--receiver-length-m", type=float, default=2.0)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (
        args.duration_s, args.communication_step_s, args.receiver_diameter_m, args.receiver_length_m,
    )) or args.cells < 1 or args.repeats < 1:
        parser.error("Finite positive geometry, duration, steps, cells and repeats required")
    if args.output.exists():
        raise FileExistsError(args.output)
    jars = list((args.library_root / "neqsim/lib").glob("neqsim-*.jar"))
    if len(jars) != 1:
        parser.error("Exactly one versioned NeqSim JAR required")
    sys.path.insert(0, str(args.library_root))
    import jpype

    jpype.startJVM("-Xmx1g", "-XX:ActiveProcessorCount=1", classpath=[str(jars[0])])
    results = []
    for index in range(args.repeats + 1):
        record = trial(jpype.JClass, args)
        results.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "rows"}), flush=True)
        if not record["complete_requested_interval"]:
            break
    complete = len(results) == args.repeats + 1 and all(r["complete_requested_interval"] for r in results)
    report = {
        "scope": __doc__, "engine": jars[0].stem,
        "jar_sha256": hashlib.sha256(jars[0].read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "java": str(jpype.JClass("java.lang.System").getProperty("java.version")),
        "request": {k: v for k, v in vars(args).items() if k not in ("output", "library_root")},
        "cold": results[0], "warm": results[1:],
        "warm_median_diagnostic_loop_wall_s": statistics.median(r["connected_loop_wall_s"] for r in results[1:]) if complete else None,
        "complete_requested_intervals": complete,
        "coupling_checks_passed": complete and all(r["coupling_checks_passed"] for r in results),
        "equivalent_full_facility_speed_comparison": False,
        "limitations": [
            "Different geometry/mixture/closures from our 16-node, 1-mm-recycle facility",
            "No gravity-resolved column, finite nozzles, pumps, compressor, controllers or reservoir",
            "Staggered pressure feedback; ordinary outlet stream is not a proven signed energy/component bridge",
            "Internal unit ledgers alone cannot qualify whole-connected conservation or time accuracy",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report["coupling_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
