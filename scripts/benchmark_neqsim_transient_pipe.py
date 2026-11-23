"""Run an isolated, versioned NeqSim transient pipe; not a whole-facility comparison.

Install neqsim into a separate directory with pip --target, then pass that path.
No external package is added to the simulator runtime or requirements.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--length-m", type=float, default=10.0)
    parser.add_argument("--diameter-m", type=float, default=0.1)
    parser.add_argument("--duration-s", type=float, default=0.05)
    parser.add_argument("--thermo-interval", type=int, default=1)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument(
        "--transport",
        choices=("fixed-phase", "compositional", "conservative-energy"),
        default="fixed-phase",
    )
    parser.add_argument("--mixture", choices=("methane", "methane-pentane"), default="methane")
    parser.add_argument("--pressure-method", choices=("coupled", "legacy"), default="coupled")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.cells, args.length_m, args.diameter_m, args.duration_s, args.repeats) <= 0:
        parser.error("Positive dimensions, duration and repeat count required")
    jars = list((args.library_root / "neqsim/lib").glob("neqsim-*.jar"))
    if len(jars) != 1:
        parser.error("Expected exactly one versioned NeqSim JAR")
    sys.path.insert(0, str(args.library_root))
    import jpype

    boot = time.perf_counter()
    jpype.startJVM("-Xmx1g", "-XX:ActiveProcessorCount=1", classpath=[str(jars[0])])
    boot = time.perf_counter() - boot
    cls = jpype.JClass
    boundary = cls("neqsim.process.equipment.pipeline.TwoFluidPipe$BoundaryCondition")
    phase = cls("neqsim.process.equipment.pipeline.TwoFluidMassBalanceReport$Phase")
    repeats = []
    # First run is retained separately; subsequent independent models share JVM/JIT warmup.
    for repeat in range(args.repeats + 1):
        start = time.perf_counter()
        fluid = cls("neqsim.thermo.system.SystemPrEos")(300.0, 50.0)
        fluid.addComponent("methane", 1.0 if args.mixture == "methane" else 0.8)
        if args.mixture != "methane":
            fluid.addComponent("n-pentane", 0.2)
        fluid.setMixingRule("classic")
        fluid.setMultiPhaseCheck(True)
        inlet = cls("neqsim.process.equipment.stream.Stream")("feed", fluid)
        # A positive stream basis is necessary for NeqSim specific enthalpy.
        # At equal end pressures its tiny initial velocity approximates rest;
        # it is explicitly reported rather than described as exact zero flow.
        inlet.setFlowRate(1e-8 * (args.diameter_m / 0.1) ** 2, "kg/sec")
        inlet.run()
        pipe = cls("neqsim.process.equipment.pipeline.TwoFluidPipe")("pipe", inlet)
        pipe.setLength(args.length_m)
        pipe.setDiameter(args.diameter_m)
        pipe.setNumberOfSections(args.cells)
        pipe.setRoughness(0.0)
        pipe.setCflNumber(args.cfl)
        pipe.setInletBoundaryCondition(boundary.CONSTANT_PRESSURE)
        pipe.setOutletBoundaryCondition(boundary.CONSTANT_PRESSURE)
        # In the pinned Java API the one-argument setters accept Pa, not bara.
        pipe.setInletPressure(5e6)
        pipe.setOutletPressure(5e6)
        pipe.setOutletBoundaryFluid(fluid)
        pipe.setAllowOutletPhaseBackflow(True)
        pipe.setIncludeEnergyEquation(True)
        pipe.setHeatTransferCoefficient(0.0)
        pipe.setThermodynamicUpdateInterval(args.thermo_interval)
        pipe.setMaximumTransientSubsteps(100000)
        if args.pressure_method == "coupled":
            pipe.setEnableInterfacialPressure(True)
            pipe.setEnableCoupledPressureMomentum(True)
        if args.transport != "fixed-phase":
            pipe.setComponentTransportEnabled(True)
            pipe.setIncludeMassTransfer(True)
        if args.transport == "conservative-energy":
            pipe.setEnableSlugTracking(False)
            pipe.setUseConservativeEnergyRecovery(True)
            pipe.setTimeIntegrationMethod(
                cls(
                    "neqsim.process.equipment.pipeline.twophasepipe.numerics.TimeIntegrator$Method"
                ).EULER
            )
        pipe.run()
        initialization = time.perf_counter() - start
        if not pipe.isSteadyStateConverged():
            raise RuntimeError("NeqSim steady initialization did not converge")
        initial_pressure = list(pipe.getPressureProfile())
        initial_temperature = list(pipe.getTemperatureProfile())
        initial_holdup = list(pipe.getLiquidHoldupProfile())
        if max(abs(v - 5e6) for v in initial_pressure) > 1.0:
            raise RuntimeError("Initial pressure differs from the requested rest state")
        pipe.setInletPressure(5.001e6)
        token = cls("java.util.UUID").randomUUID()
        before = pipe.getSimulationTime()
        start = time.perf_counter()
        try:
            pipe.runTransient(args.duration_s, token)
        except Exception as exc:
            failure = {
                "status": "failed_not_qualified",
                "engine": jars[0].stem,
                "jar_sha256": hashlib.sha256(jars[0].read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "request": {
                    k: v for k, v in vars(args).items() if k not in ("library_root", "output")
                },
                "repeat": repeat,
                "transient_wall_s_before_failure": time.perf_counter() - start,
                "simulated_s_before_failure": pipe.getSimulationTime() - before,
                "exception": str(exc),
                "completed_repeats": repeats,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
            raise
        elapsed = time.perf_counter() - start
        advanced = pipe.getSimulationTime() - before
        if not math.isclose(advanced, args.duration_s, rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError(f"Incomplete transient: {advanced} of {args.duration_s} s")
        mass = pipe.getLastMassBalanceReport()
        ledgers = {
            str(p): {
                "residual_kg": mass.getResidualKg(p),
                "relative_residual": mass.getRelativeResidual(p),
            }
            for p in phase.values()
        }
        energy = pipe.getLastThermalEnergyBalanceReport()
        conservative = pipe.getLastConservativeEnergyReport()
        components = pipe.getLastComponentConservationReport()
        record = {
            "repeat": repeat,
            "initialization_wall_s": initialization,
            "transient_wall_s": elapsed,
            "simulated_s": advanced,
            "accepted_substeps": mass.getAcceptedSubsteps(),
            "steady_converged": bool(pipe.isSteadyStateConverged()),
            "initial_pressure_pa": initial_pressure,
            "initial_temperature_k": initial_temperature,
            "initial_liquid_holdup": initial_holdup,
            "final_pressure_pa": list(pipe.getPressureProfile()),
            "final_temperature_k": list(pipe.getTemperatureProfile()),
            "final_liquid_holdup": list(pipe.getLiquidHoldupProfile()),
            "mass_ledgers": ledgers,
            "thermal_ledger": None
            if energy is None
            else {
                "residual_j": energy.getResidualJ(),
                "relative_residual": energy.getRelativeResidual(),
            },
            "conservative_energy_ledger": None
            if conservative is None
            else {
                "residual_j": conservative.getResidualJ(),
                "relative_residual": conservative.getRelativeResidual(),
                "max_relative_volume_residual": conservative.getMaxRelativeVolumeResidual(),
                "max_relative_thermodynamic_energy_residual": conservative.getMaxRelativeThermodynamicEnergyResidual(),
            },
            "component_ledger": None
            if components is None
            else {
                "converged": bool(components.isConverged()),
                "names": [str(v) for v in components.getComponentNames()],
                "residuals_kg": list(components.getInventoryResidualKg()),
                "max_relative_residual": components.getMaximumRelativeInventoryResidual(),
                "max_phase_mass_synchronization_error_kg": components.getMaximumPhaseMassSynchronizationErrorKg(),
            },
            "pressure_correction_converged": bool(pipe.isCoupledPressureMomentumConverged()),
            "pressure_correction_relative_volume_residual": pipe.getCoupledPressureMomentumVolumeResidual(),
            "integration_method": str(pipe.getTimeIntegrationMethod()),
            "inlet_phase_count": inlet.getFluid().getNumberOfPhases(),
        }
        repeats.append(record)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in [
                        "repeat",
                        "transient_wall_s",
                        "accepted_substeps",
                        "inlet_phase_count",
                    ]
                }
            ),
            flush=True,
        )
    report = {
        "scope": "External isolated pipe workload; different discretization/thermodynamic scheduling, not a speedup ratio against the coupled facility",
        "engine": jars[0].stem,
        "status": "completed_workload_not_experimental_validation",
        "jar_sha256": hashlib.sha256(jars[0].read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "java_version": str(cls("java.lang.System").getProperty("java.version")),
        "platform": platform.platform(),
        "jvm_active_processors": 1,
        "jvm_boot_wall_s": boot,
        "request": {k: v for k, v in vars(args).items() if k not in ("library_root", "output")},
        "cold": repeats[0],
        "warm": repeats[1:],
        "warm_median_transient_wall_s": statistics.median(
            r["transient_wall_s"] for r in repeats[1:]
        ),
        "limitations": [
            "No separator, finite nozzle, capillary film, controller or reservoir is included",
            "NeqSim has its own caloric data, phase closures, boundary placement and CFL control",
            "Thermal ledger is NeqSim's discrete thermal balance, not an independent full-EOS total-energy audit",
            "Initial stream basis is positive and tiny rather than exactly zero",
            "fixed-phase disables named-component transport and phase mass transfer; compositional enables both; conservative-energy additionally selects the experimental coupled-Euler full-energy recovery",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
