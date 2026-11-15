"""Work/precision comparison: native Torch liquid DAE vs audited midpoint.

Requires scikit-sundae==1.2.0 only for this optional benchmark. This is NOT
the multicomponent PR facility and is NOT a general production IDA backend.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch
from check_liquid_transient import coupled_model, coupled_rhs
from surface.torch_liquid import TorchLiquidEquations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--legacy-newton", action="store_true")
    args = parser.parse_args()
    if args.dependencies:
        sys.path.insert(0, str(args.dependencies))
    from sksundae.ida import IDA
    import sksundae
    if args.legacy_newton:
        from surface import torch_liquid
        from surface.torch_linear import TorchNewton as LegacyNewton
        torch_liquid._CpuNewton = LegacyNewton
    if args.pairs < 1:
        parser.error("positive pair count required")
    torch.set_num_threads(1)
    y0 = np.array([550000., 500000., .002, 100., 110.])
    budget = np.array([1., 1., 1e-8, 1e-4, 1e-4])
    reference = solve_ivp(coupled_rhs, (0., 1.), y0, method="DOP853",
                          rtol=1e-12, atol=[1e-7, 1e-7, 1e-15, 1e-11, 1e-11], dense_output=True)
    assert reference.success
    model = coupled_model(.001)
    e = TorchLiquidEquations(model)
    fixed = e.tensor([{"in": 400000., "out": 480000.}.get(n, 500000.) for n in model._nodes])
    commands, openings = e.tensor([5., 7.]), e.tensor([1.])
    scale = np.r_[[e.ps]*2, [e.qs]*4, e.nominal.numpy()]
    differential = [0, 1, 2, 6, 7]

    def lift(y):
        ph, pt, q, wa, wb = y
        w = np.array([wa, wb])
        qp = np.sqrt(np.maximum(0., 1000*9.80665*np.array([20, 22])*(w/[100, 110])**2-(ph-400000))
                     / (1000*9.80665*np.array([1e6, 1.2e6])))
        qv = .7*math.pi*.02**2/4*math.copysign(math.sqrt(2*abs(pt-480000)/1000), pt-480000)
        return np.r_[ph, pt, q, qp, qv, w]/scale

    def residual(y, yp, *, physical=False):
        p, q, w = y[:2]*e.ps, y[2:3]*e.qs, y[6:]*e.nominal
        ws = w.clamp_min(1.)
        x = torch.cat((y[:6], w/ws))
        row, data = e.evaluate(x, p, q, w, 1., commands, fixed, openings)
        mass = torch.cat((y.new_ones(3), y.new_zeros(3), e.nominal/ws))
        value = row + mass*yp
        return (value, data, mass) if physical else value

    # Prove the dt=1 extraction is instantaneous only in this inactive-limit
    # regime. This benchmark never claims it handles held shafts or events.
    equivalence = []
    for t in np.linspace(0., 1., 41):
        y = reference.sol(t)
        z = e.tensor(lift(y))
        zp = torch.zeros_like(z)
        zp[differential] = e.tensor(coupled_rhs(t, y)/scale[differential])
        r, data, _ = residual(z, zp, physical=True)
        re = abs(y[2])*1000*.05/(.1*(math.pi*.05**2/4))
        assert re < 2000 and min(y[3:]) > 1
        assert torch.equal(data[6], commands), "Allocation limits invalidate extraction"
        assert min(z[3:5]).item() > 0, "Pump check-valve branch changed"
        equivalence.append(float(r.abs().max()))
    assert max(equivalence) < 1e-12
    z0 = lift(y0)
    zp0 = np.zeros(8)
    zp0[differential] = coupled_rhs(0., y0)/scale[differential]
    # Exclude one-time Torch forward-AD setup from every timed case.
    torch.func.jacfwd(lambda z: residual(z, e.tensor(zp0)))(e.tensor(z0))

    def metrics(times, states):
        error = np.max(np.abs(states-reference.sol(times).T), axis=0)
        return {"max_abs_errors": error.tolist(), "worst_budget_fraction": float(max(error/budget)),
                "within_example_budget": bool(np.all(error <= budget))}

    def ida():
        counts = {"residual": 0, "jacobian": 0}
        def resfn(t, y, yp, out):
            counts["residual"] += 1
            out[:] = residual(e.tensor(y.copy()), e.tensor(yp.copy())).detach().numpy()
        def jacfn(t, y, yp, res, cj, out):
            counts["jacobian"] += 1
            z, zp = e.tensor(y.copy()), e.tensor(yp.copy())
            jac = torch.func.jacfwd(lambda v: residual(v, zp))(z)
            mass = residual(z, zp, physical=True)[2]
            out[:] = (jac + cj*torch.diag(mass)).detach().numpy()
        start = time.perf_counter()
        solver = IDA(resfn, jacfn=jacfn, algebraic_idx=[3, 4, 5],
                     rtol=1e-8, atol=1e-10, max_num_steps=10000, max_nonlin_iters=8)
        first = solver.init_step(0., z0.copy(), zp0.copy())
        assert first.success
        times, states, torque_defects, residuals = [0.], [y0.tolist()], [], []
        for _ in range(10000):
            r = solver.step(1., method="onestep", tstop=1.)
            assert r.success, r.message
            times.append(float(r.t))
            states.append((r.y*scale)[differential].tolist())
            # Checks outside the solver are included in timing and retained.
            value, data, _ = residual(e.tensor(r.y.copy()), e.tensor(r.yp.copy()), physical=True)
            torque_defects.append(float((data[6]-commands).abs().max()))
            residuals.append(float(value.abs().max()))
            if r.t >= 1.:
                break
        else:
            raise RuntimeError("IDA step budget exhausted")
        elapsed = time.perf_counter()-start
        assert max(torque_defects) == 0.
        times, states = np.array(times), np.array(states)
        steps = np.diff(times)
        return dict(method="IDA_native_torch_dae", elapsed_s=elapsed, accepted_steps=len(steps),
                    median_step_s=float(np.median(steps)), maximum_step_s=float(max(steps)),
                    calls=counts, maximum_dae_residual=max(residuals),
                    times_s=times.tolist(), states=states.tolist(), **metrics(times, states))

    def midpoint(adaptive=False):
        m = coupled_model(.02 if adaptive else .001)
        if adaptive:
            spec = m.to_dict()
            spec.update(temporal_error_control=True, relative_tolerance=1e-8,
                        pressure_tolerance_pa=.1, flow_tolerance_m3_s=1e-11, speed_tolerance_rad_s=1e-6)
            m = type(m).from_dict(spec)
        state = m.initialize({"h":550000., "t":500000.}, {"line":.002}, {"a":100., "b":110.})
        start = time.perf_counter()
        report = m.advance(state, 1., motor_torques_nm={"a":5., "b":7.})
        elapsed = time.perf_counter()-start
        times, states = [0.], [y0.tolist()]
        for step in report["steps"]:
            times.append(step["time_s"]+step["dt_s"])
            states.append([step["final_pressures_pa"][n] for n in ("h", "t")]
                          + [step["final_pipe_volume_rates_m3_s"]["line"]]
                          + [step["final_speeds_rad_s"][n] for n in ("a", "b")])
        return dict(method="midpoint_adaptive" if adaptive else "midpoint_fixed", elapsed_s=elapsed,
                    accepted_steps=len(report["steps"]), times_s=times, states=states,
                    max_energy_residual_j=max(abs(s["energy_residual_j"]) for s in report["steps"]),
                    **metrics(np.array(times), np.array(states)))

    results = []
    for pair in range(args.pairs):
        for function in ((ida, midpoint) if pair % 2 == 0 else (midpoint, ida)):
            row = function()
            row["pair"] = pair
            results.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ("times_s", "states")}), flush=True)
    results.append(midpoint(True))
    output = dict(scope="One-second single-phase liquid/pump example, NOT the PR facility",
                  legacy_newton=args.legacy_newton,
                  example_error_budget=budget.tolist(), equivalence_max_residual=max(equivalence),
                  versions=dict(torch=torch.__version__, scipy=scipy.__version__, sksundae=sksundae.__version__),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT/"src/surface").glob("*.py"))}, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False))
    assert all(r["within_example_budget"] for r in results)


if __name__ == "__main__":
    main()
