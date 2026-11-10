"""Newton on the nonlinear Schur residual of the complete finite facility.

Pipe and chemical-profile unknowns are eliminated by their existing checked
solves. The remaining unknowns are the actual N/E stocks of every column.
The joint Jacobian differentiates the complete end-profile flow map; it is
not a collection of independent per-column corrections. Its current tangent
is a scaled finite difference, with an explicit small-system size budget.
"""

import math
from dataclasses import replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
from .stratified_pr import StratifiedPRState


class FacilitySchurProblem:
    def __init__(self, facility, state, dt_s, normal_relaxation, heaters, phase_controls=None):
        self.facility, self.state, self.dt_s = facility, state, dt_s
        self.commands = {
            "normal_relaxation": normal_relaxation,
            "heaters": heaters,
            "phase_controls": phase_controls,
        }
        self.names = tuple(facility.columns)
        self.fluid = facility.columns[self.names[0]].fluid
        self.nc = len(self.fluid.names)
        self.size = len(self.names) * (self.nc + 1)
        if self.size > facility.maximum_schur_unknowns:
            raise ValueError("Facility Schur unknowns exceed the explicitly configured budget")
        self.evaluations = 0
        # A real conservative trial gives a finite thermal/phase predictor.
        # In particular, large imposed heat is not differentiated across an
        # unrelated old phase branch when constructing the first tangent.
        self.seed, self.seed_report = self.trial(state)
        self.reference = self.vector(self.seed)
        columns = dict(self.seed.columns)
        scales = []
        for name in self.names:
            c = columns[name]
            total = math.fsum(c.component_moles)
            scales.extend(max(v, total * 1e-8) for v in c.component_moles)
            scales.append(
                max(1.0, abs(c.total_energy_j), total * 8.31446261815324 * c.temperature_k)
            )
        # These are units/scales only. No stock is added to any phase/column.
        self.scale = self.fluid.tensor(scales)
        self.jacobian_builds = 0
        self.last_evaluation = None
        self.cache_hits = 0

    def vector(self, state):
        columns = dict(state.columns)
        return self.fluid.tensor(
            [
                v
                for n in self.names
                for v in (*columns[n].component_moles, columns[n].total_energy_j)
            ]
        )

    def trial(self, boundary):
        self.evaluations += 1
        return self.facility._trial(self.state, self.dt_s, boundary_state=boundary, **self.commands)

    def decode(self, point):
        if bool((point == 0).all().cpu()):
            return replace(self.state, columns=self.seed.columns, layer_slots=self.seed.layer_slots)
        values = (
            (self.reference + self.scale * point)
            .reshape(len(self.names), self.nc + 1)
            .cpu()
            .tolist()
        )
        old, bindings = dict(self.seed.columns), dict(self.seed.layer_slots)
        columns, slots = [], []
        for name, row in zip(self.names, values):
            if min(row[: self.nc]) < 0 or not math.fsum(row[: self.nc]) > 0:
                raise PropertyModelError("Schur candidate has negative or empty column stock")
            column = self.facility.columns[name]
            base = old[name]
            result = column.equilibrate(
                row[: self.nc],
                row[-1],
                pressure_guess_pa=math.exp(base.solution_variables[0]),
                temperature_guess_k=base.temperature_k,
            )
            candidate = StratifiedPRState.from_dict(result["state"])
            line, side = self.facility.column_ports[name][0]
            adapter = self.facility.adapters[line, side]
            # _match_layers reads only the actual previous profile/registry.
            from .pr_capillary_column import PRCapillaryColumnState

            previous = PRCapillaryColumnState(
                dict(self.state.pipes)[line],
                base,
                self.state.next_step_s,
                adapter.model_id,
                bindings[name],
            )
            columns.append((name, candidate))
            slots.append((name, adapter._match_layers(previous, candidate)))
        return replace(self.state, columns=tuple(columns), layer_slots=tuple(slots))

    def evaluate(self, point):
        key = tuple(point.detach().cpu().tolist())
        if self.last_evaluation is not None and self.last_evaluation[0] == key:
            self.cache_hits += 1
            return self.last_evaluation[1]
        boundary = self.decode(point)
        end, report = self.trial(boundary)
        residual = (self.vector(boundary) - self.vector(end)) / self.scale
        if not bool(torch.isfinite(residual).all().cpu()):
            raise HydraulicConvergenceError("Nonfinite global Schur residual")
        result = residual, end, report
        self.last_evaluation = key, result
        return result

    def jacobian(self, point, residual, *, step=1e-4):
        """Joint central difference; one-sided only at a real stock boundary.

        The perturbation is dimensionless, not kilograms or Joules. Reusing
        a tangent is chord Newton; final acceptance still requires the exact
        end-profile fixed point and the original physical balance checks.
        """
        columns = []
        for j in range(self.size):
            direction = torch.zeros_like(point)
            direction[j] = step
            try:
                plus = self.evaluate(point + direction)[0]
            except (PropertyModelError, HydraulicConvergenceError, ConservationError):
                minus = self.evaluate(point - direction)[0]
                columns.append((residual - minus) / step)
                continue
            try:
                minus = self.evaluate(point - direction)[0]
            except (PropertyModelError, HydraulicConvergenceError, ConservationError):
                columns.append((plus - residual) / step)
            else:
                columns.append((plus - minus) / (2 * step))
        matrix = torch.stack(columns, dim=1)
        if not bool(torch.isfinite(matrix).all().cpu()):
            raise HydraulicConvergenceError("Nonfinite global Schur Jacobian")
        self.jacobian_builds += 1
        return matrix


def solve_schur(
    facility, state, dt_s, *, normal_relaxation=True, heaters=None, phase_controls=None
):
    problem = FacilitySchurProblem(
        facility, state, dt_s, normal_relaxation, heaters, phase_controls
    )
    point = torch.zeros_like(problem.reference)
    matrix = None
    linear_solves = 0
    maximum_linear_defect = 0.0
    polishing = False
    for iteration in range(facility.maximum_coupling_iterations):
        residual, end, report = problem.evaluate(point)
        if report["coupling_converged"]:
            report["nonlinear_coupling_iterations"] = iteration + 1
            report["schur_newton"] = {
                "method": "scaled_central_difference_nonlinear_schur_chord_newton",
                "unknowns": problem.size,
                "jacobian_builds": problem.jacobian_builds,
                "linear_solves": linear_solves,
                "flow_map_evaluations": problem.evaluations,
                "last_point_cache_hits": problem.cache_hits,
                "maximum_linear_defect": maximum_linear_defect,
                "finite_difference_step": 1e-4,
                "exact_profile_polishing": polishing,
                "matrix_device": str(problem.fluid.device),
            }
            return end, report
        norm = float(residual.abs().max().cpu())
        if iteration + 1 == facility.maximum_coupling_iterations:
            break
        if norm < 2e-12 or polishing:
            # A small stock residual alone does not certify the profiles.
            # Polish on the complete state, retaining exact physical ledgers.
            polishing = True
            problem.seed = end
            problem.reference = problem.vector(end)
            problem.last_evaluation = None
            point = torch.zeros_like(point)
            continue
        if matrix is None:
            matrix = problem.jacobian(point, residual)
        correction, info = torch.linalg.solve_ex(matrix, -residual, check_errors=False)
        defect = matrix @ correction + residual
        maximum_linear_defect = max(maximum_linear_defect, float(defect.abs().max().cpu()))
        if not bool(
            (
                (info == 0)
                & torch.isfinite(correction).all()
                & (defect.abs().max() <= 1e-9 * residual.abs().max() + 1e-14)
            ).cpu()
        ):
            raise HydraulicConvergenceError("Global Schur Newton linear correction failed")
        linear_solves += 1
        accepted = False
        for power in range(8):
            trial = point + correction * 2.0 ** (-power)
            try:
                test, _, checked = problem.evaluate(trial)
            except (PropertyModelError, HydraulicConvergenceError, ConservationError):
                continue
            if checked["coupling_converged"] or float(test.abs().max().cpu()) < norm:
                point = trial
                accepted = True
                break
        if not accepted:
            raise HydraulicConvergenceError("Global Schur Newton line search failed")
    raise HydraulicConvergenceError("Global Schur Newton coupling iteration budget exhausted")
