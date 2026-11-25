"""Explicit engineering physics selection; independent of numerical tolerances.

A project contains named studies with existing backend inputs. It is not an
automatic converter between black-oil, PR, linepack and inertial checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

MODES = {
    "forecast": {
        "purpose": "Reservoir development and production forecast",
        "omitted": ["pipe momentum transients", "dynamic vessel levels", "actuator/PI transients"],
    },
    "operations": {
        "purpose": "Finite fluid/thermal inventories with quasi-steady pipe hydraulics",
        "omitted": [
            "pipe momentum inventory",
            "acoustic waves",
            "phase slip",
            "reservoir dynamics",
        ],
    },
    "transient": {
        "purpose": "Inertial PR plant dynamics and explicit controls",
        "omitted": [
            "reservoir dynamics in this backend",
            "unqualified experimental operating regimes",
        ],
    },
}

BACKENDS = {
    "fim_facility": {
        "mode": "forecast",
        "features": [
            "reservoir_dynamics",
            "well_surface_feedback",
            "facility_limits",
            "component_conservation",
            "blackoil",
            "quasi_steady_hydraulics",
        ],
        "limitations": [
            "Producing homogeneous gathering tree; one PVT region",
            "Physical adapter excludes VFP/MSW/groups, perf backflow and zero-flow holdup",
            "No dynamic thermal/level inventories of the facility",
        ],
        "audit": "independent component accounting and facility constraints; no field calibration",
    },
    "pr_linepack": {
        "mode": "operations",
        "features": [
            "pipe_storage",
            "vessel_storage",
            "wall_storage",
            "pr_thermodynamics",
            "component_conservation",
            "energy_conservation",
            "quasi_steady_hydraulics",
        ],
        "limitations": [
            "Horizontal 1-D pipes with specified viscosity and Darcy faces",
            "No fluid momentum/kinetic storage, acoustics or phase slip",
            "No actuators/PI or reservoir in this backend",
        ],
        "audit": "independent PR/profile, Darcy and component/energy accounting; no field calibration",
    },
    "pr_controlled": {
        "mode": "transient",
        "features": [
            "pipe_storage",
            "vessel_storage",
            "axial_momentum",
            "pr_thermodynamics",
            "component_conservation",
            "energy_conservation",
            "actuator_dynamics",
            "pi_control",
        ],
        "limitations": [
            "Declared PR plant/control domain; not general experimental qualification",
            "Common SDIRK2 has additional explicit restrictions and an audited no-wall/no-fallback subset",
            "CPU/CUDA selection does not certify equal performance or hardware qualification",
        ],
        "audit": "independent supported plant/control equations and balances; no general global TPD/field qualification",
    },
}

FEATURES = frozenset(f for b in BACKENDS.values() for f in b["features"]) | {
    "vfp",
    "multisegment_wells",
    "group_controls",
    "phase_slip",
    "slugging",
    "acoustic_waves",
}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(raw):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(raw, object_pairs_hook=_object, parse_constant=invalid)


def read_json(path):
    return _decode_json(Path(path).read_bytes())


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fields(value, allowed, required, label):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(
            f"Invalid {label}: expected fields {sorted(allowed)}, required {sorted(required)}"
        )


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _path(root, value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty file path")
    path = (root / value).resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


@dataclass(frozen=True)
class EngineeringCase:
    project_name: str
    name: str
    mode: str
    backend: str
    request_json: str
    options_json: str
    required_features: tuple
    sources: tuple

    @property
    def fingerprint(self):
        return hashlib.sha256(
            _json(
                {
                    "mode": self.mode,
                    "backend": self.backend,
                    "request": json.loads(self.request_json),
                    "options": json.loads(self.options_json),
                }
            ).encode()
        ).hexdigest()

    def request(self):
        return json.loads(self.request_json)

    def options(self):
        return json.loads(self.options_json)

    def verify_sources(self):
        for name, digest in self.sources:
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Project input changed after planning: {name}")

    def describe(self):
        backend = BACKENDS[self.backend]
        request = self.request()
        if self.mode == "forecast":
            horizon = {"value": self.options()["duration_days"], "unit": "day"}
        else:
            horizon = {"value": sum(c["duration_s"] for c in request["schedule"]), "unit": "s"}
        return {
            "schema": "oil_engineering_case_v1",
            "project": self.project_name,
            "case": self.name,
            "mode": self.mode,
            "purpose": MODES[self.mode]["purpose"],
            "backend": self.backend,
            "backend_capabilities": backend["features"],
            "required_features": list(self.required_features),
            "omitted_physics": MODES[self.mode]["omitted"],
            "limitations": backend["limitations"],
            "audit_scope": backend["audit"],
            "simulation_horizon": horizon,
            "backend_options": self.options(),
            "fingerprint": self.fingerprint,
            "input_sha256": dict(self.sources),
            "initialization": "backend input; no automatic cross-mode state conversion",
            "validation": "configuration_only; no solve or physical qualification performed",
        }


def load_case(project_path, case_name):
    snapshots = {}

    def read_input(path):
        raw = path.read_bytes()
        snapshots[str(path)] = hashlib.sha256(raw).hexdigest()
        return _decode_json(raw)

    project_path = Path(project_path).resolve()
    project = read_input(project_path)
    _fields(
        project,
        {"schema", "name", "root", "cases"},
        {"schema", "name", "cases"},
        "engineering project",
    )
    if project["schema"] != "oil_engineering_project_v1":
        raise ValueError("Unknown engineering project schema")
    if not isinstance(project["name"], str) or not project["name"].strip():
        raise ValueError("Project name must be nonempty")
    if not isinstance(project.get("root", "."), str):
        raise TypeError("Project root must be a path string")
    root = (project_path.parent / project.get("root", ".")).resolve()
    if not isinstance(project["cases"], dict) or case_name not in project["cases"]:
        raise ValueError(f"Unknown engineering case {case_name!r}")
    case = project["cases"][case_name]
    _fields(
        case,
        {"mode", "backend", "input", "options", "required_features"},
        {"mode", "backend", "input"},
        "engineering case",
    )
    mode, backend = case["mode"], case["backend"]
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"Unknown engineering mode; choose {', '.join(MODES)}")
    if not isinstance(backend, str) or backend not in BACKENDS:
        raise ValueError(f"Unknown engineering backend; choose {', '.join(BACKENDS)}")
    if BACKENDS[backend]["mode"] != mode:
        raise ValueError(
            f"Backend {backend} models {BACKENDS[backend]['mode']}, not requested mode {mode}; physics is never silently downgraded"
        )
    requirements = case.get("required_features", [])
    if (
        not isinstance(requirements, list)
        or any(not isinstance(x, str) or x not in FEATURES for x in requirements)
        or len(set(requirements)) != len(requirements)
    ):
        raise ValueError("Required features must be unique supported feature names")
    missing = set(requirements) - set(BACKENDS[backend]["features"])
    if missing:
        raise ValueError(
            f"Backend {backend} cannot provide required physics: {', '.join(sorted(missing))}"
        )
    input_path = _path(root, case["input"], "Backend input")
    request = read_input(input_path)
    options = dict(case.get("options", {})) if isinstance(case.get("options", {}), dict) else None
    if options is None:
        raise ValueError("Backend options must be an object")
    if mode == "forecast":
        _fields(
            options,
            {"duration_days", "report_days", "events_file", "outlet_pressure_pa"},
            {"duration_days", "report_days"},
            "forecast options",
        )
        for field in ("duration_days", "report_days"):
            options[field] = _positive(options[field], field)
        if not isinstance(request, dict) or not isinstance(request.get("simulation"), dict):
            raise ValueError("Forecast backend needs an FIM configuration")
        simulation = request["simulation"]
        if (
            simulation.get("fim_model") != "blackoil"
            or simulation.get("solver_type") != "fully_implicit"
            or not {"physical_surface_network", "coupled_facility"} <= set(simulation)
        ):
            raise ValueError(
                "Forecast backend requires black-oil FIM, physical gathering and coupled_facility"
            )
        pvt = _path(root, request["fluid"]["pvt_path"], "PVT input")
        snapshots[str(pvt)] = hashlib.sha256(pvt.read_bytes()).hexdigest()
        request["fluid"]["pvt_path"] = str(pvt)
        if "events_file" in options:
            events = _path(root, options.pop("events_file"), "Forecast events")
            options["events"] = read_input(events)
        else:
            options["events"] = []
        if not isinstance(options["events"], list):
            raise ValueError("Forecast events must be a list")
        last = -1.0
        for event in options["events"]:
            _fields(event, {"time_days", "controls"}, {"time_days", "controls"}, "forecast event")
            time = event["time_days"]
            if (
                isinstance(time, bool)
                or not isinstance(time, (int, float))
                or not math.isfinite(time)
                or not last < time < options["duration_days"]
                or time < 0
            ):
                raise ValueError("Forecast events must be ordered within [0, duration_days)")
            if not isinstance(event["controls"], dict):
                raise TypeError("Forecast event controls must be an object")
            last = time
        options["outlet_pressure_pa"] = _positive(
            options.get(
                "outlet_pressure_pa",
                simulation["coupled_facility"]["facility"]["separator_stages"][0]["pressure_pa"],
            ),
            "outlet_pressure_pa",
        )
    else:
        _fields(
            request,
            {"model", "initial", "schedule"},
            {"model", "initial", "schedule"},
            "surface request",
        )
        if not isinstance(request["schedule"], list) or not request["schedule"]:
            raise ValueError("Surface schedule must be nonempty")
        for command in request["schedule"]:
            if not isinstance(command, dict) or "duration_s" not in command:
                raise ValueError("Every surface command requires duration_s")
            _positive(command["duration_s"], "duration_s")
        model = request["model"]
        if mode == "operations":
            if options:
                raise ValueError("Linepack options belong in its existing model/integration input")
            if not isinstance(model, dict) or model.get("model") != "quasisteady_pr_linepack_v1":
                raise ValueError("Operations backend requires a finite PR linepack model")
        else:
            _fields(options, {"integrator"}, set(), "transient options")
            options.setdefault("integrator", "sampled")
            if options["integrator"] not in ("sampled", "sdirk2-common"):
                raise ValueError("Transient integrator must be sampled or sdirk2-common")
            if (
                not isinstance(model, dict)
                or model.get("model") not in ("pr_controlled_surface_v1", "pr_subcycled_surface_v1")
                or model.get("plant", {}).get("model") != "pr_gravity_transport_graph_v1"
            ):
                raise ValueError(
                    "Transient backend requires an explicit gravitational controlled PR plant"
                )
            nodes = model["plant"]["transport"]["nodes"]
            if "axial_momentum" in requirements and not any(
                n.get("kind") == "pipe_cell" for n in nodes
            ):
                raise ValueError("Requested axial momentum but model contains no pipe cells")
            if "pi_control" in requirements and not model.get("controllers"):
                raise ValueError("Requested PI control but model contains no controllers")
            if "actuator_dynamics" in requirements and not model.get("valves"):
                raise ValueError("Requested actuator dynamics but model contains no valves")
    sources = tuple(snapshots.items())
    return EngineeringCase(
        project["name"],
        case_name,
        mode,
        backend,
        _json(request),
        _json(options),
        tuple(requirements),
        sources,
    )
