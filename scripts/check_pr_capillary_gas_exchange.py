"""Audit a stored local source trajectory with independent scalar PR and NumPy."""

import argparse
import hashlib
import json
from pathlib import Path

from reference_pr_capillary_gas_exchange import audit as source_audit
from run_inertial_capillary_film import ROOT

from surface.pr_capillary_gas_exchange import PRCapillaryGasExchange
from surface.pr_interphase_relaxation import PRRelaxationPhase
from surface.pr_isothermal_capillary_film import PRIsothermalCapillaryState


def audit(config, report):
    if config != report["request"]:
        raise ValueError("Gas/film source request changed")
    for name, sha in report["source_sha256"].items():
        path = (ROOT / name).resolve()
        path.relative_to(ROOT)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Gas/film source code changed")
    source = PRCapillaryGasExchange.from_dict(config["source"])
    old = PRIsothermalCapillaryState.from_dict(config["initial_film"])
    end = PRIsothermalCapillaryState.from_dict(report["final_film"])
    gas = tuple(PRRelaxationPhase.from_dict(p) for p in config["initial_gas"])
    new_gas = tuple(PRRelaxationPhase.from_dict(p) for p in report["final_gas"])
    return source_audit(
        source,
        old,
        gas,
        end,
        new_gas,
        config["dt_s"],
        config["conductance_kg_s"],
        report["source_report"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    result = audit(json.loads(args.config.read_text()), json.loads(args.report.read_text()))
    print(json.dumps(result, allow_nan=False))
