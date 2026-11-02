"""Independent frozen CoolProp EOS/caloric references and flash recovery."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pr_reference_support import CoolPropConventionPR, cases, fluid_for, validate_one

from surface import (
    HydraulicConvergenceError,
    PengRobinsonFluid,
    PengRobinsonMixture,
    single_phase_properties,
)

DATA = json.loads((ROOT / "tests/data/surface/pr_caloric_coolprop.json").read_text())
CASES = list(cases(DATA))


@pytest.mark.parametrize(
    "kind,case,matched,method",
    CASES,
    ids=[f"{k}-{c['id']}-{m}-{method}" for k, c, m, method in CASES],
)
def test_external_reference(kind, case, matched, method):
    result = validate_one(DATA, kind, case, matched, method)
    assert result["passed"], result


def test_reference_metadata_and_oracle_material_quality_are_explicit():
    assert DATA["generator_version"] == "7.2.0"
    assert (
        DATA["generator_sha256"]
        == hashlib.sha256(
            (ROOT / "scripts/generate_pr_caloric_reference.py").read_bytes()
        ).hexdigest()
    )
    for c in DATA["vle_cases"]:
        beta = c["vapor_molar_fraction"]
        reconstructed = (1 - beta) * np.array(c["liquid_mole_fractions"]) + beta * np.array(
            c["vapor_mole_fractions"]
        )
        assert reconstructed == pytest.approx(c["mole_fractions"], abs=1e-15)
        assert reconstructed - np.array(c["requested_mole_fractions"]) == pytest.approx(
            c["coolprop_feed_residual"], abs=1e-15
        )
    assert max(abs(v) for c in DATA["vle_cases"] for v in c["coolprop_feed_residual"]) > 1e-6


@pytest.mark.parametrize("matched", [False, True])
def test_ternary_flash_recovers_valid_split_from_failed_half_quality_start(matched):
    case = DATA["vle_cases"][31]
    f = fluid_for(DATA, case, matched)
    # This nominal-feed state formerly raised before a PS temperature root
    # could reach its target. Recovery retains the original fugacity tolerance.
    result = f.flash_pt(5e6, 297.86084733540304, [0.5, 0.25, 0.25])
    assert result["phase_count"] == 2
    assert result["vapor_molar_fraction"] == pytest.approx(
        0.9420900459338785 if matched else 0.9419647512203806, abs=1e-9
    )
    assert result["log_fugacity_error"] < 1e-8
    assert result["component_balance_error"] < 1e-10


def test_exhausted_flash_recovery_does_not_invent_a_single_phase(monkeypatch):
    import surface.peng_robinson_mixture as implementation

    case = DATA["vle_cases"][31]
    f = fluid_for(DATA, case, False)

    def trivial(*args, **kwargs):
        return SimpleNamespace(x=np.r_[np.zeros(3), 0.5])

    monkeypatch.setattr(implementation, "least_squares", trivial)
    with pytest.raises(HydraulicConvergenceError, match="no resolved"):
        f.flash_pt(5e6, 297.86084733540304, [0.5, 0.25, 0.25])


def test_attraction_coefficient_is_shared_by_state_and_caloric_tangent():
    class Adapted(PengRobinsonFluid):
        @property
        def attraction_coefficient(self):
            return super().attraction_coefficient * 1.1

    components = tuple(Adapted(**spec) for spec in DATA["components"][:2])
    f = PengRobinsonMixture(components, ((0, 0), (0, 0)))
    p, t, z = 1e6, 350, [0.5, 0.5]
    q = single_phase_properties(f, p, t, z)
    h = 0.001
    derivative = (
        f.state(p, t + h, z)["molar_enthalpy_j_mol"] - f.state(p, t - h, z)["molar_enthalpy_j_mol"]
    ) / (2 * h)
    assert q["cp_j_mol_k"] == pytest.approx(derivative, rel=2e-8)
    assert CoolPropConventionPR is not PengRobinsonFluid


def test_changed_reference_fails_the_gate_and_cli_returns_nonzero(tmp_path):
    bad = copy.deepcopy(DATA)
    bad["homogeneous_cases"][0]["expected"]["cp_j_mol_k"] *= 1.2
    result = validate_one(bad, "homogeneous", bad["homogeneous_cases"][0], True)
    assert not result["passed"] and not result["metrics"]["cp_j_mol_k"]["passed"]
    ref = tmp_path / "bad.json"
    ref.write_text(json.dumps(bad))
    output = tmp_path / "comparison.json"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_pr_reference.py"),
            "--reference",
            str(ref),
            "--section",
            "homogeneous",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1 and output.exists()
    report = json.loads(output.read_text())
    assert not report["passed"] and report["failures"] > 0
    assert report["section"] == "homogeneous"
