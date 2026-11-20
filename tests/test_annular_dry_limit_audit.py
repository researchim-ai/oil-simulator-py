"""Verify the diagnostic against the actual thin-film constitutive limit."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_annular_dry_limit import dry_offset

from surface.annular_film import interfacial_conductance
from surface.pr_active_pipe_friction import darcy_factor


@pytest.mark.parametrize("speed", [5.0, 9.0])
@pytest.mark.parametrize("roughness", [0.0, 0.0002])
def test_audit_matches_actual_vanishing_film(speed, roughness):
    diameter, rho, length = 0.085, 1.2, 3.0
    alpha = torch.tensor(1e-10, dtype=torch.float64)
    effective, _, _, _, _, _ = interfacial_conductance(
        alpha,
        diameter,
        0.001002,
        1.81e-5,
        rho,
        torch.tensor(speed, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    tau = effective * speed
    wet = 2 * float(tau) / (diameter / 2 * math.sqrt(1 - float(alpha)))
    fd = float(
        darcy_factor(
            torch.tensor(rho * speed * diameter / 1.81e-5, dtype=torch.float64),
            roughness / diameter,
        )
    )
    dry = fd * rho * speed**2 / (2 * diameter)
    assert length * (wet - dry) == pytest.approx(dry_offset(speed, rho, roughness), abs=1e-6)


def test_audit_refuses_transitional_branch():
    with pytest.raises(ValueError, match="turbulent"):
        dry_offset(0.1, 1.2, 0.0)


@pytest.mark.parametrize("reynolds", [0.0, 100.0, 2000.0, 3000.0, 4000.0, 50000.0])
@pytest.mark.parametrize("roughness", [0.0, 0.0002])
def test_consistent_constitutive_limit_including_transition(reynolds, roughness):
    diameter, rho, mu = 0.085, 1.2, 1.81e-5
    speed = reynolds * mu / (rho * diameter)
    alpha = torch.tensor(1e-12, dtype=torch.float64)
    effective, _, _, _, _, _ = interfacial_conductance(
        alpha,
        diameter,
        0.001002,
        mu,
        rho,
        torch.tensor(speed, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        dry_consistent=True,
        roughness_m=roughness,
    )
    gradient = 4 * float(effective) * speed / diameter
    factor = float(darcy_factor(torch.tensor(reynolds, dtype=torch.float64), roughness / diameter))
    assert gradient == pytest.approx(factor * rho * speed**2 / (2 * diameter), rel=1e-9, abs=1e-12)
