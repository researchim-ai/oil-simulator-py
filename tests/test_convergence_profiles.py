"""Strict deck settings cannot silently accept the relaxed local residual."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.blackoil_fim import BlackOilFIMSolver, I_B
from simulator.eclipse_deck import EclipseCase

SPE1 = Path(__file__).parent / "data/spe1/SPE1CASE1.DATA"


@pytest.mark.parametrize("profile", ["strict", "relaxed"])
def test_deck_convergence_profile(profile):
    _, _, _, settings = EclipseCase(str(SPE1)).build(
        device=torch.device("cpu"), sim_overrides={"convergence_profile": profile})
    assert settings["newton_tol_cnv"] == (1e-3 if profile == "strict" else 1e-2)
    assert settings["newton_relaxed_max_pv_fraction"] == (0 if profile == "strict" else 0.03)


def test_default_convergence_is_strict():
    _, _, _, settings = EclipseCase(str(SPE1)).build(device=torch.device("cpu"))
    assert settings["newton_tol_cnv"] == 1e-3
    assert settings["newton_relaxed_max_pv_fraction"] == 0


def test_local_spike_rejected_by_strict_profile_even_with_zero_global_balance():
    fake = SimpleNamespace(tol_cnv=1e-3, tol_mb=1e-7, max_strict_iter=0,
                           tol_cnv_relaxed=1.0, tol_mb_relaxed=1e-6,
                           relaxed_max_pv_fraction=0.0, rho_sc=torch.ones(3))
    pv = torch.ones(100, dtype=torch.float64)
    residual = torch.zeros(100, 3, dtype=torch.float64)
    residual[0, 0], residual[1, 0] = 0.02, -0.02
    v = torch.ones(I_B + 3, 100, dtype=torch.float64)
    ok, detail = BlackOilFIMSolver.convergence(fake, residual.flatten(), 1, pv, None, None, v)
    assert not ok and not detail["unrelaxed_converged"] and not detail["cnv_relaxed"]
    assert detail["tol_cnv_used"] == 1e-3
    fake.relaxed_max_pv_fraction = 0.03
    ok, detail = BlackOilFIMSolver.convergence(fake, residual.flatten(), 1, pv, None, None, v)
    assert ok and detail["cnv_relaxed"] and not detail["unrelaxed_converged"]
