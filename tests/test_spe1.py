import os
import sys

import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from compare_eclipse import VECTORS, compare  # noqa: E402
from simulator.ecl_io import EclSummary  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

DATA = os.path.join(ROOT, "tests", "data", "spe1")


@pytest.mark.slow
def test_spe1_matches_eclipse_and_opm():
    refs = {"eclipse": EclSummary(os.path.join(DATA, "eclipse", "SPE1CASE1")),
            "opm": EclSummary(os.path.join(DATA, "opm", "SPE1CASE1"))}
    cells = sorted({c for _, _, c in VECTORS if c is not None})
    summary, _ = run_deck(os.path.join(DATA, "SPE1CASE1.DATA"), device=torch.device("cpu"), cells=cells)
    for row in compare(summary, refs):
        assert row["eclipse"] < 0.01, row
        assert row["opm"] < 0.01, row
