"""DRSDT и VAPPARS взаимоисключающие: действует ключевое слово, заданное последним (как Eclipse)."""
import contextlib
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from simulator.eclipse_deck import EclipseCase  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")
DRSDT = "DRSDT\n 0 /\n"
VAPPARS = "VAPPARS\n 0.5 0 /\n"


def _case(tmp_path, block):
    text = open(SPE1).read()
    assert DRSDT in text
    p = tmp_path / "SPE1CASE1.DATA"
    p.write_text(text.replace(DRSDT, block))
    with contextlib.redirect_stdout(io.StringIO()):
        return EclipseCase(str(p))


@pytest.mark.parametrize("block, drsdt, vappars", [
    (DRSDT, True, False),
    (DRSDT + VAPPARS, False, True),
    (VAPPARS + DRSDT, True, False),
])
def test_last_keyword_wins(tmp_path, block, drsdt, vappars):
    case = _case(tmp_path, block)
    assert (case.drsdt is not None) == drsdt
    assert (case.vappars is not None) == vappars
