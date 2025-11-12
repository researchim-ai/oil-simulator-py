import os
import sys
import pytest
import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.deck import load_pvt_from_deck, load_relperm_tables_from_deck
from simulator.fluid import Fluid


class DummyReservoir:
    dimensions = (4, 1, 1)


def test_load_pvt_from_deck(tmp_path):
    deck_path = Path("tests/test_data/simple_blackoil.inc")
    data = load_pvt_from_deck(str(deck_path))
    assert "pressure_MPa" in data
    assert data["pressure_MPa"][0] == pytest.approx(0.01)
    assert data["Bo"][0] == pytest.approx(1.05)
    assert data["mu_o_cP"][-1] == pytest.approx(1.40)
    assert data["Bg"][0] == pytest.approx(0.005)


def test_load_relperm_from_deck():
    deck_path = Path("tests/test_data/simple_blackoil.inc")
    tables = load_relperm_tables_from_deck(str(deck_path))
    assert "swof" in tables and "sgof" in tables
    swof = tables["swof"]
    assert swof["sw"][0] == pytest.approx(0.20)
    assert swof["krw"][-1] == pytest.approx(1.0)
    sgof = tables["sgof"]
    assert sgof["sg"][2] == pytest.approx(0.20)
    assert sgof["krg"][-1] == pytest.approx(1.0)


def test_fluid_table_relperm(monkeypatch):
    deck_path = "tests/test_data/simple_blackoil.inc"
    cfg = {
        "pressure": 10.0,
        "s_w": 0.3,
        "s_g": 0.1,
        "pvt_path": deck_path,
        "relative_permeability": {
            "model": "table"
        }
    }
    fluid = Fluid(cfg, DummyReservoir(), device=torch.device("cpu"))
    assert fluid.relperm_model == "table"
    sw = torch.tensor([0.3, 0.5])
    sg = torch.tensor([0.1, 0.2])
    krw = fluid.calc_water_kr(sw)
    assert torch.all(krw > 0)
    pcw = fluid.get_capillary_pressure(sw)
    assert torch.all(pcw > 0)
    krg = fluid.calc_gas_kr(sg)
    assert torch.all(krg > 0)

