import os
import sys
import numpy as np
import pytest
import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.deck import load_pvt_from_deck, load_relperm_tables_from_deck
from simulator.fluid import Fluid
from simulator.pvt import PVTTable


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
    assert "extended" in data
    oil_ext = data["extended"]["oil"]
    assert oil_ext["rs_values"][0] == pytest.approx(0.0)
    assert "tables" in oil_ext and pytest.approx(0.01) == oil_ext["tables"][0.0]["pressure_MPa"][0]


def test_load_pvt_extended_sections(tmp_path):
    deck_content = """PVTO
    0.0  10.0  1.05  1.20  0.00 /
    0.0  20.0  1.02  1.40  0.00 /
    80.0 10.0  1.10  2.00  0.05 /
    80.0 20.0  1.06  2.20  0.05 /
/
PVDO
    10.0  1.04  1.10 /
    20.0  1.01  1.30 /
/
PVTW
    10.0  1.00  0.50  1.0e-05  5.0e-06 /
    20.0  0.98  0.60  1.0e-05  5.0e-06 /
/
PVDG
    10.0  0.005  0.020 /
    20.0  0.004  0.025 /
/
PLYROCK
    0.0  1.0 /
    0.5  0.8 /
/
PLYVISC
    0.0  1.0 /
    0.5  2.0 /
/
PLYADS
    0.0  0.0 /
    0.5  0.1 /
/
SURFROCK
    0.0  1.0 /
    0.5  0.9 /
/
SURFADS
    0.0  0.0 /
    0.4  0.05 /
/
THERM
    50.0  10.0  1.0 /
/
DENSITY
    850.0  1000.0  1.1 /
/
"""
    deck_file = tmp_path / "extended.inc"
    deck_file.write_text(deck_content)
    data = load_pvt_from_deck(str(deck_file))
    ext = data["extended"]
    oil_ext = ext["oil"]
    assert pytest.approx(0.0) == oil_ext["rs_values"][0]
    assert pytest.approx(80.0) == oil_ext["rs_values"][1]
    rs_table = oil_ext["tables"][80.0]
    assert pytest.approx(1.10) == rs_table["Bo"][0]
    assert "polymer" in ext and "viscosity" in ext["polymer"]
    assert ext["polymer"]["viscosity"]["multiplier"][-1] == pytest.approx(2.0)
    assert "surfactant" in ext and "adsorption" in ext["surfactant"]
    pvt = PVTTable(data)
    oil_rs = pvt.eval_oil(np.array([0.02]), np.array([80.0]))
    assert oil_rs["Bo"][0] == pytest.approx(1.06)
    assert pvt.get_polymer_viscosity_multiplier(0.5) == pytest.approx(2.0)
    assert pvt.get_polymer_adsorption(0.5) == pytest.approx(0.1)


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

