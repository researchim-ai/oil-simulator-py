"""Geometrical levels and conservative implicit level-controlled phase outlets."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_surface_vessel_network import setup

from surface import (
    DynamicVessel,
    DynamicVesselNetwork,
    LevelWithdrawal,
    NetworkTrip,
    SeparatorGeometry,
    VesselState,
    advance_protected_network,
    run_network_schedule,
)


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_cylinder_geometry_against_independent_volume_quadrature(orientation):
    from scipy.integrate import quad

    g = SeparatorGeometry(2, 3, orientation)
    for fraction in (0, 1e-5, 0.1, 0.5, 0.9, 1 - 1e-5, 1):
        h = g.height_m * fraction
        volume = g.liquid_volume_m3(h)
        assert g.level_m(volume) == pytest.approx(h, abs=1e-10)
        expected = (
            math.pi * h
            if orientation == "vertical"
            else 3
            * quad(
                lambda z: 2 * math.sqrt(max(0, 1 - (z - 1) ** 2)), 0, h, epsabs=1e-12, epsrel=1e-12
            )[0]
        )
        assert volume == pytest.approx(expected, abs=1e-9)
    assert g.liquid_volume_m3(g.height_m / 2) == pytest.approx(g.volume_m3 / 2)


def case():
    f, *_ = setup()
    g = SeparatorGeometry(0.5, 0.1 / (math.pi * 0.5**2 / 4))
    v = DynamicVessel("separator", g.volume_m3, 1e5, 5e6)
    net = DynamicVesselNetwork({"separator": v}, geometries={"separator": g})
    s = net.initialize(
        f, {"separator": {"pressure_pa": 3e6, "temperature_k": 220, "mole_fractions": [0.5, 0.5]}}
    )
    q = v.inspect(s["separator"], f)["equilibrium"]
    h = g.from_equilibrium(s["separator"].component_moles, q)
    return f, g, v, net, s, h


def test_implicit_control_uses_new_level_and_accounts_phase_mass_energy():
    f, g, _v, net, s, h = case()
    control = LevelWithdrawal(h * 0.9, 0.1, 0.02, 0.001)
    r = net.advance(s, f, 0.1, withdrawals={"separator": {"phase_0": control}}, initial_dt_s=0.1)
    state = VesselState(**r["states"]["separator"])
    level = r["steps"][0]["vessels"]["separator"]["liquid_level_m"]
    assert level < h
    assert sum(r["steps"][0]["outlet_component_moles"]) == pytest.approx(
        control.rate(level) * 0.1, rel=1e-8
    )
    assert np.array(s["separator"].component_moles) - state.component_moles == pytest.approx(
        r["steps"][0]["outlet_component_moles"], abs=1e-8
    )
    assert s["separator"].internal_energy_j - state.internal_energy_j == pytest.approx(
        r["steps"][0]["outlet_enthalpy_j"], abs=1e-5
    )
    assert 0 < level < g.height_m


def test_geometrical_level_trip_stops_regulated_outlet_and_restart_is_exact():
    f, _g, _v, net, s, h = case()
    control = LevelWithdrawal(h, 0.1, 0.02, 0.01)
    probe = net.advance(s, f, 0.1, withdrawals={"separator": {"phase_0": control}})
    end = probe["steps"][-1]["vessels"]["separator"]["liquid_level_m"]
    assert end < h
    trip = NetworkTrip(
        "low-level",
        "separator",
        "liquid_level_m",
        "below",
        (h + end) / 2,
        1e-9,
        stop_withdrawal_vessels=("separator",),
    )
    r = advance_protected_network(
        net, s, f, 0.2, [trip], withdrawals={"separator": {"phase_0": control}}, initial_dt_s=0.1
    )
    assert len(r["events"]) == 1 and 0 < r["events"][0]["time_s"] < 0.1
    assert sum(r["steps"][-1]["outlet_component_moles"]) == 0
    periods = [
        {
            "duration_s": 0.1,
            "withdrawals": {
                "separator": {
                    "phase_0": {
                        "setpoint_m": h,
                        "gain_mol_s_m": 0.1,
                        "max_molar_rate_mol_s": 0.02,
                        "bias_mol_s": 0.01,
                    }
                }
            },
        },
        {"duration_s": 0.1},
    ]
    full = run_network_schedule(f, net, s, periods, protections=[trip])
    partial = run_network_schedule(f, net, s, periods, protections=[trip], stop_after=1)
    resumed = run_network_schedule(
        f, net, s, periods, protections=[trip], checkpoint=json.loads(json.dumps(partial))
    )
    assert full == resumed


def test_geometrical_domains_and_bad_control_port_fail():
    f, g, v, net, s, h = case()
    with pytest.raises(ValueError, match="volume"):
        DynamicVesselNetwork({"separator": v}, geometries={"separator": SeparatorGeometry(1, 1)})
    with pytest.raises(ValueError):
        g.level_m(-1)
    with pytest.raises(ValueError):
        net.advance(s, f, 0.1, withdrawals={"separator": {"phase_1": LevelWithdrawal(h, 1, 0.01)}})
    with pytest.raises(ValueError, match="geometry"):
        run_network_schedule(
            f,
            net,
            s,
            [
                {
                    "duration_s": 1,
                    "withdrawals": {
                        "separator": {
                            "phase_0": {
                                "setpoint_m": g.height_m + 1,
                                "gain_mol_s_m": 1,
                                "max_molar_rate_mol_s": 0.01,
                            }
                        }
                    },
                }
            ],
            stop_after=0,
        )


def test_control_limits_are_explicit():
    c = LevelWithdrawal(0.5, 2, 1, 0.2)
    assert c.rate(0) == 0 and c.rate(1) == 1 and c.rate(0.5) == 0.2
