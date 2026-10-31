"""Analytic dry-gas equipment and first-law balances."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import EquipmentOperatingError
from surface.gas import (
    GasCompressor,
    GasCooler,
    GasLetdown,
    GasState,
    GasTrain,
    IdealGas,
    mix_gas_streams,
)

GAS = IdealGas(287.0, 1.4)


def test_compressor_against_isentropic_temperature_relation():
    inlet = GasState(2.0, 1e5, 300)
    compressor = GasCompressor("c", 2e5, isentropic_efficiency=0.8, mechanical_efficiency=0.9)
    r = compressor.run(inlet, GAS)
    temperature = 300 * (1 + (2 ** (2 / 7) - 1) / 0.8)
    assert r.outlet.temperature_k == pytest.approx(temperature)
    assert r.gas_power_w == pytest.approx(2 * 1004.5 * (temperature - 300))
    assert r.shaft_power_w == pytest.approx(r.gas_power_w / 0.9)
    assert r.drive_power_w == r.shaft_power_w
    assert abs(r.energy_residual_w) < 1e-8
    assert r.outlet.mass_rate_kg_s == inlet.mass_rate_kg_s


def test_compression_intercooling_train_energy():
    inlet = GasState(1, 1e5, 300)
    train = GasTrain(
        [
            GasCompressor("c1", 2e5),
            GasCooler("cool1", 300, 0.02),
            GasCompressor("c2", 4e5),
            GasCooler("cool2", 300),
        ]
    )
    r = train.run(inlet, GAS)
    assert r["outlet"]["temperature_k"] == 300
    assert r["outlet"]["pressure_pa"] == 4e5
    assert r["gas_power_w"] == pytest.approx(r["rejected_heat_w"])
    assert abs(r["energy_residual_w"]) < 1e-8
    assert abs(r["drive_energy_residual_w"]) < 1e-8
    assert r["mass_residual_kg_s"] == 0


def test_gas_mixer_conserves_enthalpy_and_forbids_pressure_jump():
    result = mix_gas_streams([GasState(1, 1e5, 300), GasState(3, 1e5, 400)], GAS)
    assert result == GasState(4, 1e5, 375)
    with pytest.raises(EquipmentOperatingError, match="equal port pressures"):
        mix_gas_streams([GasState(1, 1e5, 300), GasState(1, 2e5, 300)], GAS)
    assert GasLetdown("v", 1e5).run(GasState(1, 2e5, 350), GAS).outlet == GasState(1, 1e5, 350)


@pytest.mark.parametrize(
    "equipment,inlet",
    [
        (GasCompressor("c", 5e5), GasState(1, 1e5, 300)),
        (GasCompressor("c", 2e5, max_drive_power_w=1), GasState(1, 1e5, 300)),
        (GasCompressor("c", 2e5, max_inlet_volume_m3_s=0.001), GasState(1, 1e5, 300)),
        (GasCompressor("c", 2e5, max_discharge_temperature_k=310), GasState(1, 1e5, 300)),
        (GasCooler("cool", 400), GasState(1, 1e5, 300)),
        (GasCooler("cool", 300, max_heat_duty_w=1), GasState(1, 1e5, 400)),
        (GasLetdown("v", 2e5), GasState(1, 1e5, 300)),
    ],
)
def test_equipment_rejects_unavailable_operations(equipment, inlet):
    with pytest.raises(EquipmentOperatingError):
        equipment.run(inlet, GAS)


def test_intercooling_ratio_allocation_accounts_for_cooler_losses():
    from surface.gas import intercooled_train

    inlet = GasState(1, 1e5, 300)
    train = intercooled_train(
        "train",
        inlet,
        GAS,
        1.6e6,
        300,
        max_stage_pressure_ratio=4,
        cooler_pressure_loss_fraction=0.02,
        max_discharge_temperature_k=550,
    )
    r = train.run(inlet, GAS)
    compressors = [e for e in r["equipment"] if e["kind"] == "adiabatic_ideal_gas_compressor"]
    assert len(compressors) == 3  # 4*4 cannot overcome the cooler losses.
    assert r["outlet"]["pressure_pa"] == pytest.approx(1.6e6)
    assert all(e["outlet"]["pressure_pa"] / e["inlet"]["pressure_pa"] <= 4 for e in compressors)
    assert abs(r["energy_residual_w"]) < 1e-7


def test_compressible_pipe_against_independent_momentum_ode():
    import math

    from scipy.integrate import solve_ivp

    from surface.gas import IsothermalGasPipe

    inlet = GasState(0.5, 1e6, 300)
    pipe = IsothermalGasPipe("pipe", 1000, 0.1, 1e-5)
    r = pipe.run(inlet, GAS)
    flux = inlet.mass_rate_kg_s / (math.pi * 0.1**2 / 4)
    a = flux**2 * 287 * 300
    f = r["friction_factor"]
    # Differential momentum, independently integrated rather than root inversion.
    solution = solve_ivp(
        lambda x, p: [-f * a / (2 * 0.1 * p[0] * (1 - a / p[0] ** 2))],
        [0, 1000],
        [1e6],
        rtol=1e-11,
        atol=1e-6,
    )
    assert r["outlet"]["pressure_pa"] == pytest.approx(solution.y[0, -1], rel=1e-10)
    assert r["outlet_mach"] < 1
    assert r["heat_added_w"] > 0
    assert abs(r["momentum_residual_pa2"]) < 1e-2


def test_gas_pipe_zero_flow_and_choking_fail_explicitly():
    from surface.gas import IsothermalGasPipe

    p = IsothermalGasPipe("pipe", 1000, 0.1)
    assert p.run(GasState(0, 1e5, 300), GAS)["outlet"]["pressure_pa"] == 1e5
    with pytest.raises(EquipmentOperatingError, match="choking|subcritical"):
        p.run(GasState(10, 1e5, 300), GAS)


def test_stage_selection_also_respects_temperature_bound():
    from surface.gas import intercooled_train

    inlet = GasState(1, 1e5, 300)
    train = intercooled_train(
        "cooler_train", inlet, GAS, 1.6e6, 300, max_discharge_temperature_k=350
    )
    result = train.run(inlet, GAS)
    compressors = [r for r in result["equipment"] if r["kind"] == "adiabatic_ideal_gas_compressor"]
    assert len(compressors) == 7
    assert max(r["outlet"]["temperature_k"] for r in compressors) <= 350
    with pytest.raises(EquipmentOperatingError, match="temperature"):
        intercooled_train(
            "not_enough", inlet, GAS, 1.6e6, 300, max_discharge_temperature_k=350, max_stages=6
        )


@pytest.mark.parametrize("re", [1000, 1e5])
def test_darcy_friction_against_laminar_and_smooth_colebrook_limits(re):
    import math

    from scipy.optimize import brentq

    from surface.gas import IsothermalGasPipe

    pipe = IsothermalGasPipe("smooth", 100, 0.1)
    mass = re * math.pi * 0.1 * GAS.viscosity_pa_s / 4
    f = pipe.friction_factor(mass, GAS)
    if re < 2000:
        reference = 64 / re
    else:
        reference = brentq(
            lambda v: 1 / math.sqrt(v) + 2 * math.log10(2.51 / (re * math.sqrt(v))), 0.005, 0.1
        )
    assert f == pytest.approx(reference, rel=1e-10)


def test_pipe_detects_downstream_choking_with_subcritical_inlet():
    from surface.gas import IsothermalGasPipe

    with pytest.raises(EquipmentOperatingError, match="isothermal choking"):
        IsothermalGasPipe("long_line", 1000, 0.1).run(GasState(0.5, 1e5, 300), GAS)
