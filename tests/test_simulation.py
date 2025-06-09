import torch
import numpy as np
import json
import os
import sys
import pytest

# Add src to path to import simulator components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import Well, WellManager
from simulator.simulation import Simulator

def run_simulation_for_test(config_path):
    """Runs the simulation with a given config and returns final pressure and saturation."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    res_params = config['reservoir']
    sim_params = config['simulation']
    fluid_params = config['fluid']
    well_params = config['wells']

    device = torch.device("cpu") # Force CPU for testing consistency

    reservoir = Reservoir(
        dimensions=tuple(res_params['dimensions']),
        grid_size=tuple(res_params['grid_size']),
        porosity=res_params['porosity'],
        permeability=res_params['permeability'],
        device=device
    )

    well_manager = WellManager()
    for well_info in well_params:
        well_manager.add_well(
            Well(
                name=well_info['name'],
                well_type=well_info['type'],
                coordinates=tuple(well_info['coordinates']),
                reservoir_dimensions=tuple(res_params['dimensions']),
                rate=well_info['rate']
            )
        )

    fluid = Fluid(
        p_init=fluid_params['pressure'],
        s_w_init=fluid_params['s_w'],
        mu_oil=fluid_params['mu_oil'],
        mu_water=fluid_params['mu_water'],
        rho_oil=fluid_params['rho_oil'],
        rho_water=fluid_params['rho_water'],
        c_oil=fluid_params['c_oil'],
        c_water=fluid_params['c_water'],
        c_rock=fluid_params['c_rock'],
        sw_cr=fluid_params['sw_cr'],
        so_r=fluid_params['so_r'],
        nw=fluid_params['nw'],
        no=fluid_params['no'],
        reservoir=reservoir,
        device=device
    )

    sim = Simulator(reservoir, fluid, well_manager)

    total_time_days = sim_params['total_time_days']
    time_step_days = sim_params['time_step_days']
    time_step_sec = time_step_days * 86400
    num_steps = int(total_time_days / time_step_days)

    for i in range(num_steps):
        sim.run_step(dt=time_step_sec)

    return fluid.pressure.cpu().numpy(), fluid.s_w.cpu().numpy()


def test_simulation_results(pytestconfig):
    """
    Tests if the simulation output matches the golden reference files.
    If golden files don't exist, it creates them.
    """
    config_path = "configs/test_config.json"
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    
    golden_pressure_path = os.path.join(test_data_dir, 'golden_pressure.npy')
    golden_saturation_path = os.path.join(test_data_dir, 'golden_saturation.npy')

    # Run the simulation
    pressure, saturation = run_simulation_for_test(config_path)

    if not os.path.exists(golden_pressure_path) or not os.path.exists(golden_saturation_path):
        print("Golden files not found. Creating them now.")
        np.save(golden_pressure_path, pressure)
        np.save(golden_saturation_path, saturation)
        pytest.fail(
            "Golden files were created. Please review them and run the tests again. "
            "If they are correct, commit them to version control."
        )
    else:
        golden_pressure = np.load(golden_pressure_path)
        golden_saturation = np.load(golden_saturation_path)

        assert np.allclose(pressure, golden_pressure, rtol=1e-5, atol=1e-5), "Pressure does not match golden file."
        assert np.allclose(saturation, golden_saturation, rtol=1e-5, atol=1e-5), "Saturation does not match golden file."

def test_simulation_results_3d_gravity(pytestconfig):
    """
    Tests if the 3D simulation with gravity matches the golden reference files.
    """
    config_path = "configs/test_config_3d.json"
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    
    golden_pressure_path = os.path.join(test_data_dir, 'golden_pressure_3d.npy')
    golden_saturation_path = os.path.join(test_data_dir, 'golden_saturation_3d.npy')

    # Run the simulation
    pressure, saturation = run_simulation_for_test(config_path)

    if not os.path.exists(golden_pressure_path) or not os.path.exists(golden_saturation_path):
        print("3D Golden files not found. Creating them now.")
        np.save(golden_pressure_path, pressure)
        np.save(golden_saturation_path, saturation)
        pytest.fail(
            "3D Golden files were created. Please review them and run the tests again."
        )
    else:
        golden_pressure = np.load(golden_pressure_path)
        golden_saturation = np.load(golden_saturation_path)

        assert np.allclose(pressure, golden_pressure, rtol=1e-5, atol=1e-5), "3D Pressure does not match golden file."
        assert np.allclose(saturation, golden_saturation, rtol=1e-5, atol=1e-5), "3D Saturation does not match golden file."

def test_adaptive_timestep_triggered(capsys):
    """
    Tests that the adaptive timestep mechanism is triggered when the solver fails to converge.
    This is forced by providing a very low max_iter count to the solver.
    """
    config_path = "configs/test_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Запускаем симуляцию с очень малым числом итераций, чтобы вызвать несходимость
    try:
        run_simulation_with_config_obj(config, solver_max_iter=1)
    except RuntimeError as e:
        print(f"Симуляция завершилась с ошибкой (это может быть нормально для теста): {e}")

    # Проверяем, что в логах была попытка уменьшения шага
    captured = capsys.readouterr()
    assert "Решатель давления не сошелся" in captured.out
    assert "Уменьшаем шаг времени" in captured.out

def run_simulation_with_config_obj(config, solver_tol=1e-6, solver_max_iter=500):
    """Helper function to run simulation from a config dictionary instead of a file."""
    res_params = config['reservoir']
    sim_params = config['simulation']
    fluid_params = config['fluid']
    well_params = config['wells']

    device = torch.device("cpu")

    reservoir = Reservoir(
        dimensions=tuple(res_params['dimensions']),
        grid_size=tuple(res_params['grid_size']),
        porosity=res_params['porosity'],
        permeability=res_params['permeability'],
        device=device
    )

    well_manager = WellManager()
    for well_info in well_params:
        well_manager.add_well(
            Well(
                name=well_info['name'],
                well_type=well_info['type'],
                coordinates=tuple(well_info['coordinates']),
                reservoir_dimensions=tuple(res_params['dimensions']),
                rate=well_info['rate']
            )
        )

    fluid = Fluid(
        p_init=fluid_params['pressure'],
        s_w_init=fluid_params['s_w'],
        mu_oil=fluid_params['mu_oil'],
        mu_water=fluid_params['mu_water'],
        rho_oil=fluid_params['rho_oil'],
        rho_water=fluid_params['rho_water'],
        c_oil=fluid_params['c_oil'],
        c_water=fluid_params['c_water'],
        c_rock=fluid_params['c_rock'],
        sw_cr=fluid_params['sw_cr'],
        so_r=fluid_params['so_r'],
        nw=fluid_params['nw'],
        no=fluid_params['no'],
        reservoir=reservoir,
        device=device
    )

    sim = Simulator(reservoir, fluid, well_manager)

    total_time_days = sim_params['total_time_days']
    time_step_days = sim_params['time_step_days']
    time_step_sec = time_step_days * 86400
    num_steps = int(total_time_days / time_step_days)
    
    # In this test, we might not have any steps if total_time < time_step
    # Let's run at least one step.
    if num_steps == 0:
        num_steps = 1

    for _ in range(num_steps):
        sim.run_step(dt=time_step_sec, solver_tol=solver_tol, solver_max_iter=solver_max_iter)
        
    return fluid.pressure.cpu().numpy(), fluid.s_w.cpu().numpy() 