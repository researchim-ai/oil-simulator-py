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
        config=res_params,
        device=device
    )

    well_manager = WellManager(well_params, reservoir)

    fluid = Fluid(
        reservoir=reservoir,
        config=fluid_params,
        device=device
    )

    sim = Simulator(reservoir, fluid, well_manager, sim_params)

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
    config_path = "configs/fully_implicit_2d.json" # Используем неявный решатель
    with open(config_path, 'r') as f:
        config = json.load(f)
        
        # Уменьшаем макс. кол-во итераций, чтобы гарантированно вызвать несходимость
        config['simulation']['newton_max_iter'] = 1

        # Запускаем симуляцию с очень малым числом итераций, чтобы вызвать несходимость
        try:
            run_simulation_with_config_obj(config)
        except RuntimeError as e:
            # Ожидаем, что симуляция упадет, но после попыток уменьшить шаг
            print(f"Симуляция предсказуемо упала: {e}")
        
        captured = capsys.readouterr()
        # Проверяем, что в выводе есть сообщение о сокращении шага
        assert "Уменьшаем шаг времени" in captured.out

def run_simulation_with_config_obj(config):
    """Helper function to run simulation from a config dictionary instead of a file."""
    res_params = config['reservoir']
    sim_params = config['simulation']
    fluid_params = config['fluid']
    well_params = config['wells']

    device = torch.device("cpu")

    reservoir = Reservoir(
        config=res_params,
        device=device
    )

    well_manager = WellManager(well_params, reservoir)

    fluid = Fluid(
        reservoir=reservoir,
        config=fluid_params,
        device=device
    )

    sim = Simulator(reservoir, fluid, well_manager, sim_params)

    total_time_days = sim_params['total_time_days']
    time_step_days = sim_params['time_step_days']
    time_step_sec = time_step_days * 86400
    num_steps = int(total_time_days / time_step_days)
    
    # Let's run at least one step.
    if num_steps == 0:
        num_steps = 1

    for _ in range(num_steps):
        sim.run_step(dt=time_step_sec)
        
    return fluid.pressure.cpu().numpy(), fluid.s_w.cpu().numpy() 