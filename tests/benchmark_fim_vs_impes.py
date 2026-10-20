import time
import torch
import json
import os
import sys
import gc
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

def run_benchmark(config_path, steps=10, name="Method"):
    # Очистка памяти перед запуском
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print(f"Запуск бенчмарка: {name}")
    print(f"Конфиг: {config_path}")
    print(f"{'='*50}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Ограничиваем вывод для чистоты
    config['output_interval'] = steps + 10 
    
    # Принудительно ставим устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"Memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
        print(f"Memory reserved:  {torch.cuda.memory_reserved(device)/1024**3:.2f} GB")
    
    # Инициализация
    start_init = time.time()
    
    # Создаем объекты
    with torch.no_grad():
        reservoir = Reservoir(config['reservoir'], device)
        well_manager = WellManager(config['wells'], reservoir)
        fluid = Fluid(config['fluid'], reservoir, device)
        sim_params = config.get('simulation', {})
        
        sim = Simulator(reservoir, fluid, well_manager, sim_params, device)
    
    init_time = time.time() - start_init
    print(f"Время инициализации: {init_time:.2f} сек")
    
    # Получаем начальный шаг из параметров
    dt_days = sim_params.get('time_step_days', 1.0)
    current_dt = dt_days * 86400.0
    
    # Прогрев (1 шаг)
    print("Прогрев (1 шаг)...")
    with torch.no_grad():
        sim.run_step(current_dt)
    
    # Замер
    print(f"Замер скорости на {steps} шагах...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_bench = time.time()
    
    total_sim_days = 0
    for i in range(steps):
        step_start = time.time()
        dt_days_val = current_dt / 86400.0
        
        with torch.no_grad():
            success = sim.run_step(current_dt)
            
        step_time = time.time() - step_start
        
        status = "OK" if success else "FAIL"
        print(f"Step {i+1}/{steps}: dt={dt_days_val:.4f} days, Time={step_time:.2f}s, Status={status}")
        
        if success:
            total_sim_days += dt_days_val
            
            # Логика увеличения шага для FIM (упрощенная)
            if name == "FIM":
                dt_increase = sim_params.get('dt_increase_factor', 1.5)
                max_dt = sim_params.get('max_time_step_days', 10.0) * 86400.0
                current_dt = min(current_dt * dt_increase, max_dt)
        else:
            print("Шаг провален, уменьшаем шаг...")
            dt_reduction = sim_params.get('dt_reduction_factor', 2.0)
            current_dt /= dt_reduction
            
    torch.cuda.synchronize() if device.type == 'cuda' else None
    total_time = time.time() - start_bench
    
    # Очистка после теста
    del sim
    del reservoir
    del fluid
    del well_manager
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{'-'*20} Результаты: {name} {'-'*20}")
    print(f"Всего времени (wall):   {total_time:.2f} сек")
    print(f"Всего дней (sim):       {total_sim_days:.4f} дней")
    if total_time > 0:
        print(f"Скорость:               {total_sim_days / total_time * 3600:.2f} сим.дней/час")
        print(f"Среднее время на шаг:   {total_time / steps:.2f} сек/шаг")
    
    return {
        "name": name,
        "wall_time": total_time,
        "sim_days": total_sim_days,
        "speed_days_per_hour": (total_sim_days / total_time * 3600) if total_time > 0 else 0
    }

def main():
    # 1. IMPES (Skipped due to OOM)
    impes_config = "configs/mega_3phase_million.json"
    # res_impes = run_benchmark(impes_config, steps=5, name="IMPES")
    print("IMPES skipped due to OOM issue on large grid.")
    res_impes = {'speed_days_per_hour': 0.0, 'wall_time': 0.0, 'sim_days': 0.0}
    
    # 2. FIM
    fim_config = "configs/mega_3phase_million_fim.json"
    # Увеличим кол-во шагов для FIM чтобы увидеть разгон dt
    res_fim = run_benchmark(fim_config, steps=20, name="FIM")
    
    print("\n\n")
    print(f"{'='*60}")
    print(f"{'ИТОГОВОЕ СРАВНЕНИЕ':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} | {'IMPES':<15} | {'FIM':<15}")
    print(f"{'-'*60}")
    print(f"{'Sim Days / Hour':<20} | {res_impes['speed_days_per_hour']:<15.2f} | {res_fim['speed_days_per_hour']:<15.2f}")
    print(f"{'Avg Step Time (s)':<20} | {res_impes['wall_time']/(5 if res_impes['wall_time']>0 else 1):<15.2f} | {res_fim['wall_time']/20:<15.2f}")
    print(f"{'Total Sim Days':<20} | {res_impes['sim_days']:<15.4f} | {res_fim['sim_days']:<15.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
