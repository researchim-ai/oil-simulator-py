"""Isolated old scalar vs device-batched BHP seed; not full forecast timing."""
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from simulator.blackoil_fim import I_P
from simulator.blackoil_surface_coupling import SurfaceCoupledBlackOilWells


def scalar(obj, v, previous):
    out = obj.p_bh.clone()
    for i in range(obj.n_w):
        upper = float((v[I_P][obj.conn_cell[obj.conn_well == i]]
                       - obj.conn_head[obj.conn_well == i]).min())
        prior = float(previous[i])
        out[i] = prior if 0 < prior < upper else .95*upper
    return out


def main():
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Explicit GPU selection required')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.12)
    dev = torch.device('cuda')
    n = 64
    obj = SimpleNamespace(n_w=n, p_bh=torch.zeros(n, dtype=torch.float64, device=dev),
                          conn_cell=torch.arange(n*2, device=dev),
                          conn_well=torch.arange(n, device=dev).repeat_interleave(2),
                          conn_head=torch.linspace(-1e5, 1e5, n*2, dtype=torch.float64, device=dev))
    v = torch.zeros((20, n*2), dtype=torch.float64, device=dev)
    v[I_P] = torch.linspace(1e6, 3e6, n*2, dtype=torch.float64, device=dev)
    previous = torch.linspace(0, 4e6, n, dtype=torch.float64, device=dev)
    def batch():
        out = obj.p_bh.clone()
        out.copy_(SurfaceCoupledBlackOilWells._previous_surface_bhp(obj, v, previous))
        return out
    methods = [lambda: scalar(obj, v, previous), batch]
    torch.testing.assert_close(methods[0](), methods[1](), rtol=0, atol=0)
    for f in methods:
        for _ in range(3):
            f()
    rows = []
    for repeat in range(6):
        for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                result = methods[index]()
            torch.cuda.synchronize()
            rows.append({'repeat':repeat, 'batch':bool(index), 'seconds_per_seed':(time.perf_counter()-start)/10})
            torch.testing.assert_close(result, methods[1-index](), rtol=0, atol=0)
    print(json.dumps({'exact':True, 'wells':n, 'connections':n*2,
                      'device':torch.cuda.get_device_name(), 'rows':rows,
                      'median_seconds':{str(b):statistics.median(r['seconds_per_seed'] for r in rows if r['batch']==b) for b in [False,True]},
                      'scope':__doc__},indent=2))


if __name__ == '__main__':
    main()
