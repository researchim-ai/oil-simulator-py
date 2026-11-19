"""Prepare stationary initial PR receiver stocks against frozen reservoir fields.

Outputs a run_dynamic_receiver configuration and a separate initialization
ledger. Supports one single-phase receiver with positive producing wells.
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from run_dynamic_receiver import fingerprints

from simulator.fluid import Fluid
from simulator.reservoir import Reservoir
from simulator.simulation import Simulator
from simulator.well import WellManager


def prepare(config, *, device='cpu', surface_subcycling=False):
    raw=deepcopy(config)
    if set(raw) != {'simulator','intervals','qualification'}:
        raise ValueError('Expected a dynamic receiver run configuration')
    spec=raw['simulator']
    receiver=spec['simulation'].get('dynamic_receiver')
    if receiver is None:
        raise ValueError('Explicit dynamic_receiver specification required')
    if type(surface_subcycling) is not bool:
        raise ValueError('surface_subcycling must be boolean')
    if surface_subcycling:
        receiver['control']['surface_subcycling']=True
    before=fingerprints()
    dev=torch.device(device)
    reservoir=Reservoir(spec['reservoir'],device=dev)
    fluid=Fluid(spec['fluid'],reservoir,device=dev)
    sim=Simulator(reservoir,fluid,WellManager(spec['wells'],reservoir),spec['simulation'],device=dev)
    state,report=sim.facility_coupling.prepare_operating_point()
    receiver['initial_state']=state.to_dict()
    if fingerprints()!=before:
        raise RuntimeError('Sources changed during initialization')
    report['source_sha256']=before
    raw['qualification'] += (' Receiver initial stocks explicitly prepared for stationary '
        'fixed-reservoir well rates; the reservoir itself is not stationary. '
        'No field calibration is implied.')
    return raw,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--surface-subcycling',action='store_true')
    args=parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError('Refusing to overwrite an existing output directory')
    raw,report=prepare(json.loads(args.config.read_text()),device=args.device,
                       surface_subcycling=args.surface_subcycling)
    args.output_directory.mkdir(parents=True,exist_ok=False)
    for name,data in [('config.json',raw),('initialization.json',report)]:
        (args.output_directory/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'prepared':str(args.output_directory/'config.json'),
                      'pressure_iterations':len(report['pressure_trials'])}))


if __name__=='__main__':
    main()
