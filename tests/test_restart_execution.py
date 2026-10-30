"""Checkpoint continuation must restore process-wide execution options."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator.eclipse_run import iter_deck, run_deck
from simulator.eclipse_deck import EclipseCase

SPE1 = str(Path(__file__).parent / 'data/spe1/SPE1CASE1.DATA')


@pytest.mark.parametrize('deterministic', [None, True, False])
def test_restart_reapplies_saved_determinism_option(tmp_path, deterministic):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    gen = None
    try:
        checkpoint = str(tmp_path / 'state.pt')
        first_report = EclipseCase(SPE1).report_days[0]
        overrides = {} if deterministic is None else {'deterministic': deterministic}
        run_deck(SPE1, device=torch.device('cpu'), max_days=first_report, checkpoint=checkpoint,
                 sim_overrides=overrides)
        assert Path(checkpoint).exists()
        # A separate Python process starts with this flag off. Reset it here
        # so a preceding fresh run cannot conceal the missing restoration.
        torch.use_deterministic_algorithms(False)
        gen = iter_deck(SPE1, device=torch.device('cpu'), resume=checkpoint, yield_start=True)
        next(gen)
        assert torch.are_deterministic_algorithms_enabled() is (deterministic is not False)
        if deterministic is not False:
            assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        if gen is not None:
            gen.close()
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
