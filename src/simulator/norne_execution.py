"""Explicit, reproducible threading for the Norne benchmark launcher."""
import os

import torch


def configure_norne_threads(device, environ=None):
    """Respect OpenMP/MKL settings; keep NORNE_THREADS as an explicit override.

    The moderate Norne CPU case benefits from a small intra-op pool. This is
    a launcher default, not a global policy for other reservoir/surface models.
    GPU runs keep one host thread. PyTorch also applies the override to MKL.
    """
    env = os.environ if environ is None else environ
    before = torch.get_num_threads()
    if "NORNE_THREADS" in env:
        try:
            count = int(env["NORNE_THREADS"])
        except (ValueError, TypeError):
            raise ValueError("NORNE_THREADS must be a positive integer") from None
        if count < 1:
            raise ValueError("NORNE_THREADS must be a positive integer")
        source = "NORNE_THREADS"
    elif "MKL_NUM_THREADS" in env or "OMP_NUM_THREADS" in env:
        count = before
        source = "OpenMP/MKL environment"
    else:
        available = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                     else os.cpu_count() or 1)
        count = min(4, available) if device.type == "cpu" else 1
        source = "Norne launcher default"
    if count != before:
        torch.set_num_threads(count)
    return {"source": source, "before": before, "effective": torch.get_num_threads()}
