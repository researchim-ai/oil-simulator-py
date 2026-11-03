"""Explicit backend selection and reproducibility metadata for liquid dynamics."""

import copy

import torch

TORCH_SOURCES = (
    "torch_backend.py",
    "torch_linear.py",
    "torch_liquid.py",
    "torch_liquid_bridge.py",
    "torch_steady.py",
    "torch_heat.py",
    "torch_heat_bridge.py",
    "torch_heat_interval.py",
)


def runtime_fingerprint():
    return {
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def liquid_backend_config(config, backend=None, device=None):
    """Copy an acoustic-liquid config and override only its numerical backend.

    Nested heat models inherit the same device. No unrelated PVT/gas model is
    rewritten, and unknown structures fail instead of silently ignoring a flag.
    """
    result = copy.deepcopy(config)
    count = 0

    def visit(value):
        nonlocal count
        if not isinstance(value, dict):
            return
        if "storage_volumes_m3" in value and "network" in value:
            if backend is not None:
                value["backend"] = backend
            if device is not None:
                value["device"] = device
            count += 1
        if "heat" in value and isinstance(value["heat"], dict):
            if backend is not None:
                value["heat"]["backend"] = backend
            if device is not None:
                value["heat"]["device"] = device
        for child in tuple(value.values()):
            if isinstance(child, dict):
                visit(child)

    visit(result)
    if count == 0:
        raise ValueError("No liquid transient physics found for backend selection")
    return result
