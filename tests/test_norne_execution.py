"""Norne threading must be bounded and reproducible without changing numerics."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.norne_execution import configure_norne_threads


@pytest.fixture
def pool(monkeypatch):
    state = {"threads": 12, "sets": []}
    monkeypatch.setattr(torch, "get_num_threads", lambda: state["threads"])
    def set_threads(n):
        state["sets"].append(n)
        state["threads"] = n
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    monkeypatch.setattr("simulator.norne_execution.os.sched_getaffinity", lambda _: {0, 1, 2, 3, 4, 5})
    return state


@pytest.mark.parametrize("device,expected", [("cpu", 4), ("cuda:0", 1)])
def test_device_defaults(pool, device, expected):
    result = configure_norne_threads(torch.device(device), {})
    assert result == {"source": "Norne launcher default", "before": 12, "effective": expected}
    assert pool["sets"] == [expected]


def test_default_respects_process_affinity(pool, monkeypatch):
    monkeypatch.setattr("simulator.norne_execution.os.sched_getaffinity", lambda _: {5})
    assert configure_norne_threads(torch.device("cpu"), {})["effective"] == 1


@pytest.mark.parametrize("setting", ["OMP_NUM_THREADS", "MKL_NUM_THREADS"])
def test_explicit_environment_is_preserved(pool, setting):
    result = configure_norne_threads(torch.device("cpu"), {setting: "12"})
    assert result["effective"] == 12
    assert not pool["sets"]


def test_norne_override_takes_priority(pool):
    result = configure_norne_threads(torch.device("cpu"), {"NORNE_THREADS": "2", "MKL_NUM_THREADS": "1"})
    assert result["effective"] == 2
    assert result["source"] == "NORNE_THREADS"


@pytest.mark.parametrize("value", ["", "0", "-1", "2.5", "many"])
def test_invalid_override_fails_before_solving(pool, value):
    with pytest.raises(ValueError, match="positive integer"):
        configure_norne_threads(torch.device("cpu"), {"NORNE_THREADS": value})
    assert not pool["sets"]
