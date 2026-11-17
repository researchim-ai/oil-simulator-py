"""Prevent a qualification worker from silently exhausting the host."""

import json
import sys
from pathlib import Path

import pytest

from simulator.execution_memory import enforce_host_memory, process_memory


def test_budget_accounts_for_swap_not_only_resident_pages():
    enforce_host_memory(2.0, {"process_rss_bytes": 1024**2, "process_swap_bytes": 0})
    with pytest.raises(MemoryError, match="RSS\\+swap"):
        enforce_host_memory(2.0, {"process_rss_bytes": 1024**2, "process_swap_bytes": 2 * 1024**2})


def test_memory_telemetry_is_measurable_on_linux():
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux telemetry")
    m = process_memory()
    assert m["process_rss_bytes"] > 0 and m["process_swap_bytes"] >= 0
    assert m["process_lifetime_peak_rss_bytes"] > 0


def test_cli_budget_failure_is_recorded_before_any_solver_work(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_fim_facility_forecast as driver

    output = tmp_path / "run"

    def forbidden(*args, **kwargs):
        raise AssertionError("Must stop before solver allocation")

    monkeypatch.setattr(driver, "run_case", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fim_facility_forecast.py",
            "--days",
            "30",
            "--host-memory-limit-mib",
            "1",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(MemoryError, match="Host memory budget"):
        driver.main()
    assert not (output / "summary.json").exists()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["stage"] == "solve" and failure["exception"] == "MemoryError"
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["host_memory_limit_mib"] == 1
