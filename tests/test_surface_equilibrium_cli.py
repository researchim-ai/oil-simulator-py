"""Atomic publication and source-snapshot protection of the process CLI."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_source_edit_during_run_preserves_previous_result(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "equilibrium_cli_test", ROOT / "scripts/run_equilibrium_process.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    # Only temporary copies are changed; the live physical implementation is
    # never touched by this test (including while a CUDA run is in flight).
    shutil.copytree(
        ROOT / "src/surface", tmp_path / "src/surface", ignore=shutil.ignore_patterns("__pycache__")
    )
    (tmp_path / "src/simulator").mkdir()
    shutil.copy(
        ROOT / "src/simulator/thermal_surface_boundary.py",
        tmp_path / "src/simulator/thermal_surface_boundary.py",
    )
    (tmp_path / "scripts").mkdir()
    shutil.copy(
        ROOT / "scripts/run_equilibrium_process.py", tmp_path / "scripts/run_equilibrium_process.py"
    )
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "scripts/run_equilibrium_process.py"))
    data = json.loads((ROOT / "configs/surface_equilibrium_process.json").read_text())
    config = tmp_path / "input.json"
    config.write_text(json.dumps(data))
    output = tmp_path / "output.json"
    output.write_text('{"accepted_before":true}\n')
    before = output.read_bytes()
    from simulator import thermal_surface_boundary
    from surface import equilibrium_process

    monkeypatch.setattr(
        thermal_surface_boundary,
        "characterize_well_thermal_boundaries",
        lambda *args: {"streams": {}},
    )

    def edit_snapshot(*args, **kwargs):
        target = tmp_path / "src/surface/equilibrium_properties.py"
        target.write_text(target.read_text() + "\n# changed during calculation\n")
        return {}

    monkeypatch.setattr(equilibrium_process, "run_equilibrium_process", edit_snapshot)
    monkeypatch.setattr(
        "sys.argv", ["run_equilibrium_process.py", str(config), "--output", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Source files changed during the calculation" in capsys.readouterr().err
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".output.json*.tmp"))
