"""Archive extraction must preserve observations and refuse unbounded transfers."""

import io
import zipfile
from pathlib import Path

import import_fetrig_measurements as importer
import pytest


def test_seekable_range_reader_uses_zip_checksums_without_whole_archive(monkeypatch):
    data = io.BytesIO()
    with (
        zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive,
        archive.open("raw.csv", "w", force_zip64=True) as member,
    ):
        member.write(b"Time,Thickness\n0,0.003\n")
    raw = data.getvalue()
    requested = []

    def request(url, **headers):
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        requested.append((start, end))
        response = io.BytesIO(raw[start : end + 1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(raw)}"}
        return response

    monkeypatch.setattr(importer, "request", request)
    with zipfile.ZipFile(importer.RangeArchive("synthetic", len(raw))) as archive:
        assert archive.read("raw.csv") == b"Time,Thickness\n0,0.003\n"
    assert requested


@pytest.mark.parametrize("status,content_range", [(200, "bytes 0-9/100"), (206, "bytes 10-19/100")])
def test_ignored_or_shifted_range_is_rejected_before_read(monkeypatch, status, content_range):
    class Response(io.BytesIO):
        def read(self, *args):
            pytest.fail("Must reject HTTP headers before downloading body")

    def request(url, **headers):
        response = Response(b"unexpected")
        response.status = status
        response.headers = {"Content-Range": content_range}
        return response

    monkeypatch.setattr(importer, "request", request)
    with pytest.raises(ValueError, match="byte range"):
        importer.RangeArchive("synthetic", 100).read(10)


def test_full_five_gigabyte_read_is_refused_before_network(monkeypatch):
    monkeypatch.setattr(importer, "request", lambda *a, **k: pytest.fail("Unexpected network"))
    with pytest.raises(ValueError, match="budget"):
        importer.RangeArchive("synthetic", importer.ARCHIVE_SIZE).read()


def test_cci_uses_output_mm_and_status_without_interpolating_sensor_alarms():
    from audit_fetrig_measurements import cci_statistics

    raw = (
        b"Sampling cycle,200us\nData count,3\nFirst index,10\n"
        b"Display unit,0.0001mm\nTrigger count,OUT01(mm),\n"
        b"10,0.4,GO\n11,99999,ALARM\n13,0.8,GO\n"
    )
    result = cci_statistics(raw)
    assert result["go_mean_thickness_m"] == pytest.approx(0.0006)
    assert result["go_sample_std_thickness_m"] == pytest.approx(0.0004 / 2**0.5)
    assert result["sensor_acceptance_fraction"] == pytest.approx(2 / 3)
    assert result["elapsed_first_to_last_s"] == pytest.approx(0.0006)
    assert result["missing_triggers"] == 1
    with pytest.raises(ValueError, match="status"):
        cci_statistics(raw.replace(b"ALARM", b"UNKNOWN"))
    with pytest.raises(ValueError, match="count"):
        cci_statistics(raw.replace(b"Data count,3", b"Data count,4"))
    with pytest.raises(ValueError, match="units"):
        cci_statistics(raw.replace(b"OUT01(mm)", b"OUT01(um)"))


def test_fixed_independent_subset_has_six_raw_cases_and_does_not_claim_a_model_pass():
    from audit_fetrig_measurements import run

    report = run(Path(__file__).resolve().parents[1] / "tests/data/surface/fetrig")
    assert report["total_cci_records"] == 600000
    assert len(report["cases"]) == 6
    assert not report["model_scored"] and not report["experimentally_qualified"]
    assert all(row["missing_triggers"] == 0 for row in report["cases"])
    assert any(row["sensor_status_counts"]["ALARM"] > 0 for row in report["cases"])
