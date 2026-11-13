"""Read raw FETRig LabView XML without executing workbook formulas.

Every channel has its own clock. Empty Unit metadata remains unconfirmed;
physical conversion is only a declared candidate until independently documented.
"""

import io
import math
import posixpath
import statistics
import xml.etree.ElementTree as ET
import zipfile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def workbook_tables(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        if workbook.find("s:workbookPr", NS) is not None and workbook.find("s:workbookPr", NS).get(
            "date1904"
        ) in ("1", "true"):
            raise ValueError("Unconfirmed alternate workbook clock epoch")
        strings = [
            "".join(node.itertext())
            for node in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)
        ]
        relations = {
            node.get("Id"): node.get("Target")
            for node in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        result = {}
        for sheet in workbook.findall("s:sheets/s:sheet", NS):
            target = relations[sheet.get(f"{{{REL}}}id")]
            path = (
                target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            )
            rows = []
            for row in ET.fromstring(archive.read(path)).findall("s:sheetData/s:row", NS):
                fields = {}
                for cell in row.findall("s:c", NS):
                    if cell.find("s:f", NS) is not None:
                        raise ValueError("Raw experimental workbook unexpectedly contains formulas")
                    value = cell.find("s:v", NS)
                    if value is None:
                        continue
                    column = "".join(c for c in cell.get("r") if c.isalpha())
                    fields[column] = (
                        strings[int(value.text)] if cell.get("t") == "s" else float(value.text)
                    )
                rows.append({"row": int(row.get("r")), "cells": fields})
            result[sheet.get("name")] = rows
        return result


def channel(tables, sheet, clock, name):
    rows = tables[sheet]
    header = rows[0]["cells"]
    columns = {value: key for key, value in header.items()}
    tcol, vcol = columns[clock], columns[name]
    times, values, source_rows = [], [], []
    missing = 0
    for row in rows[1:]:
        fields = row["cells"]
        if tcol not in fields and vcol not in fields:
            continue  # Another independently sampled channel may extend farther.
        if tcol not in fields or vcol not in fields:
            missing += 1
            continue
        t, value = float(fields[tcol]), float(fields[vcol])
        if not math.isfinite(t) or not math.isfinite(value):
            raise ValueError(f"Nonfinite experimental channel {name}")
        if times and t <= times[-1]:
            raise ValueError(f"Nonmonotone experimental channel clock {name}")
        times.append(t)
        values.append(value)
        source_rows.append(row["row"])
    if len(times) < 2:
        raise ValueError(f"Insufficient channel observations {name}")
    root = tables["Messdaten (root)"]
    records = [r["cells"] for r in root if r["cells"].get("A") == name]
    if len(records) != 1:
        raise ValueError(f"Ambiguous experimental channel metadata {name}")
    return {
        "name": name,
        "sheet": sheet,
        "clock": clock,
        "times_excel_days": times,
        "values_native": values,
        "source_rows": source_rows,
        "unit_from_workbook": records[0].get("C"),
        "missing_pairs": missing,
    }


def window_statistics(series, start_day, end_day, *, minimum_samples=10, maximum_gap_s=2.0):
    """No interpolation or positional pairing across instrument clocks."""
    pairs = [
        (t, v, r)
        for t, v, r in zip(
            series["times_excel_days"], series["values_native"], series["source_rows"], strict=True
        )
        if start_day <= t < end_day
    ]
    if len(pairs) < minimum_samples:
        return None
    times, values, rows = zip(*pairs)
    gaps = [
        (times[0] - start_day) * 86400,
        (end_day - times[-1]) * 86400,
        *((b - a) * 86400 for a, b in zip(times, times[1:])),
    ]
    if max(gaps) > maximum_gap_s:
        return None
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    return {
        "samples": len(pairs),
        "first_row": rows[0],
        "last_row": rows[-1],
        "mean_native": mean,
        "sd_native": sd,
        "minimum_native": min(values),
        "maximum_native": max(values),
        "maximum_gap_s": max(gaps),
        "coefficient_of_variation": sd / abs(mean) if mean else None,
    }
