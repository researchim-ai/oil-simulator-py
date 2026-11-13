"""Import a fixed FETRig film-data subset without downloading its 5.4 GB archive.

The rectangular channel is NOT the Hannover cylindrical pilot. This importer
preserves raw observations and their authors' processing descriptions; it does
not declare them a successful validation of an annular pipe model.
"""

import argparse
import gzip
import hashlib
import io
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SOURCE = "https://doi.org/10.18419/DARUS-4998"
METADATA = "https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:10.18419/DARUS-4998"
ARCHIVE = "https://darus.uni-stuttgart.de/api/access/datafile/379564"
ARCHIVE_SIZE = 5396814507
ARCHIVE_MD5 = "22e310e40bedfdbdfb2811cc5d17116b"
PREFIX = "Film Data_MC_RC_Water_RH80_T25Deg_CopperPlate_28_08_24/"
DATA = PREFIX + "MC_RC_Water_RH80_T25Deg_CopperPlate_28_08_24/"


def request(url, **headers):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **headers}), timeout=30
    )


class RangeArchive(io.RawIOBase):
    """Bounded HTTP seekable input; ZipFile handles central-directory ZIP64.

    Refuse ignored or shifted byte ranges before reading the body. No silent
    fallback to downloading the complete archive. Each member CRC is checked by
    ZipFile and original size/CRC/SHA are retained in the import manifest.
    """

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.bytes_read = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        position = base + offset
        if position < 0:
            raise ValueError("Negative archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        count = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if count <= 0:
            return b""
        if count > 4 * 1024**2 or self.bytes_read + count > 32 * 1024**2:
            raise ValueError("Public archive transfer exceeds declared subset budget")
        end = self.position + count - 1
        expected = f"bytes {self.position}-{end}/{self.size}"
        with request(self.url, Range=f"bytes={self.position}-{end}") as response:
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server ignored or changed archive byte range")
            data = response.read(count + 1)
        if len(data) != count:
            raise ValueError("Truncated or oversized archive range")
        self.position += count
        self.bytes_read += count
        return data


def selection():
    """Fixed before comparison: two axial stations, same liquid setpoint.

    Names encode setpoints, not independently verified measured Reynolds numbers.
    Corresponding LabView books are always kept to recover actual conditions.
    """
    names = [
        PREFIX + name
        for name in (
            "Keyence_Data_Manager.m",
            "LabView_Data_Manager.m",
            "FETRig_Report.m",
            "Sonstige_Plots_fuer das Paper/Check_Film_Thickness.m",
        )
    ]
    names += [DATA + "README.md"]
    for station, suffix in ((100, ""), (600, " 3")):
        for gas in (0, 12900, 20900):
            stem = f"ReL500_ReG{gas}"
            base = DATA + f"Y_{station}/X_50/"
            names.extend(
                [
                    base + f"Keyence Data{suffix}/{stem}.csv",
                    base + f"LabView Data{suffix}/{stem}.xlsx",
                ]
            )
    return names


def run(output, *, processing_only=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Import destination must not exist")
    with request(METADATA) as response:
        metadata_raw = response.read(2 * 1024**2)
    metadata = json.loads(metadata_raw)
    version = metadata["data"]["latestVersion"]
    if version["license"]["name"] != "CC BY 4.0":
        raise ValueError("Unexpected source license")
    source = next(f["dataFile"] for f in version["files"] if f["dataFile"]["id"] == 379564)
    if source["filesize"] != ARCHIVE_SIZE or source["checksum"] != {
        "type": "MD5",
        "value": ARCHIVE_MD5,
    }:
        raise ValueError("Published archive identity changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fetrig-import-", dir=output.parent) as temp:
        stage = Path(temp) / "verified"
        stage.mkdir()
        entries = []
        remote = RangeArchive(ARCHIVE, ARCHIVE_SIZE)
        with zipfile.ZipFile(remote) as archive:
            names = (
                [
                    PREFIX + "Post_Process_Data_Handling.mlx",
                    PREFIX + "Post_Processing/Keyence_Data_Plotter.m",
                ]
                if processing_only
                else selection()
            )
            for name in names:
                info = archive.getinfo(name)
                if info.file_size > 4 * 1024**2:
                    raise ValueError("Selected member exceeds expected measurement size")
                raw = archive.read(info)  # CRC checked by stdlib, no executable MATLAB loaded.
                relative = name.removeprefix(PREFIX)
                target = stage / (relative + ".gz")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gzip.compress(raw, mtime=0))
                entries.append(
                    {
                        "original_member": name,
                        "file": str(target.relative_to(stage)),
                        "original_size": len(raw),
                        "zip_crc32": f"{info.CRC:08x}",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
        manifest = {
            "format": "fetrig_processing_subset_v1" if processing_only else "fetrig_raw_subset_v1",
            "source": SOURCE,
            "license": version["license"],
            "archive_url": ARCHIVE,
            "archive_size": ARCHIVE_SIZE,
            "published_archive_md5_not_full_file_recomputed": ARCHIVE_MD5,
            "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
            "importer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "transferred_archive_bytes": remote.bytes_read,
            "selection": "Original processing descriptions only, not executed"
            if processing_only
            else "Y_100 and Y_600, X_50, filename ReL500; ReG0/12900/20900, both raw instruments; chosen before model scoring",
            "geometry": {
                "channel_width_m": 0.1,
                "channel_depth_m": 0.03,
                "source": "https://doi.org/10.1016/j.applthermaleng.2026.130975, section 2.1.3",
            },
            "phase": "water film with countercurrent air; rectangular copper-wall channel",
            "qualified": False,
            "limitations": [
                "Setpoint labels are not measured inputs",
                "No fit or model comparison performed by this importer",
                "No annular-to-rectangular geometry substitution",
                "Timing, actual conditions and sensor acceptance must be resolved before scoring",
            ],
            "members": entries,
        }
        (stage / "source_metadata.json.gz").write_bytes(gzip.compress(metadata_raw, mtime=0))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (stage / "LICENSE.txt").write_text(
            "FETRig dataset, Wirth et al., doi:10.18419/DARUS-4998.\n"
            "Original observations and processing descriptions, CC BY 4.0.\n"
            "https://creativecommons.org/licenses/by/4.0/\n"
            "Files extracted unchanged and gzip-compressed; original names and hashes in manifest.json.\n"
        )
        stage.rename(output)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--processing-only", action="store_true")
    args = p.parse_args()
    report = run(args.output, processing_only=args.processing_only)
    print(
        json.dumps(
            {
                "members": len(report["members"]),
                "transferred_archive_bytes": report["transferred_archive_bytes"],
                "qualified": report["qualified"],
            }
        )
    )
