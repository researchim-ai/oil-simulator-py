"""Preserve public dry-channel velocity observations and their source contract."""

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

from import_fetrig_measurements import METADATA, SOURCE, request

FILES = {
    379445: "53d3cb9556af90a0356ec9cb2b0b5914",
    379719: "d6f1867a9590bdf502efd8703aac870d",
    379536: "1daab9fad268aa05919548d3c9107d23",
}


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Import destination must not exist")
    with request(METADATA) as response:
        metadata_raw = response.read(2 * 1024**2)
    metadata = json.loads(metadata_raw)
    version = metadata["data"]["latestVersion"]
    if version["license"]["name"] != "CC BY 4.0":
        raise ValueError("Unexpected source license")
    available = {f["dataFile"]["id"]: f["dataFile"] for f in version["files"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fetrig-profile-", dir=output.parent) as temp:
        stage = Path(temp) / "verified"
        stage.mkdir()
        members = []
        for identifier, expected in FILES.items():
            item = available[identifier]
            name = item["filename"]
            if (
                Path(name).name != name
                or item["filesize"] > 1024**2
                or item["checksum"] != {"type": "MD5", "value": expected}
            ):
                raise ValueError("Public gas-profile source identity changed")
            url = f"https://darus.uni-stuttgart.de/api/access/datafile/{identifier}"
            with request(url) as response:
                raw = response.read(item["filesize"] + 1)
            if len(raw) != item["filesize"] or hashlib.md5(raw).hexdigest() != expected:
                raise ValueError("Downloaded profile identity changed")
            (stage / (name + ".gz")).write_bytes(gzip.compress(raw, mtime=0))
            members.append(
                {
                    "original_member": name,
                    "file": name + ".gz",
                    "datafile_id": identifier,
                    "original_size": len(raw),
                    "md5": expected,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        manifest = {
            "format": "fetrig_dry_gas_profiles_v1",
            "source": SOURCE,
            "license": version["license"],
            "members": members,
            "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
            "selection": "Published velocity overview, author plotting script and campaign description; no scoring-based selection",
            "qualified_for_wet_interfacial_shear": False,
            "importer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (stage / "source_metadata.json.gz").write_bytes(gzip.compress(metadata_raw, mtime=0))
        (stage / "LICENSE.txt").write_text(
            "FETRig dataset, Wirth et al., doi:10.18419/DARUS-4998.\nCC BY 4.0 https://creativecommons.org/licenses/by/4.0/\nOriginal files preserved unchanged and gzip compressed; hashes in manifest.json.\n"
        )
        stage.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"members": len(result["members"]), "wet_shear_qualification": False}))
