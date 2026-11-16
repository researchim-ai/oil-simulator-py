"""Durable partial benchmark evidence; interrupted series are never complete."""

import json
import os
import tempfile
from pathlib import Path


def atomic_json(path, value):
    """Publish a complete JSON document or preserve the previous document."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class BenchmarkJournal:
    def __init__(self, directory, protocol):
        self.directory = Path(directory)
        # A previous interrupted directory also contains evidence. Never
        # silently overwrite it or mix runs from different source versions.
        self.directory.mkdir(parents=True, exist_ok=False)
        atomic_json(self.directory / "protocol.json", protocol)

    def checkpoint(self, runs, comparisons, certificates):
        atomic_json(
            self.directory / "progress.json",
            {
                "status": "incomplete",
                "runs": runs,
                "comparisons": comparisons,
                "audit_certificates": certificates,
            },
        )

    def complete(self, summary):
        atomic_json(self.directory / "summary.json", {**summary, "status": "complete"})
