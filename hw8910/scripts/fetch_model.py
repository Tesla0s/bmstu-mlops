"""Download the separate model artifact, verify archive and every file, then export normally."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--destination", type=Path, default=Path(".local/model-source"))
    args = p.parse_args()
    manifest = json.loads((ROOT / "model-artifact.json").read_text())
    origin = json.loads((ROOT / "reports/model-origin.json").read_text())
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "model.zip"
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(manifest["url"], timeout=60) as response:
                    data = response.read(manifest["bytes"] + 1)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                time.sleep(attempt + 1)
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        if len(data) != manifest["bytes"] or hashlib.sha256(data).hexdigest() != manifest["sha256"]:
            raise ValueError("Model archive checksum mismatch; no retry for invalid content")
        archive.write_bytes(data)
    with zipfile.ZipFile(archive) as bundle:
        expected = {"adapter/" + name for name in origin["file_hashes"]}
        if set(bundle.namelist()) != expected or len(bundle.namelist()) != len(expected):
            raise ValueError("Unexpected or duplicate archive entries")
        for entry in bundle.namelist():
            name = entry.removeprefix("adapter/")
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe artifact path")
            data = bundle.read(entry)
            if hashlib.sha256(data).hexdigest() != origin["file_hashes"][name]:
                raise ValueError("Artifact file checksum mismatch: " + name)
            target = destination / "adapter" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print("VERIFIED_MODEL", origin["model_sha256"], destination / "adapter")


if __name__ == "__main__":
    main()
