"""Small dependency-free contracts; the workflow separately verifies real artifacts and the HW67 gate."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

DIGEST = re.compile(r"^[0-9a-f]{64}$")
BINDINGS = ("model_id", "model_sha256", "protocol_sha256", "golden_sha256", "runtime_sha256")


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return digest(canonical(value))


def read(path):
    return json.loads(Path(path).read_text())


def cache_key(data_path: Path, config: dict) -> str:
    return fingerprint({"data_sha256": digest(Path(data_path).read_bytes()), "config": config})


def validate_release(manifest: dict, report: dict) -> None:
    try:
        if not isinstance(manifest, dict) or not isinstance(report, dict):
            raise ValueError("Manifest and report must be objects")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ValueError("Unsupported schema_version")
        for key in ("model_id", "artifact_uri"):
            if not isinstance(manifest[key], str) or not manifest[key].strip():
                raise ValueError("Empty " + key)
        for key in BINDINGS[1:]:
            if not isinstance(manifest[key], str) or not DIGEST.fullmatch(manifest[key]):
                raise ValueError("Invalid digest: " + key)
        runtime = manifest["runtime"]
        if not isinstance(runtime, dict) or not runtime:
            raise ValueError("Empty runtime")
        if fingerprint(runtime) != manifest["runtime_sha256"]:
            raise ValueError("Runtime hash mismatch")
        if report["status"] != "complete" or report["passed"] is not True:
            raise ValueError("Evaluation incomplete or rejected")
        if not isinstance(report["evaluation_run_id"], str) or not report["evaluation_run_id"].strip():
            raise ValueError("Missing evaluation_run_id")
        for key in BINDINGS:
            if report[key] != manifest[key]:
                raise ValueError("Foreign evaluation: " + key)
        for key in ("accuracy", "format_rate"):
            value = report["metrics"][key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid metric: " + key)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Missing or malformed release field: " + str(exc)) from exc


def atomic_bytes(target, payload):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix="." + target.name + "-", suffix=".tmp", delete=False
        ) as f:
            temp = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, target)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def write(target, value):
    atomic_bytes(target, canonical(value) + b"\n")


def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest, report)
    payload = canonical(manifest) + b"\n"
    target = Path(target)
    if target.exists() and target.read_bytes() == payload:
        return
    atomic_bytes(target, payload)
