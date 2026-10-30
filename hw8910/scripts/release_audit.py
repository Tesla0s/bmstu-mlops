"""Recorded real-model gate checks: restart cache and refusal preserve the accepted pointer."""

import argparse
import json
from pathlib import Path
import shutil
from audit_support import Audit, release


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports/real-release"))
    args = p.parse_args()
    root = args.workspace.resolve()
    audit = Audit(args.output)
    prefix = [
        "lab.py",
        "--workspace",
        str(root),
        "run",
        "--manifest",
        "manifest-real.json",
        "--report",
        "release-report.json",
    ]
    before = release.digest((root / "accepted.json").read_bytes())
    original = (root / "input.txt").read_bytes()
    first = audit.run("01-failure-after-preparation", prefix + ["--fail-after-prepare"], "nonzero")
    assert "PREPARED" in first.stdout or "CACHE_HIT" in first.stdout
    recovered = audit.run("02-recover", prefix)
    assert "CACHE_HIT" in recovered.stdout
    audit.run("03-repeat", prefix)
    assert len((root / "publications.jsonl").read_text().splitlines()) == 1
    try:
        row = json.loads(original.splitlines()[0])
        row["id"] += "-modified"
        (root / "input.txt").write_bytes(original + json.dumps(row).encode() + b"\n")
        changed = audit.run("04-changed-evaluation-input", prefix, "nonzero")
        assert "PREPARED" in changed.stdout and "CACHE_HIT" not in changed.stdout
        after = release.digest((root / "accepted.json").read_bytes())
        assert before == after
        audit.record({"step": "changed-data-refused", "before_sha256": before, "after_sha256": after})
    finally:
        (root / "input.txt").write_bytes(original)
    bad = release.read(root / "manifest-real.json")
    bad["runtime"]["dtype"] = "float16"
    bad["runtime_sha256"] = release.fingerprint(bad["runtime"])
    release.write(root / "manifest-incompatible-runtime.json", bad)
    audit.run(
        "05-different-runtime-with-consistent-hash",
        [
            "lab.py",
            "--workspace",
            str(root),
            "run",
            "--manifest",
            "manifest-incompatible-runtime.json",
            "--report",
            "report-mismatch.json",
        ],
        "nonzero",
    )
    after = release.digest((root / "accepted.json").read_bytes())
    assert before == after
    audit.record({"step": "runtime-refused", "before_sha256": before, "after_sha256": after})
    audit.run("06-restored-valid-input", prefix)
    for name in (
        "accepted.json",
        "manifest-real.json",
        "release-report.json",
        "attempts.jsonl",
        "publications.jsonl",
    ):
        shutil.copyfile(root / name, audit.output / name)
    audit.record({"step": "complete", "passed": True, "publication_events": 1})


if __name__ == "__main__":
    main()
