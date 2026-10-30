"""Deliberately break validation in an isolated copy, run the same fast suite, discard the copy."""

import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = (ROOT / "release.py").read_bytes()
    original_sha = hashlib.sha256(original).hexdigest()
    before = original.decode()
    after = before.replace(
        "def validate_release(manifest: dict, report: dict) -> None:\n",
        "def validate_release(manifest: dict, report: dict) -> None:\n"
        "    return  # DELIBERATE MUTATION: skip the whole release contract\n",
        1,
    )
    assert before != after
    reports = ROOT / "reports"
    (reports / "intentional-break.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="release.py",
                tofile="release.py.mutated",
            )
        )
    )
    with tempfile.TemporaryDirectory(prefix="hw8910-mutation-") as tmp:
        copy = Path(tmp)
        for path in ROOT.glob("*.py"):
            shutil.copyfile(path, copy / path.name)
        for name in ("tests", "checks", "checks_extra", "scripts"):
            shutil.copytree(ROOT / name, copy / name, ignore=shutil.ignore_patterns("__pycache__"))
        (copy / "release.py").write_text(after)
        p = subprocess.run(
            [sys.executable, str(copy / "scripts/check_all.py")],
            cwd=copy,
            capture_output=True,
            text=True,
            timeout=300,
        )
        (reports / "fast-red.log").write_text(p.stdout + p.stderr)
    restored_sha = hashlib.sha256((ROOT / "release.py").read_bytes()).hexdigest()
    result = {
        "mutated_check_exit_code": p.returncode,
        "original_sha256": original_sha,
        "after_sha256": restored_sha,
        "original_unchanged": original_sha == restored_sha,
        "intentional_mutation_detected": p.returncode != 0,
    }
    (reports / "mutation-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert result["intentional_mutation_detected"] and result["original_unchanged"]


if __name__ == "__main__":
    main()
