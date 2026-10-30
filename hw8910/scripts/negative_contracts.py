"""Additional negative evidence; lightweight checks do not load or imitate a real model."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release  # noqa: E402
from lab import initialize  # noqa: E402
from publication import publish_once  # noqa: E402
from backends import load_backend  # noqa: E402


def main():
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "fixture"
        initialize(root)
        m, original = release.read(root / "manifest-B.json"), release.read(root / "report-B.json")
        before = release.digest((root / "accepted.json").read_bytes())
        cases = {}
        for key, value in (
            ("status", "running"),
            ("passed", "true"),
            ("passed", False),
            ("model_id", "foreign"),
            ("evaluation_run_id", ""),
        ):
            report = copy.deepcopy(original)
            report[key] = value
            cases[key + "=" + str(value)] = report
        for value in (float("nan"), float("inf"), True):
            report = copy.deepcopy(original)
            report["metrics"]["accuracy"] = value
            cases["accuracy=" + str(value)] = report
        for name, report in cases.items():
            try:
                release.publish(m, report, root / "accepted.json")
            except ValueError as exc:
                error = str(exc)
            else:
                raise AssertionError(name)
            after = release.digest((root / "accepted.json").read_bytes())
            assert before == after
            results.append(
                {
                    "case": name,
                    "rejected": True,
                    "reason": error,
                    "before_sha256": before,
                    "after_sha256": after,
                }
            )
        with patch("os.replace", side_effect=OSError("TEST disk error")):
            try:
                release.publish(m, original, root / "accepted.json")
            except OSError:
                pass
            else:
                raise AssertionError("Disk failure was ignored")
        assert not list(root.glob("*.tmp"))
        assert release.digest((root / "accepted.json").read_bytes()) == before
        results.append(
            {
                "case": "atomic-replace-error",
                "rejected": True,
                "before_sha256": before,
                "after_sha256": before,
                "temporary_files": 0,
            }
        )
        (root / "model-B.txt").write_text("Modified artifact bytes\n")
        try:
            publish_once(root, m, original, {"input_sha256": "fixture"})
        except ValueError as exc:
            results.append({"case": "file-tampering-before-publish", "rejected": True, "reason": str(exc)})
        else:
            raise AssertionError("Tampered model was published")
        (root / "model-A.txt").write_text("Modified accepted artifact bytes\n")
        try:
            load_backend(root)
        except ValueError as exc:
            results.append({"case": "file-tampering-before-load", "rejected": True, "reason": str(exc)})
        else:
            raise AssertionError("Tampered model was loaded")
    Path("reports/negative-contracts.json").write_text(json.dumps(results, indent=2) + "\n")
    print("NEGATIVE_CASES_PASSED", len(results))


if __name__ == "__main__":
    main()
