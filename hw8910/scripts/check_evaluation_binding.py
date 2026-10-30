"""Check that both saved reports belong to the pinned evaluation run, using an isolated copy."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import binding  # noqa: E402
import release  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    workspace = args.workspace.resolve()
    manifest = release.read(workspace / "accepted.json")
    prepared = {"input_sha256": release.digest((workspace / "input.txt").read_bytes())}
    records = []
    with tempfile.TemporaryDirectory(prefix="hw8910-evaluation-binding-") as tmp:
        source = Path(tmp)
        for name in ("evaluation", "reports/evaluation", "data/eval"):
            shutil.copytree(binding.HW67 / name, source / name)
        shutil.copyfile(binding.HW67 / "uv.lock", source / "uv.lock")
        paths = {role: source / f"reports/evaluation/{role}-eval.json" for role in ("baseline", "candidate")}
        originals = {role: path.read_bytes() for role, path in paths.items()}
        with patch.object(binding, "HW67", source):
            valid = binding.verify_saved_evaluation(workspace, manifest, prepared)
            records.append(
                {
                    "case": "unchanged-real-reports",
                    "passed": valid["passed"],
                    "evaluation_run_id": valid["evaluation_run_id"],
                }
            )
            for roles in (["candidate"], ["baseline"], ["candidate", "baseline"]):
                for role, content in originals.items():
                    paths[role].write_bytes(content)
                for role in roles:
                    report = json.loads(originals[role])
                    report["evaluation_run_id"] = "different-evaluation-run"
                    release.write(paths[role], report)
                try:
                    bound = binding.verify_saved_evaluation(workspace, manifest, prepared)
                except ValueError as exc:
                    records.append(
                        {"case": "foreign-" + "-and-".join(roles), "rejected": True, "reason": str(exc)}
                    )
                else:
                    records.append(
                        {
                            "case": "foreign-" + "-and-".join(roles),
                            "rejected": False,
                            "accepted_evaluation_run_id": bound["evaluation_run_id"],
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    if not all(x.get("rejected", True) for x in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
