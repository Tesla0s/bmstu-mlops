"""Replay the saved 60+4 evaluation inputs through the real HTTP API, then benchmark and restart."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from audit_support import Audit, ROOT, release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/re-audit/real-api"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = release.read(args.workspace / "accepted.json")
    hw67 = ROOT.parent / "hw67"
    cfg = release.read(hw67 / "evaluation/params.json")
    rows = [
        json.loads(line)
        for key in ("golden", "behavioral")
        for line in (hw67 / cfg[key]).read_text().splitlines()
        if line.strip()
    ]
    saved = {row["id"]: row for row in release.read(hw67 / "reports/evaluation/candidate-outputs.json")}
    audit = Audit(args.output)
    comparisons = []
    try:
        url, server = audit.server(args.workspace, warmup=2)
        assert audit.http(url, "/health")[0] == 200
        assert audit.http(url, "/ready")[0] == 503
        assert audit.http(url, "/generate", {"prompt": "not ready yet"})[0] == 503
        audit.ready(url)
        before = audit.http(url, "/debug")[1]["calls"]
        for body in (
            {"prompt": ""},
            {"prompt": "ok", "max_new_tokens": 33},
            {"prompt": "x" * 8193},
            {"prompt": "x " * 3500},
        ):
            assert 400 <= audit.http(url, "/generate", body)[0] < 500
        assert audit.http(url, "/debug")[1]["calls"] == before
        for row in rows:
            status, answer = audit.http(
                url, "/generate", {"prompt": row["input"], "max_new_tokens": 32}, row["id"]
            )
            comparison = {
                "id": row["id"],
                "status": status,
                "expected": saved[row["id"]]["raw"],
                "actual": answer.get("answer"),
                "matches_saved": answer.get("answer") == saved[row["id"]]["raw"],
                "model_id": answer.get("model_id"),
                "runtime_sha256": answer.get("runtime_sha256"),
                "output_tokens": answer.get("output_tokens"),
                "generation_seconds": answer.get("generation_seconds"),
            }
            comparisons.append(comparison)
            (args.output / "comparisons.json").write_text(
                json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n"
            )
            assert status == 200 and comparison["model_id"] == manifest["model_id"]
            assert comparison["runtime_sha256"] == manifest["runtime_sha256"]
            print(
                "REAL_HTTP",
                len(comparisons),
                "/",
                len(rows),
                "MATCH",
                comparison["matches_saved"],
                flush=True,
            )
        assert all(x["matches_saved"] for x in comparisons), (
            "API output differs from the saved HW67 evaluation"
        )
        initial = ROOT / "reports/re-audit/initial-benchmark"
        if not initial.exists():
            shutil.copytree(ROOT / "reports/benchmark", initial)
            shutil.copyfile(ROOT / "reports/benchmark.log", initial / "run.log")
        with (ROOT / "reports/benchmark.log").open("w") as log:
            benchmark = subprocess.run(
                [sys.executable, "scripts/benchmark.py", "--url", url],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        assert benchmark.returncode == 0
        server.terminate()
        server.wait(timeout=30)
        url, _ = audit.server(args.workspace)
        ready = audit.ready(url)
        assert ready["model_id"] == manifest["model_id"]
        status, answer = audit.http(
            url, "/generate", {"prompt": rows[0]["input"], "max_new_tokens": 32}, "after-real-restart"
        )
        assert status == 200 and answer["answer"] == saved[rows[0]["id"]]["raw"]
        assert answer["runtime_sha256"] == manifest["runtime_sha256"]
        audit.record(
            {
                "step": "complete",
                "passed": True,
                "matched_evaluation_answers": len(comparisons),
                "expected_evaluation_answers": 64,
                "benchmark_attempts": 16,
                "restarted_real_model": True,
            }
        )
        print("REAL_RECHECK_PASSED", len(comparisons), flush=True)
    finally:
        audit.close()


if __name__ == "__main__":
    main()
