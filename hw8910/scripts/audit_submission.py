"""Offline consistency audit of the recorded assignment evidence (no model load)."""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import release  # noqa: E402
from binding import HW67, quality_gate  # noqa: E402


def main():
    reports = ROOT / "reports"
    checks = {}

    def require(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] = True

    for name, directory, expected in (
        ("hw8910-tests-unchanged", ROOT / "tests", "09bd65e6b886"),
        ("hw67-tests-unchanged", HW67 / "tests", "c42ac7788026"),
    ):
        h = hashlib.sha256()
        for p in sorted(directory.iterdir()):
            if p.is_file():
                h.update(p.name.encode())
                h.update(p.read_bytes())
        require(name, h.hexdigest().startswith(expected))
    defects = list(csv.DictReader((ROOT / "defects.csv").open()))
    require("seven-independent-defects", len(defects) == 7 and all(all(x.values()) for x in defects))
    for row in defects:
        require(
            "evidence-" + row["id"],
            (ROOT / row["before_log"]).is_file() and (ROOT / row["after_log"]).is_file(),
        )
    require(
        "all-seven-red-on-original",
        "FAILED (failures=7)" in (reports / "integration-original.log").read_text(),
    )
    require(
        "twenty-green-without-weights",
        "FAST_CHECK_EXIT_CODES [0, 0]" in (reports / "fast-green.log").read_text(),
    )
    mutation = release.read(reports / "mutation-summary.json")
    require(
        "deliberate-break-red-original-restored",
        mutation["mutated_check_exit_code"] != 0
        and mutation["original_sha256"] == mutation["after_sha256"]
        and mutation["after_sha256"] == release.digest((ROOT / "release.py").read_bytes()),
    )
    require("eleven-negative-cases", len(release.read(reports / "negative-contracts.json")) == 11)
    manifest = release.read(reports / "real-release/accepted.json")
    report = release.read(reports / "real-release/release-report.json")
    release.validate_release(manifest, report)
    require("release-contract", True)
    for name, expected in report["source_sha256"].items():
        require("evaluation-source:" + name, release.digest((HW67 / name).read_bytes()) == expected)
    require(
        "origin-binding",
        report["source_origin_sha256"] == release.digest((reports / "model-origin.json").read_bytes()),
    )
    cfg = release.read(HW67 / "evaluation/params.json")
    baseline, candidate = [
        release.read(HW67 / f"reports/evaluation/{r}-eval.json") for r in ("baseline", "candidate")
    ]
    require(
        "unchanged-full-quality-gate",
        quality_gate(baseline, candidate, cfg["critical_slices"]) == report["quality_gate"],
    )
    attempts = [json.loads(x) for x in (reports / "real-release/attempts.jsonl").read_text().splitlines()]
    require(
        "real-cache-and-invalidation",
        any(x["tasks"].get("prepare") == "Cached" for x in attempts)
        and any(
            x["status"] == "failed"
            and "Changed evaluation data" in x.get("error", "")
            and x["tasks"]["prepare"] == "Completed"
            for x in attempts
        ),
    )
    refusals = [x for x in release.read(reports / "real-release/evidence.json") if "before_sha256" in x]
    require(
        "real-refusals-preserve-bytes",
        len(refusals) == 2 and all(x["before_sha256"] == x["after_sha256"] for x in refusals),
    )
    require(
        "one-real-publication",
        len((reports / "real-release/publications.jsonl").read_text().splitlines()) == 1,
    )
    api = release.read(reports / "real-api/evidence.json")
    responses = [x["response"] for x in api if x["step"].startswith("real-answer")]
    require(
        "real-model-two-starts",
        len(responses) == 2
        and all(
            x["model_id"] == manifest["model_id"]
            and x["runtime_sha256"] == manifest["runtime_sha256"]
            and 0 < x["output_tokens"] <= 32
            and "STUB ONLY" not in x["answer"]
            for x in responses
        ),
    )
    require(
        "real-readiness-and-validation",
        any(x.get("request") == "/ready" and x["status"] == 503 for x in api)
        and sum(x.get("status") == 422 for x in api) == 10,
    )
    debug = [x["response"]["calls"] for x in api if x.get("request") == "/debug"]
    require("no-generation-on-invalid-requests", debug == [1, 1, 1, 1])
    stub = release.read(reports / "stub-experiment/evidence.json")
    stages = (
        "loaded-A",
        "pointer-B-memory-A",
        "restarted-B",
        "rollback-pointer-A-memory-B",
        "rollback-restarted-A",
    )
    require(
        "controlled-A-B-A",
        [next(x["response"]["model_id"] for x in stub if x["step"] == s) for s in stages]
        == ["fixture-A", "fixture-A", "fixture-B", "fixture-B", "fixture-A"],
    )
    require(
        "crash-recovery-no-duplicate",
        "RECOVERED_JOURNAL" in (reports / "stub-experiment/04-recover-journal.log").read_text(),
    )
    rows = [json.loads(x) for x in (reports / "benchmark/attempts.jsonl").read_text().splitlines()]
    summary = release.read(reports / "benchmark/summary.json")
    require(
        "all-sixteen-real-attempts",
        len(rows) == 16
        and all(
            x["status"] == 200
            and x["error"] is None
            and x["model_id"] == manifest["model_id"]
            and x["runtime_sha256"] == manifest["runtime_sha256"]
            for x in rows
        ),
    )
    for s in summary:
        batch = [x for x in rows if x["concurrency"] == s["concurrency"]]
        require(
            "benchmark-summary-" + str(s["concurrency"]),
            len(batch) == 8
            and s["successes"] == 8
            and s["errors"] == 0
            and math.isclose(
                statistics.mean(x["client_seconds"] for x in batch), s["successful_mean_seconds"]
            )
            and math.isclose(max(x["client_seconds"] for x in batch), s["successful_max_seconds"])
            and sum(x["output_tokens"] for x in batch) == s["output_tokens"],
        )
    require(
        "same-inputs-and-answers",
        {x["id"]: x["answer"] for x in rows if x["concurrency"] == 1}
        == {x["id"]: x["answer"] for x in rows if x["concurrency"] == 2},
    )
    access = release.read(reports / "public-artifact-check.json")
    artifact = release.read(ROOT / "model-artifact.json")
    require(
        "public-artifact-anonymous-and-verified",
        access["anonymous_download"]
        and access["archive_sha256"] == artifact["sha256"]
        and access["bytes"] == artifact["bytes"]
        and access["download_exit_code"] == 0,
    )
    require(
        "public-artifact-identical-release",
        access["manifest_matches_original"]
        and access["report_matches_original"]
        and access["accepted_sha256"] == release.digest((reports / "real-release/accepted.json").read_bytes())
        and access["release_report_sha256"]
        == release.digest((reports / "real-release/release-report.json").read_bytes())
        and access["export_exit_code"] == access["release_exit_code"] == 0,
    )
    result = {"passed": True, "checks": checks, "n_checks": len(checks), "video_recorded": False}
    (reports / "submission-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("SUBMISSION_AUDIT_PASSED", len(checks))


if __name__ == "__main__":
    main()
