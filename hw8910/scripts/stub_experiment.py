"""Explicit STUB ONLY experiment: failures, content cache, A→B→A and controlled rollback."""

import argparse
import json
from pathlib import Path
import shutil
from audit_support import Audit, release


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports/stub-experiment"))
    args = p.parse_args()
    root = args.workspace.resolve()
    audit = Audit(args.output)
    prefix = ["lab.py", "--workspace", str(root)]
    try:
        audit.run("01-init", prefix + ["init"])
        original_a = release.digest((root / "accepted.json").read_bytes())
        a_id = next((root / "accepted-history").glob("*.json")).stem
        url, a = audit.server(root, warmup=2)
        assert audit.http(url, "/health")[0] == 200
        assert audit.http(url, "/ready")[0] == 503
        assert audit.http(url, "/generate", {"prompt": "hello"})[0] == 503
        audit.ready(url)
        assert audit.http(url, "/generate", {"prompt": "hello"}, "loaded-A")[1]["model_id"] == "fixture-A"
        first = audit.run("02-fail-after-prepare", prefix + ["run", "--fail-after-prepare"], "nonzero")
        assert "PREPARED" in first.stdout
        assert release.digest((root / "accepted.json").read_bytes()) == original_a
        crashed = audit.run("03-crash-after-pointer", prefix + ["run", "--fail-after-pointer"], "nonzero")
        assert "CACHE_HIT" in crashed.stdout
        assert release.read(root / "accepted.json")["model_id"] == "fixture-B"
        assert (root / "publication-intent.json").exists()
        restored = audit.run("04-recover-journal", prefix + ["run"])
        assert "RECOVERED_JOURNAL" in restored.stdout
        audit.run("05-repeat-new-process", prefix + ["run"])
        events = [json.loads(x) for x in (root / "publications.jsonl").read_text().splitlines()]
        assert len(events) == 1
        audit.record({"step": "single-B-publication", "events": events})
        assert (
            audit.http(url, "/generate", {"prompt": "hello"}, "pointer-B-memory-A")[1]["model_id"]
            == "fixture-A"
        )
        a.terminate()
        a.wait(timeout=15)
        url, b = audit.server(root)
        audit.ready(url)
        assert audit.http(url, "/generate", {"prompt": "hello"}, "restarted-B")[1]["model_id"] == "fixture-B"
        old = release.digest((root / "accepted.json").read_bytes())
        for name in ("report-reject.json", "report-incomplete.json", "report-A.json"):
            audit.run("refuse-" + name, prefix + ["run", "--report", name], "nonzero")
            assert release.digest((root / "accepted.json").read_bytes()) == old
            audit.record({"step": "refusal-unchanged-" + name, "before_sha256": old, "after_sha256": old})
        with (root / "input.txt").open("a") as f:
            f.write("Изменённое содержимое по тому же пути\n")
        changed = audit.run("06-changed-input", prefix + ["run"])
        assert "PREPARED" in changed.stdout and "CACHE_HIT" not in changed.stdout
        audit.run("07-controlled-rollback", prefix + ["rollback", a_id])
        assert release.digest((root / "accepted.json").read_bytes()) == original_a
        assert (
            audit.http(url, "/generate", {"prompt": "hello"}, "rollback-pointer-A-memory-B")[1]["model_id"]
            == "fixture-B"
        )
        b.terminate()
        b.wait(timeout=15)
        url, _ = audit.server(root)
        audit.ready(url)
        assert (
            audit.http(url, "/generate", {"prompt": "hello"}, "rollback-restarted-A")[1]["model_id"]
            == "fixture-A"
        )
        for name in ("attempts.jsonl", "publications.jsonl", "rollback-last.json"):
            shutil.copyfile(root / name, audit.output / name)
        audit.record({"step": "complete", "passed": True, "kind": "STUB ONLY"})
    finally:
        audit.close()


if __name__ == "__main__":
    main()
