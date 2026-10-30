"""One-writer write-ahead intent: recover pointer/journal consistency after a process crash."""

from pathlib import Path
import json
import time
import release
from artifacts import verify_files


def events(root):
    path = Path(root) / "publications.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def remember(root, manifest, report):
    rid = release.fingerprint({"manifest": manifest, "decision": report})
    release.write(root / "accepted-history" / (rid + ".json"), {"manifest": manifest, "report": report})
    return rid


def finish_journal(root, intent):
    rows = events(root)
    if not any(x["release_id"] == intent["release_id"] for x in rows):
        rows.append(intent["event"])
        release.atomic_bytes(
            root / "publications.jsonl", b"".join(release.canonical(x) + b"\n" for x in rows)
        )
    remember(root, intent["manifest"], intent["report"])
    (root / "publication-intent.json").unlink(missing_ok=True)


def recover(root):
    path = root / "publication-intent.json"
    if not path.exists():
        return
    intent = release.read(path)
    manifest, report = intent["manifest"], intent["report"]
    release.validate_release(manifest, report)
    verify_files(manifest, root)
    expected = release.canonical(manifest) + b"\n"
    actual = (root / "accepted.json").read_bytes() if (root / "accepted.json").exists() else b""
    if actual != expected:
        if release.digest(actual) != intent["previous_sha256"]:
            raise ValueError("Pointer diverged from pending transaction")
        release.publish(manifest, report, root / "accepted.json")
    finish_journal(root, intent)
    print("RECOVERED_JOURNAL", intent["release_id"], flush=True)


def publish_once(root, manifest, report, prepared, fail_after_pointer=False):
    root = Path(root)
    release.validate_release(manifest, report)
    verify_files(manifest, root)
    recover(root)
    rid = release.fingerprint({"manifest": manifest, "decision": report})
    old = (root / "accepted.json").read_bytes() if (root / "accepted.json").exists() else b""
    expected = release.canonical(manifest) + b"\n"
    existing = next((x for x in events(root) if x["release_id"] == rid), None)
    if existing and old == expected:
        print("ALREADY_PUBLISHED", manifest["model_id"], rid, flush=True)
        return {"release_id": rid, "created": False, "accepted_sha256": release.digest(old)}
    event = existing or {
        "release_id": rid,
        "model_id": manifest["model_id"],
        "runtime_sha256": manifest["runtime_sha256"],
        "evaluation_run_id": report["evaluation_run_id"],
        "decision_sha256": release.fingerprint(report),
        "prepared": prepared["input_sha256"],
        "published_at": time.time(),
    }
    intent = {
        "release_id": rid,
        "manifest": manifest,
        "report": report,
        "event": event,
        "previous_sha256": release.digest(old),
    }
    release.write(root / "publication-intent.json", intent)
    try:
        release.publish(manifest, report, root / "accepted.json")
    except OSError:
        # A failed atomic replace leaves old bytes intact; remove the uncommitted intent.
        if ((root / "accepted.json").read_bytes() if (root / "accepted.json").exists() else b"") == old:
            (root / "publication-intent.json").unlink(missing_ok=True)
        raise
    if fail_after_pointer:
        raise RuntimeError("Injected crash after pointer, before publication journal")
    finish_journal(root, intent)
    print("PUBLISHED", manifest["model_id"], rid, flush=True)
    return {
        "release_id": rid,
        "created": existing is None,
        "accepted_sha256": release.digest((root / "accepted.json").read_bytes()),
    }


def rollback(root, release_id):
    root = Path(root)
    if not release.DIGEST.fullmatch(release_id):
        raise ValueError("Rollback requires the exact accepted release ID")
    record = release.read(root / "accepted-history" / (release_id + ".json"))
    release.validate_release(record["manifest"], record["report"])
    verify_files(record["manifest"], root)
    before = release.digest((root / "accepted.json").read_bytes())
    result = publish_once(root, record["manifest"], record["report"], {"input_sha256": "rollback"})
    release.write(
        root / "rollback-last.json", {"target_release_id": release_id, "before_sha256": before, **result}
    )
    return result
