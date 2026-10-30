"""Four real Prefect tasks. Cached preparation persists across Python processes."""

import json
import os
from pathlib import Path
import time
import uuid

os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4208/api")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run
import release
from artifacts import verify_files
from publication import publish_once


def preparation_config(root):
    cfg = release.read(root / "config.json")
    return {**cfg, "implementation_sha256": release.digest(Path(__file__).read_bytes())}


def preparation_key(context, parameters):
    root = Path(parameters["workspace"])
    # Workspace isolates result URIs; the public cache_key itself is path-independent.
    return release.fingerprint(
        {
            "workspace": str(root.resolve()),
            "input": release.cache_key(root / "input.txt", preparation_config(root)),
        }
    )


@task(persist_result=True, cache_key_fn=preparation_key, retries=0)
def prepare(workspace):
    root = Path(workspace)
    cfg = preparation_config(root)
    raw = (root / "input.txt").read_bytes()
    lines = [x for x in raw.decode("utf-8").splitlines() if x.strip()]
    if not lines:
        raise ValueError("Empty preparation input")
    if cfg.get("format") == "golden-jsonl":
        rows = [json.loads(x) for x in lines]
        ids = set()
        for row in rows:
            if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
                raise ValueError("Duplicate/missing example ID")
            ids.add(row["id"])
            if not isinstance(row.get("input"), str) or not row["input"].strip():
                raise ValueError("Empty example input")
            if row.get("expected") not in row.get("choices", []):
                raise ValueError("Expected label is outside choices")
    else:
        rows = [{"id": str(i), "input": x} for i, x in enumerate(lines)]
    key = release.cache_key(root / "input.txt", cfg)
    output = root / "prepared" / (key + ".json")
    release.write(output, {"rows": rows, "config": cfg})
    print("PREPARED", key, "rows", len(rows), flush=True)
    return {
        "uri": str(output),
        "sha256": release.digest(output.read_bytes()),
        "input_sha256": release.digest(raw),
        "cache_key": key,
        "rows": len(rows),
    }


@task(cache_policy=NO_CACHE, retries=0)
def get_existing_artifact(workspace, manifest_name, prepared):
    root = Path(workspace)
    if release.digest(Path(prepared["uri"]).read_bytes()) != prepared["sha256"]:
        raise ValueError("Prepared artifact changed")
    manifest = release.read(root / manifest_name)
    description = verify_files(manifest, root)
    return {**description, "manifest_uri": str(root / manifest_name)}


def save_rejection(root, manifest_path, report_name, reason, real_model):
    """Keep the actual rejected inputs, including malformed JSON values, as raw files."""
    run_id = str(flow_run.id or uuid.uuid4())
    directory = root / "rejections" / run_id
    sources = {
        "manifest.json": Path(manifest_path),
        "input.txt": root / "input.txt",
        "config.json": root / "config.json",
    }
    if real_model:
        from binding import HW67

        sources.update(
            {
                role + "-eval.json": HW67 / f"reports/evaluation/{role}-eval.json"
                for role in ("baseline", "candidate")
            }
        )
    else:
        sources["submitted-report.json"] = root / report_name
    hashes = {}
    for name, source in sources.items():
        if source.is_file():
            content = source.read_bytes()
            release.atomic_bytes(directory / name, content)
            hashes[name] = release.digest(content)
    pointer = root / "accepted.json"
    release.write(
        directory / "decision.json",
        {
            "status": "rejected",
            "passed": False,
            "flow_run_id": run_id,
            "reason": reason,
            "input_sha256": hashes,
            "accepted_sha256": release.digest(pointer.read_bytes() if pointer.exists() else b""),
        },
    )
    print("REJECTED_REPORT", directory / "decision.json", flush=True)


@task(cache_policy=NO_CACHE, retries=0)
def evaluate_release(workspace, artifact, report_name, prepared):
    root = Path(workspace)
    manifest = release.read(artifact["manifest_uri"])
    real_model = manifest.get("backend") == "transformers"
    try:
        if real_model:
            from binding import verify_saved_evaluation

            report = verify_saved_evaluation(root, manifest, prepared)
            release.write(root / report_name, report)
        else:
            report = release.read(root / report_name)
            if (
                report.get("synthetic") is not True
                or manifest["runtime"].get("base_revision") != "fixture-only"
            ):
                raise ValueError("Synthetic gate is restricted to explicit STUB ONLY fixtures")
        release.validate_release(manifest, report)
    except Exception as exc:
        # Preserve invalid/missing structure as well as a failed quality decision, then re-raise.
        save_rejection(root, artifact["manifest_uri"], report_name, str(exc), real_model)
        raise
    # Content-addressed decisions cannot be overwritten by a rejected later attempt.
    decision_path = root / "decisions" / (release.fingerprint(report) + ".json")
    release.write(decision_path, report)
    return {
        "manifest_uri": artifact["manifest_uri"],
        "report_uri": str(decision_path),
        "decision_sha256": release.fingerprint(report),
        "passed": True,
    }


@task(cache_policy=NO_CACHE, retries=0)
def publish_release(workspace, decision, prepared, fail_after_pointer):
    root = Path(workspace)
    return publish_once(
        root,
        release.read(decision["manifest_uri"]),
        release.read(decision["report_uri"]),
        prepared,
        fail_after_pointer,
    )


def append_attempt(root, row):
    path = root / "attempts.jsonl"
    previous = path.read_bytes() if path.exists() else b""
    release.atomic_bytes(path, previous + release.canonical(row) + b"\n")


@flow(name="hw8910-release", log_prints=False, retries=0)
def release_flow(
    workspace,
    report_name="report-B.json",
    fail_after_prepare=False,
    manifest_name="manifest-B.json",
    fail_after_pointer=False,
):
    root = Path(workspace).resolve()
    attempt = {
        "attempt_id": str(uuid.uuid4()),
        "flow_run_id": str(flow_run.id),
        "started_at": time.time(),
        "tasks": {},
        "status": "running",
    }
    pointer = root / "accepted.json"
    attempt["accepted_before"] = release.digest(pointer.read_bytes() if pointer.exists() else b"")
    try:
        future = prepare.with_options(result_storage=root / "prefect-results").submit(str(root))
        prepared = future.result()
        future.wait()
        attempt["tasks"]["prepare"] = future.state.name
        if (
            not Path(prepared["uri"]).exists()
            or release.digest(Path(prepared["uri"]).read_bytes()) != prepared["sha256"]
        ):
            future = prepare.with_options(result_storage=root / "prefect-results", refresh_cache=True).submit(
                str(root)
            )
            prepared = future.result()
            future.wait()
            attempt["tasks"]["prepare"] = future.state.name
        if attempt["tasks"]["prepare"] == "Cached":
            print("CACHE_HIT", prepared["cache_key"], flush=True)
        attempt["prepared"] = prepared
        if fail_after_prepare:
            raise RuntimeError("Injected failure after successful preparation")
        artifact = get_existing_artifact(str(root), manifest_name, prepared)
        attempt["tasks"]["get_existing_artifact"] = "Completed"
        decision = evaluate_release(str(root), artifact, report_name, prepared)
        attempt["tasks"]["evaluate_release"] = "Completed"
        result = publish_release(str(root), decision, prepared, fail_after_pointer)
        attempt["tasks"]["publish_release"] = "Completed"
        attempt.update(status="complete", publication=result)
        return result
    except Exception as exc:
        attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        attempt["accepted_after"] = release.digest(pointer.read_bytes() if pointer.exists() else b"")
        attempt["finished_at"] = time.time()
        append_attempt(root, attempt)
