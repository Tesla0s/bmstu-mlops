"""All 16 real HTTP attempts are saved; failed requests are never dropped."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request


def request(url, prompt):
    started = time.perf_counter()
    try:
        req = urllib.request.Request(
            url + "/generate",
            data=json.dumps({"prompt": prompt, "max_new_tokens": 32}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            status, body = response.status, json.loads(response.read())
        return {
            "status": status,
            "error": None if status == 200 else json.dumps(body),
            "client_seconds": time.perf_counter() - started,
            **{
                k: body.get(k)
                for k in (
                    "request_id",
                    "generation_seconds",
                    "queue_seconds",
                    "total_seconds",
                    "model_id",
                    "runtime_sha256",
                    "output_tokens",
                    "answer",
                )
            },
        }
    except Exception as exc:
        return {
            "status": None,
            "error": f"{type(exc).__name__}: {exc}",
            "client_seconds": time.perf_counter() - started,
            **{
                k: None
                for k in (
                    "request_id",
                    "generation_seconds",
                    "queue_seconds",
                    "total_seconds",
                    "model_id",
                    "runtime_sha256",
                    "output_tokens",
                    "answer",
                )
            },
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8098")
    p.add_argument("--prompts", type=Path, default=Path("benchmark-prompts.json"))
    p.add_argument("--output", type=Path, default=Path("reports/benchmark"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.prompts.read_text())
    assert len(prompts) == 8
    with urllib.request.urlopen(args.url + "/ready") as r:
        assert r.status == 200
    warmup = request(args.url, prompts[0]["prompt"])
    (args.output / "warmup.json").write_text(json.dumps(warmup, indent=2))
    assert warmup["status"] == 200
    rows, summary = [], []
    with (args.output / "attempts.jsonl").open("w") as log:
        for concurrency in (1, 2):
            started = time.perf_counter()

            def one(item):
                return {
                    "id": item["id"],
                    "concurrency": concurrency,
                    "max_new_tokens": 32,
                    **request(args.url, item["prompt"]),
                }

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                batch = list(pool.map(one, prompts))
            elapsed = time.perf_counter() - started
            for row in batch:
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
            rows.extend(batch)
            good = [r for r in batch if r["status"] == 200 and r["error"] is None]
            bad = [r for r in batch if r not in good]
            summary.append(
                {
                    "concurrency": concurrency,
                    "n": len(batch),
                    "successes": len(good),
                    "errors": len(bad),
                    "batch_seconds": elapsed,
                    "successful_requests_per_second": len(good) / elapsed,
                    "successful_mean_seconds": statistics.mean(r["client_seconds"] for r in good)
                    if good
                    else None,
                    "successful_max_seconds": max((r["client_seconds"] for r in good), default=None),
                    "error_mean_seconds": statistics.mean(r["client_seconds"] for r in bad) if bad else None,
                    "error_max_seconds": max((r["client_seconds"] for r in bad), default=None),
                    "mean_queue_seconds": statistics.mean(r["queue_seconds"] for r in good) if good else None,
                    "mean_generation_seconds": statistics.mean(r["generation_seconds"] for r in good)
                    if good
                    else None,
                    "output_tokens": sum(r["output_tokens"] for r in good),
                }
            )
    with (args.output / "attempts.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if any(r["errors"] for r in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
