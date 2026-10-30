"""Real accepted model: load, warmup, strict validation, answer and restart identity."""

import argparse
from pathlib import Path
from audit_support import Audit, release


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports/real-api"))
    args = p.parse_args()
    audit = Audit(args.output)
    manifest = release.read(args.workspace / "accepted.json")
    try:
        for restart in (0, 1):
            url, server = audit.server(args.workspace, warmup=2)
            assert audit.http(url, "/health")[0] == 200
            assert audit.http(url, "/ready")[0] == 503
            assert audit.http(url, "/generate", {"prompt": "hello"})[0] == 503
            ready = audit.ready(url)
            assert ready["model_id"] == manifest["model_id"]
            before = audit.http(url, "/debug")[1]["calls"]
            for body in (
                {"prompt": ""},
                {"prompt": "hello", "max_new_tokens": 10000},
                {"prompt": "hello", "max_new_tokens": True},
                {"prompt": "x" * 8193},
                {"prompt": "x " * 3500},
            ):
                status, _ = audit.http(url, "/generate", body)
                assert 400 <= status < 500
            assert audit.http(url, "/debug")[1]["calls"] == before
            status, answer = audit.http(
                url,
                "/generate",
                {
                    "prompt": "Title: Editor crashes on save\nDescription: Pressing Ctrl+S closes the editor and loses my changes.",
                    "max_new_tokens": 32,
                },
                f"real-answer-restart-{restart}",
            )
            assert status == 200
            assert answer["model_id"] == manifest["model_id"]
            assert answer["runtime_sha256"] == manifest["runtime_sha256"]
            assert "STUB ONLY" not in answer["answer"]
            assert 0 < answer["output_tokens"] <= 32
            server.terminate()
            server.wait(timeout=30)
        audit.record({"step": "complete", "passed": True, "kind": "REAL TRANSFORMERS + PEFT / MPS"})
    finally:
        audit.close()


if __name__ == "__main__":
    main()
