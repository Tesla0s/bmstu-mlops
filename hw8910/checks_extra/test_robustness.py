"""Additional audit of the implemented service; separate from seven prototype regressions."""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx
from fastapi import HTTPException
import release
import publication
from lab import initialize
from service import GenerateRequest, ServiceState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_support import Audit  # noqa: E402


class BlockingBackend:
    def __init__(self):
        self.loaded_id = "AUDIT ONLY"
        self.runtime_sha256 = "a" * 64
        self.calls = 0
        self.active = 0
        self.maximum = 0
        self.entered = threading.Event()
        self.released = threading.Event()
        self.counter_lock = threading.Lock()

    def validate(self, prompt):
        return prompt

    def generate(self, prompt, limit):
        with self.counter_lock:
            self.calls += 1
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        self.entered.set()
        try:
            if not self.released.wait(5):
                raise TimeoutError("Audit did not release the blocking backend")
            return {"answer": "AUDIT ONLY", "output_tokens": 1}
        finally:
            with self.counter_lock:
                self.active -= 1


class ServingLimits(unittest.IsolatedAsyncioTestCase):
    def state(self):
        state = ServiceState(Path("."), 0)
        state.backend = BlockingBackend()
        state.ready = True
        return state

    async def test_cancelled_generation_keeps_compute_slot_until_thread_finishes(self):
        state = self.state()
        first = asyncio.create_task(state.generate(GenerateRequest(prompt="first")))
        second = None
        try:
            self.assertTrue(await asyncio.to_thread(state.backend.entered.wait, 2))
            first.cancel()
            await asyncio.sleep(0.05)
            first.cancel()  # repeated cancellation must not interrupt cleanup
            second = asyncio.create_task(state.generate(GenerateRequest(prompt="second")))
            await asyncio.sleep(0.1)
            print("CANCELLATION_MAX_PARALLEL", state.backend.maximum, flush=True)
            self.assertEqual(
                state.backend.maximum, 1, "Cancellation released the slot while work was running"
            )
            self.assertEqual(state.pending, 2)
        finally:
            state.backend.released.set()
            results = await asyncio.gather(*([first, second] if second else [first]), return_exceptions=True)
        self.assertIsInstance(results[0], asyncio.CancelledError)
        self.assertEqual(results[1]["model_id"], "AUDIT ONLY")
        self.assertEqual(state.pending, 0)
        self.assertEqual(state.backend.calls, 2)

    async def test_ninth_request_rejected_without_starting_more_work(self):
        state = self.state()
        tasks = [asyncio.create_task(state.generate(GenerateRequest(prompt=str(i)))) for i in range(8)]
        try:
            self.assertTrue(await asyncio.to_thread(state.backend.entered.wait, 2))
            await asyncio.sleep(0.05)
            self.assertEqual(state.pending, 8)
            with self.assertRaises(HTTPException) as rejected:
                await state.generate(GenerateRequest(prompt="ninth"))
            self.assertEqual(rejected.exception.status_code, 429)
            self.assertEqual(state.backend.calls, 1)
            self.assertEqual(state.backend.maximum, 1)
        finally:
            state.backend.released.set()
            await asyncio.gather(*tasks)
        self.assertEqual(state.backend.calls, 8)
        self.assertEqual(state.backend.maximum, 1)
        self.assertEqual(state.pending, 0)


class FilesAndHTTP(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "fixture"
        initialize(self.root)
        self.audit = Audit(Path(self.tmp.name) / "http")

    def tearDown(self):
        self.audit.close()
        self.tmp.cleanup()

    def test_http_body_limit_with_and_without_content_length(self):
        url, _ = self.audit.server(self.root)
        self.audit.ready(url)
        before = self.audit.http(url, "/debug")[1]["calls"]
        with httpx.Client(timeout=10) as client:
            for body in (b"x" * 32769, iter([b"x" * 20000, b"x" * 20000])):
                response = client.post(url + "/generate", content=body)
                self.assertEqual(response.status_code, 413)
            self.assertEqual(
                client.post(url + "/generate", json={"prompt": "ok", "extra": 1}).status_code, 422
            )
        self.assertEqual(self.audit.http(url, "/debug")[1]["calls"], before)
        self.assertEqual(self.audit.http(url, "/health")[0], 200)
        self.assertEqual(self.audit.http(url, "/generate", {"prompt": "valid"})[0], 200)

    def test_http_corrupt_artifact_never_becomes_ready(self):
        (self.root / "model-A.txt").write_text("Changed accepted artifact")
        url, _ = self.audit.server(self.root)
        deadline = time.monotonic() + 5
        while True:
            status, response = self.audit.http(url, "/ready")
            if response["error"] or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        self.assertEqual(status, 503)
        self.assertIn("Artifact file changed", response["error"])
        self.assertEqual(self.audit.http(url, "/health")[0], 200)
        self.assertEqual(self.audit.http(url, "/generate", {"prompt": "valid"})[0], 503)
        self.assertEqual(self.audit.http(url, "/debug")[1]["calls"], 0)

    def test_rejected_and_incomplete_reports_are_preserved_byte_for_byte(self):
        from pipeline import evaluate_release

        before = (self.root / "accepted.json").read_bytes()
        nan_report = release.read(self.root / "report-B.json")
        nan_report["metrics"]["accuracy"] = float("nan")
        (self.root / "report-nan.json").write_text(json.dumps(nan_report))
        (self.root / "report-list.json").write_text("[]\n")
        names = ("report-incomplete.json", "report-reject.json", "report-nan.json", "report-list.json")
        for name in names:
            with self.assertRaises((ValueError, AttributeError)):
                evaluate_release.fn(
                    str(self.root), {"manifest_uri": str(self.root / "manifest-B.json")}, name, {}
                )
        decisions = list((self.root / "rejections").glob("*/decision.json"))
        self.assertEqual(len(decisions), len(names))
        received = []
        for path in decisions:
            record = release.read(path)
            self.assertIs(record["passed"], False)
            self.assertEqual(record["accepted_sha256"], release.digest(before))
            for name in ("input.txt", "config.json"):
                self.assertEqual((path.parent / name).read_bytes(), (self.root / name).read_bytes())
            payload = (path.parent / "submitted-report.json").read_bytes()
            received.append(payload)
            self.assertEqual(record["input_sha256"]["submitted-report.json"], release.digest(payload))
        self.assertCountEqual(
            received,
            [(self.root / name).read_bytes() for name in names],
        )
        self.assertEqual((self.root / "accepted.json").read_bytes(), before)

    def test_recovery_after_journal_written_but_history_failed(self):
        manifest = release.read(self.root / "manifest-B.json")
        report = release.read(self.root / "report-B.json")
        with patch("publication.remember", side_effect=OSError("TEST history write failed")):
            with self.assertRaises(OSError):
                publication.publish_once(self.root, manifest, report, {"input_sha256": "fixture"})
        self.assertTrue((self.root / "publication-intent.json").exists())
        self.assertEqual(len(publication.events(self.root)), 1)
        before = (self.root / "accepted.json").read_bytes()
        result = publication.publish_once(self.root, manifest, report, {"input_sha256": "fixture"})
        self.assertFalse(result["created"])
        self.assertEqual(len(publication.events(self.root)), 1)
        self.assertFalse((self.root / "publication-intent.json").exists())
        self.assertEqual((self.root / "accepted.json").read_bytes(), before)
        self.assertTrue((self.root / "accepted-history" / (result["release_id"] + ".json")).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
