"""Same real HTTP/process tests for the original prototype and final implementation."""

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

TARGET = Path(os.environ.get("HW8910_TARGET", Path(__file__).resolve().parents[1])).resolve()
EVIDENCE = Path(os.environ.get("HW8910_EVIDENCE", TARGET / "reports" / "integration"))
EVIDENCE.mkdir(parents=True, exist_ok=True)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "case"
        self.events = []
        self.servers = []
        self.cli("init")

    def tearDown(self):
        for p, log in self.servers:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
            log.close()
        (EVIDENCE / (self._testMethodName + ".json")).write_text(
            json.dumps(self.events, ensure_ascii=False, indent=2) + "\n"
        )
        self.tmp.cleanup()

    def process(self, args, ok=True):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        p = subprocess.run(
            [sys.executable, *args], cwd=TARGET, env=env, capture_output=True, text=True, timeout=180
        )
        self.events.append(
            {"command": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        )
        if ok:
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p

    def cli(self, *args, ok=True):
        return self.process(["lab.py", "--workspace", str(self.root), *args], ok)

    def sha(self):
        return hashlib.sha256((self.root / "accepted.json").read_bytes()).hexdigest()

    def start_server(self, warmup=0):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        log = (Path(self.tmp.name) / f"server-{port}.log").open("w")
        p = subprocess.Popen(
            [
                sys.executable,
                "service.py",
                "--workspace",
                str(self.root),
                "--port",
                str(port),
                "--warmup-seconds",
                str(warmup),
            ],
            cwd=TARGET,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        self.servers.append((p, log))
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            self.assertIsNone(p.poll(), "Server stopped; inspect " + str(log.name))
            try:
                if self.http(url, "/health", record=False)[0] == 200:
                    return url
            except OSError:
                pass
            time.sleep(0.05)
        self.fail("Health did not become reachable")

    def wait_ready(self, url):
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if self.http(url, "/ready", record=False)[0] == 200:
                return
            time.sleep(0.05)
        self.fail("Ready timeout")

    def http(self, url, route, data=None, record=True):
        req = urllib.request.Request(
            url + route,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(req, timeout=20)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            status = response.status
            body = json.loads(response.read())
        if record:
            self.events.append({"request": route, "body": data, "status": status, "response": body})
        return status, body

    def test_01_recovery_and_cache_invalidation_in_new_process(self):
        failed = self.cli("run", "--fail-after-prepare", ok=False)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("PREPARED", failed.stdout)
        resumed = self.cli("run")
        self.assertIn("CACHE_HIT", resumed.stdout)
        with (self.root / "input.txt").open("a") as f:
            f.write("Изменённые данные по тому же пути\n")
        changed = self.cli("run")
        self.assertIn("PREPARED", changed.stdout, "Changed bytes reused stale cache")
        self.assertNotIn("CACHE_HIT", changed.stdout)

    def test_02_incomplete_foreign_and_runtime_report_rejected(self):
        before = self.sha()
        incomplete = self.cli("run", "--report", "report-incomplete.json", ok=False)
        self.assertNotEqual(incomplete.returncode, 0, "Incomplete report was published")
        self.assertEqual(before, self.sha())
        foreign = self.cli("run", "--report", "report-A.json", ok=False)
        self.assertNotEqual(foreign.returncode, 0, "Report for A accepted for B")
        self.assertEqual(before, self.sha())
        m = json.loads((self.root / "manifest-B.json").read_text())
        m["runtime"]["dtype"] = "float16"
        (self.root / "manifest-B.json").write_text(json.dumps(m))
        runtime = self.cli("run", ok=False)
        self.assertNotEqual(runtime.returncode, 0, "Stale runtime hash was accepted")
        self.assertEqual(before, self.sha())
        self.events.append({"accepted_sha256_before": before, "accepted_sha256_after": self.sha()})

    def test_03_repeat_publication_and_recovery_after_pointer_commit(self):
        script = """import lab,release,sys
from pathlib import Path
original=release.publish
def fail_after_pointer(*args,**kwargs):
 original(*args,**kwargs)
 raise RuntimeError('TEST crash after pointer, before journal')
release.publish=fail_after_pointer
lab.run(Path(sys.argv[1]),'report-B.json')
"""
        crash = self.process(["-c", script, str(self.root)], ok=False)
        self.assertNotEqual(crash.returncode, 0)
        self.assertIn("TEST crash after pointer", crash.stderr)
        self.assertEqual(json.loads((self.root / "accepted.json").read_text())["model_id"], "fixture-B")
        self.cli("run")
        before = self.sha()
        self.cli("run")
        events = [json.loads(x) for x in (self.root / "publications.jsonl").read_text().splitlines()]
        self.assertEqual(
            sum(e["model_id"] == "fixture-B" for e in events),
            1,
            "Repeated process created duplicate publication",
        )
        self.assertEqual(before, self.sha())

    def test_04_http_ready_blocks_generation_before_warmup(self):
        url = self.start_server(warmup=3)
        self.assertEqual(self.http(url, "/health")[0], 200)
        before = self.http(url, "/debug")[1]["calls"]
        self.assertEqual(self.http(url, "/ready")[0], 503, "Ready was green before warmup")
        self.assertEqual(self.http(url, "/generate", {"prompt": "hello"})[0], 503)
        self.assertEqual(self.http(url, "/debug")[1]["calls"], before)
        self.wait_ready(url)
        self.assertTrue(self.http(url, "/debug")[1]["warmup_finished"])
        self.assertEqual(self.http(url, "/generate", {"prompt": "hello"})[0], 200)

    def test_05_http_loaded_identity_restart_and_rollback(self):
        a = self.start_server()
        self.wait_ready(a)
        before = self.http(a, "/generate", {"prompt": "valid"})[1]
        self.assertEqual(before["model_id"], "fixture-A")
        self.cli("run")
        stale = self.http(a, "/generate", {"prompt": "valid"})[1]
        self.assertEqual(stale["model_id"], "fixture-A", "Advertised ID follows disk, not loaded model")
        self.assertIn("fixture-A", stale["answer"])
        b = self.start_server()
        self.wait_ready(b)
        changed = self.http(b, "/generate", {"prompt": "valid"})[1]
        self.assertEqual(changed["model_id"], "fixture-B")
        self.assertIn("fixture-B", changed["answer"])
        script = """import json,sys,release
from pathlib import Path
r=Path(sys.argv[1])
release.publish(json.loads((r/'manifest-A.json').read_text()),json.loads((r/'report-A.json').read_text()),r/'accepted.json')
"""
        self.process(["-c", script, str(self.root)])
        still_b = self.http(b, "/generate", {"prompt": "valid"})[1]
        self.assertEqual(still_b["model_id"], "fixture-B")
        restored = self.start_server()
        self.wait_ready(restored)
        answer = self.http(restored, "/generate", {"prompt": "valid"})[1]
        self.assertEqual(answer["model_id"], "fixture-A")
        self.assertIn("fixture-A", answer["answer"])

    def test_06_http_invalid_requests_do_not_call_backend(self):
        url = self.start_server()
        self.wait_ready(url)
        for request in (
            {"prompt": ""},
            {"prompt": "   "},
            {"prompt": "ok", "max_new_tokens": 0},
            {"prompt": "ok", "max_new_tokens": 100000},
            {"prompt": "ok", "max_new_tokens": "32"},
            {"prompt": "ok", "max_new_tokens": True},
        ):
            before = self.http(url, "/debug")[1]["calls"]
            status, _ = self.http(url, "/generate", request)
            self.assertTrue(400 <= status < 500, f"Invalid input generated: {request!r}")
            self.assertEqual(self.http(url, "/debug")[1]["calls"], before)
        before = self.http(url, "/debug")[1]["calls"]
        status, result = self.http(url, "/generate", {"prompt": "ok", "max_new_tokens": 32})
        self.assertEqual(status, 200)
        self.assertIn("STUB ONLY", result["answer"])
        self.assertEqual(self.http(url, "/debug")[1]["calls"], before + 1)

    def test_07_atomic_replace_failure_in_separate_process(self):
        before = self.sha()
        script = """import json,sys,release
from pathlib import Path
from unittest.mock import patch
r=Path(sys.argv[1])
m=json.loads((r/'manifest-B.json').read_text());q=json.loads((r/'report-B.json').read_text())
with patch('os.replace',side_effect=OSError('TEST disk error')):
 try:release.publish(m,q,r/'accepted.json')
 except OSError:print('EXPECTED_DISK_ERROR');sys.exit(23)
raise AssertionError('Replacement error was bypassed')
"""
        p = self.process(["-c", script, str(self.root)], ok=False)
        self.assertEqual(p.returncode, 23, "Atomic replace was not used")
        self.assertEqual(before, self.sha())
        self.assertFalse(list(self.root.glob("*.tmp")))
        self.events.append({"accepted_sha256_before": before, "accepted_sha256_after": self.sha()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
