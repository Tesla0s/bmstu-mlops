"""Utilities for recorded experiments; assertions describe observed behavior."""

import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import release  # noqa: E402


class Audit:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.servers = []

    def record(self, value):
        self.records.append(value)
        release.write(self.output / "evidence.json", self.records)

    def run(self, label, args, expected=0):
        p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=240)
        (self.output / (label + ".log")).write_text(p.stdout + p.stderr)
        self.record(
            {
                "step": label,
                "command": [sys.executable, *args],
                "exit_code": p.returncode,
                "expected_exit": expected,
                "log": label + ".log",
            }
        )
        if expected == "nonzero":
            assert p.returncode != 0, label
        else:
            assert p.returncode == expected, (label, p.stdout, p.stderr)
        return p

    def http(self, url, route, body=None, label=None):
        req = urllib.request.Request(
            url + route,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            status, answer = response.status, json.loads(response.read())
        self.record(
            {"step": label or route, "request": route, "body": body, "status": status, "response": answer}
        )
        return status, answer

    def server(self, workspace, warmup=0):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        log = (self.output / f"server-{len(self.servers)}.log").open("w")
        p = subprocess.Popen(
            [
                sys.executable,
                "service.py",
                "--workspace",
                str(workspace),
                "--port",
                str(port),
                "--warmup-seconds",
                str(warmup),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        self.servers.append((p, log))
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            assert p.poll() is None, "Server stopped"
            try:
                with urllib.request.urlopen(url + "/health", timeout=1) as r:
                    if r.status == 200:
                        return url, p
            except OSError:
                time.sleep(0.1)
        raise TimeoutError("Server health")

    def ready(self, url):
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            status, answer = self.http(url, "/ready")
            if status == 200:
                return answer
            if answer.get("error"):
                raise ValueError(answer["error"])
            time.sleep(0.5)
        raise TimeoutError("Model readiness")

    def close(self):
        for process, log in self.servers:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            log.close()
