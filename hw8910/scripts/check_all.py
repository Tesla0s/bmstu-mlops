"""One command, no model dependencies or weights: contracts + actual process/HTTP regressions."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="hw8910-check-") as tmp:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(
            os.environ,
            PREFECT_API_URL=f"http://127.0.0.1:{port}/api",
            PREFECT_HOME=str(Path(tmp) / "prefect"),
            PREFECT_SERVER_ANALYTICS_ENABLED="false",
            PYTHONUNBUFFERED="1",
        )
        with (Path(tmp) / "server.log").open("w") as log:
            server = subprocess.Popen(
                [
                    str(Path(sys.executable).parent / "prefect"),
                    "server",
                    "start",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError((Path(tmp) / "server.log").read_text())
                    try:
                        with urllib.request.urlopen(
                            env["PREFECT_API_URL"] + "/health", timeout=1
                        ) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(0.2)
                else:
                    raise TimeoutError("Prefect server did not start")
                codes = []
                for directory in ("tests", "checks"):
                    process = subprocess.run(
                        [sys.executable, "-m", "unittest", "discover", "-s", directory, "-v"],
                        cwd=ROOT,
                        env=env,
                        timeout=240,
                    )
                    codes.append(process.returncode)
                print("FAST_CHECK_EXIT_CODES", codes, flush=True)
                return int(any(codes))
            finally:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
