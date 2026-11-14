"""Open-loop load generator, run inside each of four Kubernetes client Pods."""
import asyncio
import json
import socket
import sys
import time
import httpx


async def run(config):
    origin = time.monotonic() + config["start_wall"] - time.time()
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=httpx.Timeout(config.get("timeout", 30.0)), limits=limits) as client:
        async def one(item):
            planned = origin + item["offset"]
            await asyncio.sleep(max(0, planned-time.monotonic()))
            sent = time.monotonic()
            row = {"experiment": config["experiment"], "client": socket.gethostname(),
                   "client_ip": socket.gethostbyname(socket.gethostname()),
                   "request_id": item["body"]["request_id"], "planned_mono": planned,
                   "sent_mono": sent, "planned_wall": config["start_wall"]+item["offset"],
                   "sent_wall": time.time(), "send_lag_seconds": sent-planned,
                   "first_mono": None, "end_mono": None, "http_status": None,
                   "outcome": None, "chunks": 0, "inter_chunk_seconds": [],
                   "pod": None, "model_id": None, "artifact_sha256": None,
                   "queue_seconds": None, "cache_hit": None, "backend_mode": None,
                   "terminal": None, "expected_chunks": item["body"].get("max_chunks",5),
                   "request": item["body"], "keep_alive": False}
            previous = None
            try:
                async with client.stream("POST", config["url"] + "/generate", json=item["body"],
                                         headers={"Connection": "close"}) as response:
                    row["http_status"] = response.status_code
                    row["pod"] = response.headers.get("x-pod")
                    row["model_id"] = response.headers.get("x-model-id")
                    row["backend_mode"] = response.headers.get("x-backend-mode")
                    if response.status_code != 200:
                        row["outcome"] = "rejected" if response.status_code == 429 else "http_error"
                        row["error_body"] = (await response.aread()).decode()[:1500]
                    else:
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            data = json.loads(line)
                            now = time.monotonic()
                            if data.get("done"):
                                row["terminal"] = data
                                continue
                            if data["request_id"] != row["request_id"]:
                                raise ValueError("response request ID mismatch")
                            if row["first_mono"] is None:
                                row["first_mono"] = now
                            if previous is not None:
                                row["inter_chunk_seconds"].append(now-previous)
                            previous = now
                            row["chunks"] += 1
                            for key in ("pod", "model_id", "artifact_sha256", "backend_mode",
                                        "queue_seconds", "cache_hit"):
                                row[key] = data[key]
                            row["server_observed_client_ip"] = data["client_ip"]
                        row["outcome"] = ("completed" if row["terminal"] and
                                          row["chunks"] == row["expected_chunks"] else "broken")
            except httpx.TimeoutException as exc:
                row["outcome"] = "expired"
                row["error"] = type(exc).__name__
            except (httpx.HTTPError, ValueError) as exc:
                row["outcome"] = "broken" if row["http_status"] == 200 else "connection_error"
                row["error"] = str(exc)
            row["end_mono"] = time.monotonic()
            row["end_wall"] = time.time()
            row["total_seconds"] = row["end_mono"]-sent
            row["ttfc_seconds"] = None if row["first_mono"] is None else row["first_mono"]-sent
            print(json.dumps(row, ensure_ascii=False), flush=True)
        await asyncio.gather(*(one(item) for item in config["schedule"]))


if __name__ == "__main__":
    asyncio.run(run(json.load(sys.stdin)))
