"""HW8–10 API extended with a CPU simulator, streaming, probes and bounded admission."""
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

POD = os.environ.get("POD_NAME", "local")
WARMUP = float(os.environ.get("WARMUP_SECONDS", "20"))
CAPACITY = int(os.environ.get("QUEUE_CAPACITY", "8"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "16"))
HIT_CPU = float(os.environ.get("HIT_CPU_SECONDS", ".02"))
MISS_CPU = float(os.environ.get("MISS_CPU_SECONDS", ".16"))
CHUNK_CPU = float(os.environ.get("CHUNK_CPU_SECONDS", ".025"))
CHUNK_DELAY = float(os.environ.get("CHUNK_DELAY_SECONDS", ".1"))
BUCKETS = (.01,.025,.05,.1,.2,.4,.6,.8,1,1.5,2,3,4,6,8,12,20,30,60)
running = Gauge("hw11_running", "Requests currently holding compute slots")
waiting = Gauge("hw11_waiting", "Admitted requests waiting for a compute slot")
ready_metric = Gauge("hw11_backend_ready", "Loaded and warmed, and not draining")
started = Counter("hw11_started", "All valid requests received")
results = Counter("hw11_results", "Terminal request outcome", ["outcome"])
cache_metric = Counter("hw11_cache", "Explicit simulated prefix cache", ["result"])
ttfc_hist = Histogram("hw11_ttfc_seconds", "Server time to first chunk, completed requests only", buckets=BUCKETS)
total_hist = Histogram("hw11_total_seconds", "Completed stream duration", buckets=BUCKETS)
queue_hist = Histogram("hw11_queue_seconds", "Completed stream queue time", buckets=BUCKETS)
inter_hist = Histogram("hw11_inter_chunk_seconds", "Intervals of completed streams", buckets=BUCKETS)
for outcome in ("completed", "rejected", "unready", "broken"):
    results.labels(outcome)
for outcome in ("hit", "miss"):
    cache_metric.labels(outcome)


def log(event, **data):
    print(json.dumps({"wall": time.time(), "event": event, "pod": POD,
                      "backend_mode": "simulated", "model_id": state.model_id,
                      "artifact_sha256": state.sha, **data}, ensure_ascii=False), flush=True)


def cpu_work(seconds):
    """Spend measured CPU time, not wall-clock sleep; OpenSSL work releases the GIL."""
    target = time.thread_time() + seconds
    while time.thread_time() < target:
        hashlib.pbkdf2_hmac("sha256", b"hw11-compute", b"fixed-salt", 1000)


async def finish_thread_before_cancelling(function, *args):
    # Same ownership rule as hw8910/service.py: cancellation cannot release a live worker's slot.
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1, max_length=8192)
    history: list[str] = Field(default_factory=list, max_length=16)
    prefix: str = Field(default="github-issues", max_length=1024)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=128)
    session_id: str = Field(default="anonymous", max_length=128)
    max_chunks: int = Field(default=5, ge=1, le=100)
    cpu_scale: float = Field(default=1.0, ge=.1, le=20)
    chunk_delay: float | None = Field(default=None, ge=0, le=2)

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("prompt must contain text")
        return value

    @field_validator("history")
    @classmethod
    def bounded_history(cls, value):
        if sum(len(x) for x in value) > 8192:
            raise ValueError("history exceeds 8192 characters")
        return value


class State:
    def __init__(self):
        self.loaded = False
        self.draining = False
        self.model_id = None
        self.sha = None
        self.error = None
        self.pending = 0
        self.cache = OrderedDict()
        self.slots = asyncio.Semaphore(CONCURRENCY)

    async def load(self):
        try:
            raw = Path(os.environ.get("MODEL_PATH", "/model/model.json")).read_bytes()
            self.sha = hashlib.sha256(raw).hexdigest()
            expected = os.environ["MODEL_SHA256"]
            if self.sha != expected:
                raise ValueError("artifact SHA256 mismatch")
            model = json.loads(raw)
            if model["backend_mode"] != "simulated":
                raise ValueError("This backend supports simulated artifacts only")
            self.model_id = model["version"]
            log("load_started", warmup_seconds=WARMUP, source=os.environ.get("MODEL_SOURCE"))
            await asyncio.sleep(WARMUP)
            if os.environ.get("FAIL_WARMUP") == "true":
                raise RuntimeError("Intentional failed warmup for rollback experiment")
            await finish_thread_before_cancelling(cpu_work, .01)
            self.loaded = True
            ready_metric.set(1)
            log("warmup_finished")
        except Exception as exc:
            self.error = str(exc)
            log("load_failed", error=self.error)


state = State()


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(state.load())
    yield
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


app = FastAPI(title="GitHub issue classifier — HW11 CPU experiment", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"alive": True, "backend_mode": "simulated", "pod": POD}


@app.get("/startup")
async def startup():
    return JSONResponse({"loaded": state.loaded, "error": state.error},
                        status_code=200 if state.loaded else 503)


@app.get("/ready")
async def ready():
    ok = state.loaded and not state.draining
    return JSONResponse({"ready": ok, "model_id": state.model_id, "pod": POD,
                         "artifact_sha256": state.sha, "backend_mode": "simulated",
                         "error": state.error}, status_code=200 if ok else 503)


@app.post("/drain")
async def drain():
    # Local lab lifecycle hook; not exposed through an ingress.
    state.draining = True
    ready_metric.set(0)
    log("draining", pending=state.pending)
    return {"draining": True}


@app.post("/reset-cache")
async def reset_cache():
    if state.pending:
        return JSONResponse({"error": "busy"}, status_code=409)
    state.cache.clear()
    log("cache_reset")
    return {"size": 0}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/debug")
async def debug():
    return {"pod": POD, "model_id": state.model_id, "artifact_sha256": state.sha,
            "backend_mode": "simulated", "loaded": state.loaded, "pending": state.pending,
            "cache_entries": len(state.cache), "compute_slots": CONCURRENCY}


@app.post("/generate")
async def generate(body: GenerateRequest, request: Request):
    started.inc()
    received = time.monotonic()
    identity = {"request_id": body.request_id, "session_id": body.session_id,
                "pod": POD, "model_id": state.model_id, "artifact_sha256": state.sha,
                "backend_mode": "simulated", "client_ip": request.client.host}
    headers = {"X-Pod": POD, "X-Model-Id": state.model_id or "unloaded",
               "X-Request-Id": body.request_id, "X-Backend-Mode": "simulated"}
    log("received", request_id=body.request_id, session_id=body.session_id,
        client_ip=request.client.host, warm=state.loaded, pending=state.pending)
    if not state.loaded:
        results.labels("unready").inc()
        log("unready", request_id=body.request_id)
        return JSONResponse({**identity, "error": "backend unavailable"}, 503, headers=headers)
    if state.pending >= CAPACITY:
        results.labels("rejected").inc()
        log("rejected", request_id=body.request_id)
        return JSONResponse({**identity, "error": "bounded queue is full"}, 429, headers=headers)

    state.pending += 1
    waiting.inc()

    async def stream():
        acquired = False
        complete = False
        first = None
        previous = None
        intervals = []
        count = 0
        try:
            await state.slots.acquire()
            acquired = True
            waiting.dec()
            running.inc()
            queued = time.monotonic() - received
            key = hashlib.sha256(body.prefix.encode()).hexdigest()
            hit = key in state.cache
            cache_metric.labels("hit" if hit else "miss").inc()
            await finish_thread_before_cancelling(cpu_work, (HIT_CPU if hit else MISS_CPU) * body.cpu_scale)
            state.cache[key] = True
            state.cache.move_to_end(key)
            while len(state.cache) > CACHE_SIZE:
                state.cache.popitem(last=False)
            history_sha = hashlib.sha256(json.dumps(body.history, ensure_ascii=False).encode()).hexdigest()
            for index in range(body.max_chunks):
                await finish_thread_before_cancelling(cpu_work, CHUNK_CPU * body.cpu_scale)
                await asyncio.sleep(CHUNK_DELAY if body.chunk_delay is None else body.chunk_delay)
                now = time.monotonic()
                if first is None:
                    first = now
                else:
                    intervals.append(now - previous)
                previous = now
                count += 1
                # Echo/history are explicit simulator evidence, not predictions of the real model.
                part = {**identity, "chunk": index, "queue_seconds": queued, "cache_hit": hit,
                        "history_sha256": history_sha, "text": f"simulated fragment {index + 1}"}
                yield json.dumps(part, ensure_ascii=False).encode() + b"\n"
            yield json.dumps({**identity, "done": True, "chunks": count,
                              "history_sha256": history_sha, "history_echo": body.history,
                              "prompt_echo": body.prompt, "answer": "simulated:question"}).encode() + b"\n"
            complete = True
            duration = time.monotonic() - received
            ttfc_hist.observe(first - received)
            total_hist.observe(duration)
            queue_hist.observe(queued)
            for interval in intervals:
                inter_hist.observe(interval)
            results.labels("completed").inc()
            log("completed", request_id=body.request_id, session_id=body.session_id,
                queue_seconds=queued, ttfc_seconds=first-received, total_seconds=duration,
                chunks=count, cache_hit=hit, history_sha256=history_sha)
        finally:
            if not complete:
                results.labels("broken").inc()
                log("broken", request_id=body.request_id, chunks=count)
            state.pending -= 1
            if acquired:
                running.dec()
                state.slots.release()
            else:
                waiting.dec()

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers=headers)
