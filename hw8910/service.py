"""FastAPI service: readiness after loading/warmup, fixed loaded identity, bounded generation."""

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from backends import load_backend


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1, max_length=8192)
    max_new_tokens: StrictInt = 32

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("prompt must contain text")
        return value

    @field_validator("max_new_tokens")
    @classmethod
    def fixed_generation(cls, value):
        if value != 32:
            raise ValueError("This evaluated runtime supports exactly max_new_tokens=32")
        return value


async def finish_thread_before_cancelling(function, *args):
    """A cancelled request still owns its compute slot until the synchronous work stops."""
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # Cancelling an asyncio waiter cannot stop an already running Python/GPU thread.
        # Keep the caller's lock and queue slot even if cancellation is requested again.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()  # Retrieve a possible backend exception before propagating cancellation.
        raise


class ServiceState:
    def __init__(self, root, warmup_seconds):
        self.root = root
        self.warmup_seconds = warmup_seconds
        self.ready = False
        self.backend = None
        self.error = None
        self.lock = asyncio.Lock()
        self.pending = 0

    async def load(self):
        try:
            self.backend = await asyncio.to_thread(load_backend, self.root)
            await asyncio.sleep(self.warmup_seconds)
            await asyncio.to_thread(self.backend.warmup)
            self.ready = True
            print("READY", self.backend.loaded_id, self.backend.runtime_sha256, flush=True)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print("LOAD_FAILED", self.error, flush=True)

    async def generate(self, request):
        if not self.ready:
            raise HTTPException(503, "Model loading/warmup has not completed")
        if self.pending >= 8:
            raise HTTPException(429, "Generation queue is full")
        started = time.perf_counter()
        self.pending += 1
        try:
            async with self.lock:
                queue_seconds = time.perf_counter() - started
                try:
                    batch = await finish_thread_before_cancelling(self.backend.validate, request.prompt)
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
                generation_start = time.perf_counter()
                result = await finish_thread_before_cancelling(
                    self.backend.generate, batch, request.max_new_tokens
                )
                result.update(
                    text=result["answer"],
                    request_id=str(uuid.uuid4()),
                    model_id=self.backend.loaded_id,
                    runtime_sha256=self.backend.runtime_sha256,
                    generation_seconds=time.perf_counter() - generation_start,
                    queue_seconds=queue_seconds,
                    total_seconds=time.perf_counter() - started,
                )
                return result
        finally:
            self.pending -= 1


def create_app(root, warmup_seconds=0):
    state = ServiceState(Path(root), warmup_seconds)

    @asynccontextmanager
    async def lifespan(app):
        worker = asyncio.create_task(state.load())
        yield
        if not worker.done():
            worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    app = FastAPI(title="GitHub issue classifier — HW 8–10", lifespan=lifespan)
    app.state.service = state

    @app.middleware("http")
    async def body_limit(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > 32768):
            return JSONResponse({"detail": "Request body limit: 32768 bytes"}, status_code=413)
        if request.method == "POST":
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 32768:
                    return JSONResponse({"detail": "Request body limit: 32768 bytes"}, status_code=413)
            request._body = bytes(body)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"alive": True}

    @app.get("/ready")
    async def ready():
        return JSONResponse(
            {
                "ready": state.ready,
                "model_id": state.backend.loaded_id if state.ready else None,
                "error": state.error,
            },
            status_code=200 if state.ready else 503,
        )

    @app.get("/debug")
    async def debug():
        # Loopback diagnostic endpoint; no prompts or weights.
        return {
            "loaded_id": state.backend.loaded_id if state.backend else None,
            "calls": state.backend.calls if state.backend else 0,
            "warmup_finished": state.ready,
            "pending": state.pending,
            "compute_slots": 1,
        }

    @app.post("/generate")
    async def generate(body: GenerateRequest):
        return await state.generate(body)

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(".lab"))
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--warmup-seconds", type=float, default=0)
    args = parser.parse_args()
    uvicorn.run(create_app(args.workspace, args.warmup_seconds), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
