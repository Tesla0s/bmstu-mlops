"""Раздельное измерение загрузки, скорости генерации и пиковой памяти."""

from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import torch

from src.config import load_params
from src.model import generate_tokens, load_model, prepare_inputs, synchronize


def peak_rss_mb() -> float:
    """Максимум RSS за жизнь процесса в МиБ, включая загрузку и прогрев."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2) if sys.platform == "darwin" else peak / 1024


def benchmark(params: dict) -> dict:
    if params["bench"]["warmup_runs"] < 1 or params["bench"]["measure_runs"] < 1:
        raise ValueError("Нужны хотя бы один прогрев и одно измерение.")

    t0 = time.perf_counter()
    tokenizer, model = load_model(params)
    synchronize(model.device)
    load_time = time.perf_counter() - t0

    inputs = prepare_inputs(tokenizer, model, params, params["bench"]["prompt"])
    for _ in range(params["bench"]["warmup_runs"]):
        generate_tokens(model, params, inputs)
        synchronize(model.device)

    runs = []
    for _ in range(params["bench"]["measure_runs"]):
        synchronize(model.device)
        started = time.perf_counter()
        new_tokens = generate_tokens(model, params, inputs)
        synchronize(model.device)
        elapsed = time.perf_counter() - started
        n_tokens = len(new_tokens)
        if elapsed <= 0 or n_tokens == 0:
            raise RuntimeError("Не удалось получить положительное время и число токенов.")
        runs.append({
            "elapsed_sec": elapsed,
            "new_tokens": n_tokens,
            "tokens_per_sec": n_tokens / elapsed,
        })

    # Декодирование и сбор описания окружения не входят в скорость генерации.
    answer = tokenizer.decode(new_tokens.cpu(), skip_special_tokens=True)
    speeds = [run["tokens_per_sec"] for run in runs]
    report = {
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": params["model"]["name"],
        "model_revision": getattr(model.config, "_commit_hash", None),
        "device": str(model.device),
        "dtype": str(model.dtype).removeprefix("torch."),
        "parameters": sum(p.numel() for p in model.parameters()),
        "config": params,
        "input_tokens": inputs["input_ids"].shape[1],
        "load_time_sec": load_time,
        "tokens_per_sec": statistics.median(speeds),
        "tokens_per_sec_all": speeds,
        "runs": runs,
        "answer": answer,
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in (
                "torch", "transformers", "accelerate", "numpy", "pyyaml"
            )},
            "os": platform.platform(),
            "torch_num_threads": torch.get_num_threads(),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "0"),
        },
        "peak_rss_mb": peak_rss_mb(),
        "memory_unit": "MiB (2**20 bytes)",
    }
    return report


def main() -> None:
    report = benchmark(load_params())
    output = json.dumps(report, ensure_ascii=False, indent=2)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/bench.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
