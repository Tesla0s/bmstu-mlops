"""Выбор устройства, воспроизводимость и наблюдаемый расход памяти."""

import os
import random
import threading

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import psutil
import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"Неизвестный тип весов: {name}")
    return DTYPES[name]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def allocated_bytes(device: torch.device) -> int:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    return psutil.Process().memory_info().rss


def memory_metric(device: torch.device) -> str:
    return {
        "mps": "torch.mps.driver_allocated_memory; sampled every 0.1 s",
        "cuda": "torch.cuda.memory_allocated; sampled every 0.1 s",
    }.get(device.type, "process RSS; sampled every 0.1 s")


class MemoryMonitor:
    """Наблюдаемый максимум, включая оценку; не обещает поймать каждый всплеск."""

    def __init__(self, device):
        self.device = device
        self.peak = allocated_bytes(device)
        self.rss_peak = psutil.Process().memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        self.peak = max(self.peak, allocated_bytes(self.device))
        self.rss_peak = max(self.rss_peak, psutil.Process().memory_info().rss)

    def _run(self):
        while not self.stop_event.wait(0.1):
            self.sample()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
