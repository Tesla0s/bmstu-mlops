"""Проверка единого файла параметров."""

from pathlib import Path

import yaml


def load_params(path: str = "params.yaml") -> dict:
    p = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    t = p["train"]
    for k in ("epochs", "batch_size", "grad_accum", "eval_every", "eval_batch_size"):
        if not isinstance(t[k], int) or t[k] < 1:
            raise ValueError(f"train.{k} должен быть положительным целым")
    if t.get("max_steps") is not None and t["max_steps"] < 1:
        raise ValueError("max_steps должен быть положительным либо null")
    if t["lr"] <= 0 or not 0 <= t["warmup_ratio"] < 1:
        raise ValueError("Неверные lr или warmup_ratio")
    if p["model"]["enable_thinking"] is not False:
        raise ValueError("Режим рассуждений должен совпадать с ДЗ4: false")
    if len(p["compare"]["prompts"]) != 5:
        raise ValueError("Нужно пять заранее зафиксированных вопросов")
    return p
