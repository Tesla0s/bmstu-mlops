"""Чтение и проверка единого файла параметров."""

from pathlib import Path
import math

import yaml


def load_params(path: str = "params.yaml") -> dict:
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise ValueError("Конфигурация должна содержать разделы model, generate, bench.")
    for section in ("model", "generate", "bench"):
        if not isinstance(params.get(section), dict):
            raise ValueError(f"Отсутствует раздел {section}.")

    model, generation, bench = (params[k] for k in ("model", "generate", "bench"))
    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise ValueError("model.name должен содержать имя модели.")
    if model.get("device") not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("model.device: допустимы auto, cpu, cuda, mps.")
    if model.get("dtype") not in ("float32", "float16", "bfloat16"):
        raise ValueError("model.dtype: допустимы float32, float16, bfloat16.")
    for section, key in (
        (generation, "max_new_tokens"),
        (bench, "warmup_runs"),
        (bench, "measure_runs"),
    ):
        if type(section.get(key)) is not int or section[key] < 1:
            raise ValueError(f"{key} должен быть целым числом не меньше 1.")
    temperature = generation.get("temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature должна быть конечным неотрицательным числом.")
    if type(generation.get("seed")) is not int or not 0 <= generation["seed"] < 2**32:
        raise ValueError("seed должен быть целым числом от 0 до 2**32 - 1.")
    if type(generation.get("enable_thinking")) is not bool:
        raise ValueError("enable_thinking должен быть true или false.")
    if not isinstance(bench.get("prompt"), str) or not bench["prompt"].strip():
        raise ValueError("bench.prompt должен содержать непустой вопрос.")
    return params
