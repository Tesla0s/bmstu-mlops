"""Параметры обработки; ошибки конфигурации обнаруживаются до записи артефактов."""

from pathlib import Path

import yaml


def load_params(path: str = "params.yaml") -> dict:
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = params["tokenize"]
    if not isinstance(cfg["max_seq_len"], int) or cfg["max_seq_len"] <= 0:
        raise ValueError("max_seq_len должен быть положительным целым числом")
    if not 0 <= cfg["truncated_warn_ratio"] <= 1:
        raise ValueError("truncated_warn_ratio должен лежать от 0 до 1")
    if cfg["padding_side"] != "left":
        raise ValueError("tokenize.padding_side должен быть left")
    if params["packing"]["enabled"]:
        raise ValueError("Packing выключен: в этом проекте нет маски между упакованными примерами")
    if params["train_estimate"]["epochs"] <= 0:
        raise ValueError("Число эпох должно быть положительным")
    return params
