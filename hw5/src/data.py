"""Чтение проверенных файлов ДЗ4 и динамическое дополнение слева."""

import random
from itertools import islice
from pathlib import Path

import torch

LABEL_PAD_ID = -100


def load_split(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"Нет {p}: скопируйте выход tokenize из ДЗ4")
    blob = torch.load(p, map_location="cpu", weights_only=True)
    if not blob.get("examples"):
        raise ValueError(f"{p}: нет примеров")
    seen = set()
    for e in blob["examples"]:
        if e["id"] in seen:
            raise ValueError(f"{p}: повторный id {e['id']}")
        seen.add(e["id"])
        if not len(e["input_ids"]) == len(e["labels"]) == len(e["attention_mask"]):
            raise ValueError(f"{p}: разные длины полей")
        if len(e["input_ids"]) < 2 or any(x != 1 for x in e["attention_mask"]):
            raise ValueError(f"{p}: ожидаются непустые последовательности без дополнения")
        if not any(x != LABEL_PAD_ID for x in e["labels"][1:]):
            raise ValueError(f"{p}: нет обучаемых следующих токенов")
        if any(y not in (LABEL_PAD_ID, x) for x, y in zip(e["input_ids"], e["labels"])):
            raise ValueError(f"{p}: метки не совпадают с исходными токенами")
    return blob


def pad_batch(features: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    if not features:
        raise ValueError("Пустой батч")
    width = max(len(f["input_ids"]) for f in features)
    return {
        key: torch.tensor(
            [[value] * (width - len(f[key])) + list(f[key]) for f in features],
            dtype=torch.long,
        )
        for key, value in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100))
    }


def batches(examples, batch_size: int, pad_id: int, shuffle: bool, seed: int):
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), batch_size):
        yield pad_batch([examples[j] for j in order[i : i + batch_size]], pad_id)


def accumulation_groups(iterator, count: int):
    """Последняя неполная группа тоже получает шаг оптимизатора."""
    if count < 1:
        raise ValueError("grad_accum должен быть положительным")
    iterator = iter(iterator)
    while group := list(islice(iterator, count)):
        yield group
