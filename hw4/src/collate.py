"""Дополнение слева только до длины самого длинного примера в батче."""

import torch

LABEL_PAD_ID = -100


class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int, padding_side: str = "left") -> None:
        if padding_side != "left":
            raise ValueError("Для данного проекта используется только левый паддинг")
        if not isinstance(pad_token_id, int):
            raise ValueError("pad_token_id должен быть целым числом")
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Пустой батч")
        keys = ("input_ids", "attention_mask", "labels")
        for f in features:
            lengths = [len(f[key]) for key in keys]
            if not lengths[0] or len(set(lengths)) != 1:
                raise ValueError("Длины input_ids, attention_mask и labels должны совпадать")
            if any(x != 1 for x in f["attention_mask"]):
                raise ValueError("Коллатор ожидает примеры без предварительного паддинга")
        width = max(len(f["input_ids"]) for f in features)
        values = (self.pad_token_id, 0, LABEL_PAD_ID)
        return {
            key: torch.tensor(
                [[value] * (width - len(f[key])) + list(f[key]) for f in features],
                dtype=torch.long,
            )
            for key, value in zip(keys, values, strict=True)
        }
