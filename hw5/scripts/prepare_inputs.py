"""Копировать только подтверждённые тензоры ДЗ4, не меняя существующие данные."""

import hashlib
import shutil
from pathlib import Path

from src.config import load_params


def main():
    p = load_params()
    for split, target in p["data"].items():
        source = Path("../hw4/data/tokenized") / f"{split}.pt"
        expected = p["provenance"]["tokenized_sha256"][split]
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Изменился источник ДЗ4: {source}")
        dest = Path(target)
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Отказываюсь заменять другой набор: {dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        print(f"{split}: исходный файл ДЗ4 подтверждён, SHA-256 {expected}")


if __name__ == "__main__":
    main()
