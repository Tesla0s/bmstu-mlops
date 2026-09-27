"""Копирование опубликованной версии ДЗ 3 с проверкой контрольных сумм."""

import hashlib
import shutil
from pathlib import Path

from src.config import load_params


def main():
    params = load_params()
    for name in ("train", "val"):
        dest = Path(params["data"][f"{name}_jsonl"])
        source = Path(params["provenance"]["hw3_directory"]) / "data" / f"{name}.jsonl"
        expected = params["provenance"]["sha256"][name]
        if not source.is_file():
            raise SystemExit(f"Нет {source}: сначала восстановите hw3-v2 через DVC в ДЗ 3")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"{source} не совпадает с закреплённой версией hw3-v2")
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
            raise SystemExit(
                f"{dest} уже содержит другой файл: автоматическая перезапись запрещена"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(source, dest)
        print(f"{name}: файл ДЗ 3 сохранён без изменений; SHA-256 {expected}")


if __name__ == "__main__":
    main()
