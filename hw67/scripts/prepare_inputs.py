"""Подключение неизменённых тензоров из собственного ДЗ 4."""

from pathlib import Path
import shutil

from src.common import digest

EXPECTED = {
    "train": "ff6c2a91d69ae6a90c04a2395d2d4e804dd8b92ffe816626e19bb2a19a4a3c96",
    "val": "892713373d4a012388cbec518554cf2145b458e0affd4032037e3dde5819c879",
}


def main():
    for split, expected in EXPECTED.items():
        src = Path("../hw4/data/tokenized") / (split + ".pt")
        dst = Path("data/tokenized") / src.name
        if not src.is_file() or digest(src) != expected:
            raise ValueError(f"Восстановите исходный артефакт ДЗ 4: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and digest(dst) != expected:
            raise ValueError(f"Файл {dst} содержит другую версию")
        if not dst.exists():
            shutil.copyfile(src, dst)
        print(f"{split}: SHA-256 совпал с ДЗ 4")


if __name__ == "__main__":
    main()
