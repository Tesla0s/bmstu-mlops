"""Один раз загрузить закреплённую базу и подготовить проверку преподавателя."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from src.config import load_params


def main():
    p = load_params()
    home = Path(os.environ.get("HF_HOME", ".cache/huggingface"))
    cache = home / "hub"
    path = Path(
        snapshot_download(
            p["model"]["name"],
            revision=p["model"]["revision"],
            cache_dir=cache,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        )
    )
    if path.name != p["model"]["revision"]:
        raise ValueError("Получена другая ревизия")
    ref = path.parent.parent / "refs/main"
    # Оригинальный check.sh загружает базу без revision. В выделенном кэше
    # его offline-вызов должен видеть ту же базу, что и обучение.
    if ref.exists() and ref.read_text().strip() != path.name:
        raise ValueError("В этом кэше main указывает на другую версию; выберите отдельный HF_HOME")
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(path.name)
    print(f"Базовая модель и offline-ссылка проверены: {path}")


if __name__ == "__main__":
    main()
