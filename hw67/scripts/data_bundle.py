"""Публичный переносимый архив только данных проверки и исходных текстов DVC."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import yaml

STAGES = ("eval.dvc", "origin.dvc", "review-selection.json.dvc")
DEFAULT_OUTPUT = Path("../../hw6+7/workspace/submission/hw67-evaluation-data.zip")
URL = "https://github.com/Tesla0s/bmstu-mlops/releases/download/hw67-data-v1/hw67-evaluation-data.zip"


def objects():
    hashes = set()
    for stage in STAGES:
        for out in yaml.safe_load(Path(stage).read_text())["outs"]:
            value = out["md5"]
            hashes.add(value)
            if value.endswith(".dir"):
                path = Path(".dvc/cache/files/md5") / value[:2] / value[2:]
                hashes.update(item["md5"] for item in json.loads(path.read_text()))
    return hashes


def pack(output):
    hashes = objects()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for value in sorted(hashes):
            path = Path(".dvc/cache/files/md5") / value[:2] / value[2:]
            data = path.read_bytes()
            if hashlib.md5(data).hexdigest() != value.removesuffix(".dir"):
                raise ValueError(f"Повреждён DVC-объект {value}")
            archive.writestr(f"files/md5/{value[:2]}/{value[2:]}", data)
    manifest = dict(
        url=URL,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        bytes=output.stat().st_size,
        objects=sorted(hashes),
        stages=list(STAGES),
        contents="golden, behavioral, полный train-index, исходные train/val JSONL, разбор меток; без тензоров и весов",
    )
    Path("data-bundle.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def fetch():
    manifest = json.loads(Path("data-bundle.json").read_text())
    with urllib.request.urlopen(manifest["url"], timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError("Контрольная сумма публичного архива не совпала")
    wanted = {f"files/md5/{h[:2]}/{h[2:]}": h for h in manifest["objects"]}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if set(archive.namelist()) != set(wanted) or len(archive.namelist()) != len(wanted):
            raise ValueError("Архив содержит неожиданные файлы")
        for member, value in wanted.items():
            blob = archive.read(member)
            if hashlib.md5(blob).hexdigest() != value.removesuffix(".dir"):
                raise ValueError(f"Повреждён объект {value}")
            dest = Path(".dvc/cache") / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
    import subprocess

    subprocess.run(["dvc", "checkout", *manifest["stages"]], check=True)
    print("Данные проверки восстановлены; SHA-256 и все DVC-объекты проверены.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pack", "fetch"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pack(args.output) if args.command == "pack" else fetch()


if __name__ == "__main__":
    main()
