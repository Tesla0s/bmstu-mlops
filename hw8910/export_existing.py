"""Export the immutable HW67 run; this step does not train or select another model."""

import argparse
import os
from pathlib import Path
import shutil
import release
from binding import HW67, TOKENIZER_FILES, expected_runtime
from artifacts import verify_files


def base_snapshot(runtime):
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=runtime["name"], revision=runtime["revision"], allow_patterns=["*.json", "*.safetensors"]
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "accepted.json").exists():
        raise ValueError("Refusing to reinitialize a published workspace")
    spec = release.read(HW67 / "reports/evaluation/candidate-model.json")
    if args.source_dir:
        source = args.source_dir
    else:
        os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5006")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
        import mlflow
        import yaml

        client = mlflow.MlflowClient()
        version = client.get_model_version(spec["name"], spec["version"])
        if version.run_id != spec["run_id"]:
            raise ValueError("Registry version no longer points at evaluated run")
        model = Path(client.download_artifacts(spec["run_id"], "model"))
        info = yaml.safe_load((model / "MLmodel").read_text())
        source = model / info["flavors"]["python_function"]["artifacts"]["adapter"]["path"]
    target = root / "artifacts/adapter"
    shutil.copytree(source, target, dirs_exist_ok=True)
    file_hashes = {
        p.relative_to(target).as_posix(): release.digest(p.read_bytes())
        for p in sorted(target.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    cfg = release.read(HW67 / "evaluation/params.json")
    training = release.read(target / "manifest.json")
    snapshot = base_snapshot(training["model"])
    base_files = {
        p.name: release.digest(p.read_bytes())
        for p in sorted(snapshot.iterdir())
        if p.name in ("config.json", "generation_config.json") or p.name.endswith(".safetensors")
    }
    if not any(x.endswith(".safetensors") for x in base_files):
        raise ValueError("Base weights unavailable")
    runtime = expected_runtime(target, cfg, base_files)
    report = release.read(HW67 / "reports/evaluation/candidate-eval.json")
    manifest = {
        "schema_version": 1,
        "backend": "transformers",
        "model_id": f"{spec['name']}/{spec['version']}",
        "artifact_uri": f"runs:/{spec['run_id']}/model",
        "artifact_dir": "artifacts/adapter",
        "file_hashes": file_hashes,
        "model_sha256": release.fingerprint(file_hashes),
        "tokenizer_files": list(TOKENIZER_FILES),
        "protocol_sha256": report["protocol_sha256"],
        "golden_sha256": report["golden_sha256"],
        "runtime": runtime,
        "runtime_sha256": release.fingerprint(runtime),
    }
    origin = {
        "training_run_id": spec["run_id"],
        "evaluation_run_id": report["evaluation_run_id"],
        "registry_uri": spec["uri"],
        "artifact_uri": manifest["artifact_uri"],
        "model_sha256": manifest["model_sha256"],
        "file_hashes": file_hashes,
        "base_files": base_files,
        "reuse": "Same HW67 adapter, exact base revision, tokenizer, prompt, greedy decoding, dtype and MPS. No retraining.",
    }
    origin_path = Path(__file__).parent / "reports/model-origin.json"
    if origin_path.exists() and release.read(origin_path) != origin:
        raise ValueError("Downloaded artifact differs from the pinned evaluation origin")
    release.write(origin_path, origin)
    release.write(root / "manifest-real.json", manifest)
    release.write(
        root / "config.json",
        {
            "format": "golden-jsonl",
            "tokenizer_revision": runtime["base_revision"],
            "template_sha256": runtime["template_sha256"],
            "max_length": 2048,
            "code_revision": "schema-prepare-v1",
        },
    )
    shutil.copyfile(HW67 / cfg["golden"], root / "input.txt")
    verify_files(manifest, root)
    print("EXPORTED_EXISTING_RUN", spec["run_id"], manifest["model_sha256"])


if __name__ == "__main__":
    main()
