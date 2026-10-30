"""Compatible CLI for the original incidents; final runs execute the Prefect flow."""

import argparse
import os
from pathlib import Path
from release import canonical, digest, read, write


def initialize(root):
    if root.exists():
        raise FileExistsError("Рабочий каталог уже существует. Выберите новый --workspace.")
    root.mkdir(parents=True)
    runtime = {
        "base_revision": "fixture-only",
        "tokenizer_sha256": "a" * 64,
        "template_sha256": "b" * 64,
        "generation": {"max_new_tokens": 32, "do_sample": False},
        "dtype": "float32",
    }
    for version in ["A", "B"]:
        artifact = root / f"model-{version}.txt"
        artifact.write_text(f"STUB ONLY — version {version}\n")
        files = {artifact.name: digest(artifact.read_bytes())}
        m = {
            "schema_version": 1,
            "model_id": f"fixture-{version}",
            "artifact_uri": artifact.name,
            "model_sha256": digest(canonical(files)),
            "protocol_sha256": "d" * 64,
            "golden_sha256": "e" * 64,
            "runtime_sha256": digest(canonical(runtime)),
            "runtime": runtime,
            "file_hashes": files,
        }
        report = {
            k: m[k]
            for k in ["model_id", "model_sha256", "protocol_sha256", "golden_sha256", "runtime_sha256"]
        }
        report.update(
            status="complete",
            passed=True,
            evaluation_run_id=f"fixture-eval-{version}",
            metrics={"accuracy": 0.8, "format_rate": 1.0},
            synthetic=True,
        )
        write(root / f"manifest-{version}.json", m)
        write(root / f"report-{version}.json", report)
    write(root / "accepted.json", read(root / "manifest-A.json"))
    report = read(root / "report-B.json")
    report["status"] = "running"
    report["passed"] = "false"
    write(root / "report-incomplete.json", report)
    report = read(root / "report-B.json")
    report["passed"] = False
    write(root / "report-reject.json", report)
    write(
        root / "config.json",
        {
            "tokenizer_revision": "r1",
            "template_sha256": "b" * 64,
            "max_length": 128,
            "code_revision": "prepare-v1",
        },
    )
    (root / "input.txt").write_text("Первый пример\nВторой пример\n")
    from publication import remember

    remember(root, read(root / "manifest-A.json"), read(root / "report-A.json"))
    print("Создан синтетический стенд:", root)


def run(
    root, report_name, fail_after_prepare=False, manifest_name="manifest-B.json", fail_after_pointer=False
):
    from pipeline import release_flow

    return release_flow(
        str(Path(root).resolve()), report_name, fail_after_prepare, manifest_name, fail_after_pointer
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, default=Path(".lab"))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    r = sub.add_parser("run")
    r.add_argument("--report", default="report-B.json")
    r.add_argument("--manifest", default="manifest-B.json")
    r.add_argument("--fail-after-prepare", action="store_true")
    r.add_argument("--fail-after-pointer", action="store_true")
    rb = sub.add_parser("rollback")
    rb.add_argument("release_id")
    args = p.parse_args()
    os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4208/api")
    if args.command == "init":
        initialize(args.workspace)
    elif args.command == "rollback":
        from publication import rollback

        print(rollback(args.workspace, args.release_id))
    else:
        run(args.workspace, args.report, args.fail_after_prepare, args.manifest, args.fail_after_pointer)


if __name__ == "__main__":
    main()
