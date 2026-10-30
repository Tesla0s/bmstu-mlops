"""Recompute the HW67 gate from all saved answers, then bind it to exact serving files."""

from pathlib import Path
import sys
import json
import tomllib

import release
from artifacts import artifact_root, verify_files

HW67 = Path(__file__).resolve().parents[1] / "hw67"
sys.path.insert(0, str(HW67))
from evaluation.contract import fingerprint as evaluation_fingerprint, validate_golden  # noqa: E402
from evaluation.solution import quality_gate, score  # noqa: E402

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)
LIBRARIES = ("torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors", "huggingface-hub")


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def expected_runtime(adapter, cfg, base_files):
    training = release.read(adapter / "manifest.json")
    model = training["model"]
    packages = tomllib.loads((HW67 / "uv.lock").read_text())["package"]
    versions = {p["name"]: p["version"] for p in packages if p["name"] in LIBRARIES}
    if set(versions) != set(LIBRARIES):
        raise ValueError("Incomplete historical library versions")
    return {
        "base_model": model["name"],
        "base_revision": model["revision"],
        "base_files": base_files,
        "tokenizer_revision": model["revision"],
        "tokenizer_sha256": release.fingerprint(
            {x: release.digest((adapter / x).read_bytes()) for x in TOKENIZER_FILES}
        ),
        "template_sha256": release.digest((adapter / "chat_template.jinja").read_bytes()),
        "generation": {k: cfg[k] for k in ("seed", "max_new_tokens", "enable_thinking", "system")}
        | {"do_sample": False, "use_model_defaults": False},
        "dtype": model["dtype"],
        "device": "mps",
        "attention": "eager",
        "libraries": versions,
    }


def verify_saved_evaluation(workspace, manifest, prepared):
    workspace = Path(workspace)
    verify_files(manifest, workspace)
    cfg = release.read(HW67 / "evaluation/params.json")
    golden, behavior, training = [rows(HW67 / cfg[k]) for k in ("golden", "behavioral", "training_index")]
    leakage = validate_golden(golden, training, behavior)
    if prepared["input_sha256"] != release.digest((workspace / "input.txt").read_bytes()):
        raise ValueError("Input changed after preparation")
    if rows(workspace / "input.txt") != golden:
        raise ValueError("Changed evaluation data: run HW67 evaluation again")
    protocol = {
        "config": cfg,
        "golden": golden,
        "behavioral": behavior,
        "training_index": training,
        "evaluation_code": {p.name: p.read_text() for p in sorted((HW67 / "evaluation").glob("*.py"))},
    }
    protocol_hash = evaluation_fingerprint(protocol)
    baseline_spec = release.read(HW67 / "evaluation/baseline.json")
    candidate_spec = release.read(HW67 / "reports/evaluation/candidate-model.json")
    origin = release.read(Path(__file__).parent / "reports/model-origin.json")
    if (
        origin["training_run_id"] != candidate_spec["run_id"]
        or origin["model_sha256"] != manifest["model_sha256"]
    ):
        raise ValueError("Artifact differs from the evaluated MLflow training run")
    if manifest["model_id"] != f"{candidate_spec['name']}/{candidate_spec['version']}":
        raise ValueError("Wrong candidate model_id")
    if manifest["artifact_uri"] != f"runs:/{candidate_spec['run_id']}/model":
        raise ValueError("Wrong immutable artifact URI")
    evaluation_run_id = origin.get("evaluation_run_id")
    if not isinstance(evaluation_run_id, str) or not evaluation_run_id.strip():
        raise ValueError("Missing pinned evaluation run")
    expected = expected_runtime(artifact_root(manifest, workspace), cfg, origin["base_files"])
    if manifest["runtime"] != expected or manifest["runtime_sha256"] != release.fingerprint(expected):
        raise ValueError("Serving runtime differs from HW67 evaluation; new evaluation required")
    if baseline_spec != {
        "kind": "base",
        "name": expected["base_model"],
        "revision": expected["base_revision"],
    }:
        raise ValueError("Base revision differs from evaluated baseline")
    if manifest["protocol_sha256"] != protocol_hash or manifest["golden_sha256"] != evaluation_fingerprint(
        golden
    ):
        raise ValueError("Protocol/golden hash mismatch")
    reports = {}
    sources = [
        HW67 / "evaluation/params.json",
        HW67 / "evaluation/baseline.json",
        HW67 / "reports/evaluation/candidate-model.json",
        HW67 / "uv.lock",
    ]
    sources += sorted((HW67 / "evaluation").glob("*.py"))
    sources += [HW67 / cfg[k] for k in ("golden", "behavioral", "training_index")]
    for role, spec in (("baseline", baseline_spec), ("candidate", candidate_spec)):
        report_path = HW67 / f"reports/evaluation/{role}-eval.json"
        output_path = HW67 / f"reports/evaluation/{role}-outputs.json"
        report, outputs = release.read(report_path), release.read(output_path)
        sources.extend((report_path, output_path))
        if report.get("evaluation_run_id") != evaluation_run_id:
            raise ValueError("Foreign evaluation run: " + role)

        if report["model"] != spec:
            raise ValueError("Evaluation model mismatch: " + role)
        hashes = {
            "protocol_sha256": protocol_hash,
            "golden_sha256": evaluation_fingerprint(golden),
            "baseline_spec_sha256": evaluation_fingerprint(baseline_spec),
            "outputs_sha256": evaluation_fingerprint(outputs),
        }
        if any(report[k] != v for k, v in hashes.items()):
            raise ValueError("Evaluation evidence changed: " + role)
        by_id = {o["id"]: o for o in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {r["id"] for r in golden + behavior}:
            raise ValueError("Missing/duplicate saved answers")
        recalculated = score(golden, [by_id[r["id"]] for r in golden])
        if any(recalculated[k] != report[k] for k in ("overall", "slices", "per_item")):
            raise ValueError("Metrics do not match raw answers")
        if score(behavior, [by_id[r["id"]] for r in behavior]) != report["behavioral"]:
            raise ValueError("Behavioral metrics do not match raw answers")
        reports[role] = report
    decision = quality_gate(reports["baseline"], reports["candidate"], cfg["critical_slices"])
    result = {k: manifest[k] for k in release.BINDINGS}
    result.update(
        status="complete",
        passed=decision["passed"],
        evaluation_run_id=reports["candidate"]["evaluation_run_id"],
        metrics={k: reports["candidate"]["overall"][k] for k in ("accuracy", "format_rate")},
        quality_gate=decision,
        leakage=leakage,
        evaluation_reused=True,
        source_sha256={str(p.relative_to(HW67)): release.digest(p.read_bytes()) for p in sources},
        source_origin_sha256=release.digest(
            (Path(__file__).parent / "reports/model-origin.json").read_bytes()
        ),
    )
    release.validate_release(manifest, result)
    return result
