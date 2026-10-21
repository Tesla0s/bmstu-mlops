"""Проверка сохранённых ответов без MLflow, модели и GPU."""

import copy
import json
from pathlib import Path

from evaluation.contract import fingerprint, read_json, read_rows, validate_golden
from evaluation.solution import quality_gate, score


def main():
    cfg = read_json("evaluation/params.json")
    rows = read_rows(cfg["golden"])
    behavior = read_rows(cfg["behavioral"])
    train = read_rows(cfg["training_index"])
    proof = validate_golden(rows, train, behavior)
    baseline_spec = read_json("evaluation/baseline.json")
    protocol = dict(
        config=cfg,
        golden=rows,
        behavioral=behavior,
        training_index=train,
        evaluation_code={p.name: p.read_text() for p in sorted(Path("evaluation").glob("*.py"))},
    )
    reports = {}
    for role in ("baseline", "candidate"):
        report = read_json(f"reports/evaluation/{role}-eval.json")
        outputs = read_json(f"reports/evaluation/{role}-outputs.json")
        by_id = {o["id"]: o for o in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {r["id"] for r in rows + behavior}:
            raise ValueError("Неполный набор ответов")
        expected_hashes = dict(
            protocol_sha256=fingerprint(protocol),
            golden_sha256=fingerprint(rows),
            baseline_spec_sha256=fingerprint(baseline_spec),
            outputs_sha256=fingerprint(outputs),
        )
        for key, value in expected_hashes.items():
            if report.get(key) != value:
                raise ValueError(f"{role}: не совпал {key}")
        result = score(rows, [by_id[r["id"]] for r in rows])
        for key in ("overall", "slices", "per_item"):
            if result[key] != report[key]:
                raise ValueError(f"{role}: неверные {key}")
        if score(behavior, [by_id[r["id"]] for r in behavior]) != report["behavioral"]:
            raise ValueError("Неверный результат поведения")
        reports[role] = report
    a = {r["id"]: r for r in reports["baseline"]["per_item"]}
    b = {r["id"]: r for r in reports["candidate"]["per_item"]}
    diff = dict(
        improved=[i for i in a if not a[i]["correct"] and b[i]["correct"]],
        regressed=[i for i in a if a[i]["correct"] and not b[i]["correct"]],
    )
    if diff != read_json("reports/evaluation/eval-diff.json"):
        raise ValueError("Неверный список исправленных/потерянных ответов")
    saved = read_json("reports/evaluation/quality-gate.json")
    try:
        actual = quality_gate(reports["baseline"], reports["candidate"], cfg["critical_slices"])
    except ValueError as exc:
        actual = dict(passed=False, reason=str(exc))
    if actual != saved:
        raise ValueError("Сохранённое решение не соответствует ответам")
    broken = copy.deepcopy(reports["candidate"])
    broken.update(score(rows, [dict(id=r["id"], raw="INVALID", error=None) for r in rows]))
    try:
        quality_gate(reports["baseline"], broken, cfg["critical_slices"])
    except ValueError:
        pass
    else:
        raise ValueError("Испорченные ответы прошли проверку")
    print(
        json.dumps(
            dict(
                verified=True,
                leakage=proof,
                baseline=reports["baseline"]["overall"],
                candidate=reports["candidate"]["overall"],
                decision=actual,
                negative_rejected=True,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
