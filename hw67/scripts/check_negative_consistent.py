"""Отказ на согласованной копии метрик, пересчитанной из испорченных ответов."""

import copy

from evaluation.cli import current_reports
from evaluation.contract import fingerprint, read_json, read_rows, write_json
from evaluation.solution import quality_gate, score


def main():
    baseline, candidate = current_reports()
    config = read_json("evaluation/params.json")
    golden = read_rows(config["golden"])
    behavioral = read_rows(config["behavioral"])
    original = read_json("metrics/candidate-outputs.json")
    broken_outputs = copy.deepcopy(original)
    golden_ids = {r["id"] for r in golden}
    for row in broken_outputs:
        if row["id"] in golden_ids:
            row.update(raw="INVALID", error=None)
    by_id = {r["id"]: r for r in broken_outputs}
    broken = copy.deepcopy(candidate)
    broken.update(score(golden, [by_id[r["id"]] for r in golden]))
    broken["behavioral"] = score(behavioral, [by_id[r["id"]] for r in behavioral])
    broken["outputs_sha256"] = fingerprint(broken_outputs)
    try:
        quality_gate(baseline, broken, config["critical_slices"])
    except ValueError as exc:
        result = dict(
            rejected=True,
            reason=str(exc),
            metrics_recomputed_from_raw=True,
            original_outputs_unchanged=original == read_json("metrics/candidate-outputs.json"),
            note="Согласованные counts; отказ именно из-за падения accuracy.",
        )
    else:
        raise ValueError("Испорченная копия прошла проверку")
    if not result["original_outputs_unchanged"]:
        raise ValueError("Исходные ответы изменились")
    write_json("reports/negative-consistent/outputs.json", broken_outputs)
    write_json("reports/negative-consistent/evaluation.json", broken)
    write_json("reports/negative-consistent/gate.json", result)
    print(result)


if __name__ == "__main__":
    main()
