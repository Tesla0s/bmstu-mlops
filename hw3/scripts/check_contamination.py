#!/usr/bin/env python3
"""Independent fail-closed check of all three pairs of partitions."""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_params
from src.contamination import is_clean, report
from src.io_utils import write_json
from src.schema import iter_examples


def main():
    params = load_params()
    paths, nd = params["paths"], params["clean"]["near_dup"]
    if not nd["enabled"]:
        raise SystemExit("near-duplicate removal must be enabled")
    if nd["threshold"] != params["contamination"]["threshold"]:
        raise SystemExit("clean and contamination thresholds must be identical")
    buckets = {name: list(iter_examples(paths[name])) for name in ("train", "val", "test")}
    results = {}
    for left, right in combinations(buckets, 2):
        rep = report(
            buckets[left],
            buckets[right],
            nd["shingle_words"],
            nd["num_perm"],
            nd["threshold"],
        )
        results[f"{left}_{right}"] = rep
        print(
            f"{left}/{right}: пересечение по id {rep['id_overlap']}, по тексту {rep['text_overlap']}, по группам {rep['group_overlap']}; near-dup {rep['near_dup_pairs']}"
        )
    clean_ids = [ex.id for ex in iter_examples(paths["clean"])]
    partition_ids = [ex.id for rows in buckets.values() for ex in rows]
    complete = len(partition_ids) == len(clean_ids) and set(partition_ids) == set(clean_ids)
    passed = complete and all(is_clean(rep) for rep in results.values()) and all(buckets.values())
    write_json(
        paths["metrics_contamination"],
        {
            "version": params["collect"]["version"],
            "threshold": nd["threshold"],
            "pairs": results,
            "complete_partition": complete,
            "passed": passed,
        },
    )
    print(
        "Контаминации нет; все очищенные записи учтены."
        if passed
        else "КОНТАМИНАЦИЯ или неполное разбиение"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
