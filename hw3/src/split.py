"""Whole issue families, ordered by time within repository and category."""

import json
from collections import Counter, defaultdict

from src.config import load_params
from src.io_utils import write_json, write_jsonl
from src.schema import iter_examples
from src.textnorm import normalize_group


def assign_groups(examples, metadata, ratios):
    groups = defaultdict(list)
    for ex in examples:
        groups[normalize_group(ex.topic)].append(ex)
    strata = defaultdict(list)
    mixed = 0
    for key, members in groups.items():
        categories = Counter(json.loads(ex.assistant)["category"] for ex in members)
        repositories = Counter(metadata[ex.id]["repository"] for ex in members)
        if len(categories) > 1 or len(repositories) > 1:
            mixed += 1
        # A mixed family remains intact. Ties resolve lexically, never randomly.
        category = sorted(categories, key=lambda x: (-categories[x], x))[0]
        repository = sorted(repositories, key=lambda x: (-repositories[x], x))[0]
        date = max(metadata[ex.id]["created_at"] for ex in members)
        strata[(repository, category)].append((date, key, members))
    buckets = {name: [] for name in ratios}
    strata_stats = {}
    assignments = {}
    for stratum, groups_here in sorted(strata.items()):
        groups_here.sort(key=lambda item: (item[0], item[1]))
        total = sum(len(item[2]) for item in groups_here)
        if len(groups_here) < len(ratios):
            raise ValueError(f"Too few independent groups for {stratum}")
        cuts = [total * ratios["train"], total * (ratios["train"] + ratios["val"])]
        offset = 0
        for _date, key, members in groups_here:
            midpoint = offset + len(members) / 2
            name = "train" if midpoint <= cuts[0] else "val" if midpoint <= cuts[1] else "test"
            buckets[name].extend(members)
            assignments[key] = name
            offset += len(members)
        strata_stats["/".join(stratum)] = {
            name: sum(len(members) for _, key, members in groups_here if assignments[key] == name)
            for name in ratios
        }
    for rows in buckets.values():
        rows.sort(key=lambda ex: ex.id)
    return buckets, strata_stats, mixed


def main():
    params = load_params()
    paths, cfg = params["paths"], params["split"]
    if cfg["group_key"] != "topic" or cfg["strategy"] != "temporal_grouped":
        raise SystemExit("Unsupported split grouping/strategy")
    ratios = cfg["ratios"]
    if (
        set(ratios) != {"train", "val", "test"}
        or any(v <= 0 for v in ratios.values())
        or abs(sum(ratios.values()) - 1) > 1e-9
    ):
        raise SystemExit("Split ratios must be positive and sum to one")
    examples = list(iter_examples(paths["clean"]))
    with open(paths["metadata"], encoding="utf-8") as fh:
        metadata = {row["id"]: row for row in map(json.loads, fh)}
    missing = {ex.id for ex in examples} - metadata.keys()
    if missing:
        raise SystemExit(f"Missing metadata for {len(missing)} examples")
    buckets, strata, mixed = assign_groups(examples, metadata, ratios)
    classes = params["diversity"]["required_categories"]
    counts = {
        name: dict(Counter(json.loads(ex.assistant)["category"] for ex in rows))
        for name, rows in buckets.items()
    }
    for name in buckets:
        for category in classes:
            if counts[name].get(category, 0) < cfg["min_class_examples"]:
                raise SystemExit(f"split: {name}/{category} has too few examples")
    for name, rows in buckets.items():
        write_jsonl(paths[name], (ex.model_dump() for ex in rows))
    metrics = {
        "version": params["collect"]["version"],
        "strategy": "temporal within repository/class; whole connected issue families",
        "group_key": cfg["group_key"],
        "groups_total": len({ex.topic for ex in examples}),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {name: len({ex.topic for ex in rows}) for name, rows in buckets.items()},
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "class_counts": counts,
        "strata_sizes": strata,
        "mixed_families": mixed,
        "date_ranges": {
            name: {
                "first": min(metadata[ex.id]["created_at"] for ex in rows),
                "last": max(metadata[ex.id]["created_at"] for ex in rows),
            }
            for name, rows in buckets.items()
        },
    }
    write_json(paths["metrics_split"], metrics)
    print("split: " + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items()))


if __name__ == "__main__":
    main()
