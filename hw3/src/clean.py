"""Schema -> length -> personal data -> exact duplicates -> near duplicates."""

import time
from collections import Counter, defaultdict

from src.config import load_params
from src.dedup import UnionFind, near_pairs
from src.io_utils import write_json, write_jsonl
from src.pii import PATTERNS, scrub
from src.schema import iter_examples
from src.stats import spread
from src.textnorm import normalize_group, normalize_text


def main():
    params = load_params()
    cfg, paths = params["clean"], params["paths"]
    started = time.perf_counter()
    examples = list(iter_examples(paths["raw"]))
    ids = [ex.id for ex in examples]
    if len(set(ids)) != len(ids):
        raise SystemExit("clean: repeated id in raw data")
    kept, dropped_length = [], 0
    for ex in examples:
        if (
            not cfg["min_user_chars"] <= len(ex.user) <= cfg["max_user_chars"]
            or len(ex.assistant) < cfg["min_assistant_chars"]
        ):
            dropped_length += 1
        else:
            kept.append(ex)
    hits = Counter()
    touched_rows = 0
    if cfg["pii"]["enabled"]:
        for ex in kept:
            touched = False
            for message in ex.messages:
                message.content, found = scrub(message.content)
                hits.update(found)
                touched |= bool(found)
            touched_rows += touched
    topics = UnionFind(sorted({normalize_group(ex.topic) for ex in kept}))
    buckets = defaultdict(list)
    for i, ex in enumerate(kept):
        buckets[normalize_text(ex.user)].append(i)
    exact, conflicting = set(), set()
    for indices in buckets.values():
        first = indices[0]
        for i in indices[1:]:
            topics.union(normalize_group(kept[first].topic), normalize_group(kept[i].topic))
        if len({normalize_text(kept[i].assistant) for i in indices}) > 1:
            conflicting.update(indices)
        else:
            exact.update(indices[1:])
    kept = [ex for i, ex in enumerate(kept) if i not in exact | conflicting]
    near = set()
    near_conflicts = set()
    pairs = []
    if cfg["near_dup"]["enabled"]:
        nd = cfg["near_dup"]
        pairs = near_pairs(
            [normalize_text(ex.user) for ex in kept],
            nd["shingle_words"],
            nd["threshold"],
        )
        clusters = UnionFind(range(len(kept)))
        for i, j in pairs:
            clusters.union(i, j)
            topics.union(normalize_group(kept[i].topic), normalize_group(kept[j].topic))
        components = defaultdict(list)
        for i in range(len(kept)):
            components[clusters.find(i)].append(i)
        for indices in components.values():
            if len({normalize_text(kept[i].assistant) for i in indices}) > 1:
                near_conflicts.update(indices)
            else:
                near.update(indices[1:])
    kept = [ex for i, ex in enumerate(kept) if i not in near | near_conflicts]
    for ex in kept:
        ex.topic = topics.find(normalize_group(ex.topic))
    write_jsonl(paths["clean"], (ex.model_dump() for ex in kept))
    metrics = {
        "version": params["collect"]["version"],
        "rows_in": len(examples),
        "rows_out": len(kept),
        "dropped_length": dropped_length,
        "dropped_exact_dup": len(exact),
        "dropped_near_dup": len(near),
        "dropped_conflicting_labels": len(conflicting) + len(near_conflicts),
        "near_pairs_before_removal": len(pairs),
        "pii_rows_masked": touched_rows,
        "pii_hits": {key: hits[key] for key in PATTERNS},
        "groups": len({ex.topic for ex in kept}),
        "user_chars": spread([len(ex.user) for ex in kept]),
        "assistant_chars": spread([len(ex.assistant) for ex in kept]),
    }
    write_json(paths["metrics_clean"], metrics)
    print(
        f"clean: {len(examples)} -> {len(kept)}; length -{dropped_length}, exact -{len(exact)}, near -{len(near)}, conflicting -{metrics['dropped_conflicting_labels']}; {time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
