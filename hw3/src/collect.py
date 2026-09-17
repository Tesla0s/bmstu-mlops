"""Convert the frozen GitHub snapshot to chat examples and separate metadata."""

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.config import load_params, source_files
from src.dedup import UnionFind, similar_set_pairs
from src.io_utils import write_json, write_jsonl
from src.textnorm import normalize_text

PREFIX = re.compile(
    r"^\s*(?:\[\s*(BUG|ENH|ENHANCEMENT|FEAT|FEATURE(?: REQUEST)?|QST|QUESTION)\s*\]\s*:?[ ]*|(BUG|ENH|ENHANCEMENT|FEAT|FEATURE REQUEST|QST|QUESTION)\s*:\s*)",
    re.I,
)
TYPE_LINE = re.compile(
    r"(?im)^\s*(?:issue\s+)?type\s*:\s*(?:<b>)?\s*(bug|feature request|question|enhancement)\b"
)
CATEGORY = {
    "bug": "bug",
    "enh": "feature_request",
    "enhancement": "feature_request",
    "feature": "feature_request",
    "feat": "feature_request",
    "bug report": "bug",
    "feature request": "feature_request",
    "qst": "question",
    "question": "question",
}
COMMENTS = re.compile(r"<!--.*?-->", re.S)
DETAILS = re.compile(r"<details\b[^>]*>.*?</details>", re.S | re.I)
TEMPLATE_HEADING = re.compile(
    r"(?im)^\s*#{1,6}\s*(?:Pandas version checks|Reproducible Example|Issue Description|Expected Behavior|Actual Behavior|Feature Type|Problem Description|Feature Description|Solution|Describe the solution you'd like|Describe alternatives you've considered|Additional Context|Additional context|Is your feature request related to a problem.*|Is your feature request related to a problem\?)\s*$"
)


def rank(identifier, seed):
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def strip_template(title, body):
    counts = Counter()
    title, counts["title_prefixes"] = PREFIX.subn("", title)
    body, counts["html_comments"] = COMMENTS.subn("", body)

    def detail(match):
        block = match.group()
        if re.search(
            r"<summary>\s*(?:System Info|Process Info|Workspace Info|Extensions\b|A/B Experiments|Output of .?pd\.show_versions|Installed Versions)|INSTALLED VERSIONS",
            block,
            re.I,
        ):
            counts["diagnostic_blocks"] += 1
            return ""
        return block

    body = DETAILS.sub(detail, body)
    body, counts["empty_diagnostic_tail"] = re.subn(
        r"(?is)<details>\s*<summary>\s*(?:Process Info|Workspace Info|System Info)\s*</summary>\s*$",
        "",
        body,
    )
    # pandas' package inventory follows its final heading; actual examples above stay.
    body, n = re.subn(
        r"(?ims)^#{1,6}\s*(?:Installed Versions|Output of .?pd\.show_versions[^\n]*)\s*\n.*\Z",
        "",
        body,
    )
    counts["diagnostic_blocks"] += n
    body, counts["checkbox_lines"] = re.subn(
        r"(?m)^\s*[-*]\s*\[\s*[xX]?\s*\][^\n]*(?:\n|$)", "", body
    )
    body, counts["type_lines"] = re.subn(r"(?im)^\s*(?:issue\s+)?type\s*:[^\n]*(?:\n|$)", "", body)
    body, counts["feature_type_field"] = re.subn(
        r"(?ims)^#{1,6}\s*Feature Type\s*\n.*?(?=^#{1,6}\s|\Z)", "", body
    )
    body, counts["form_headings"] = TEMPLATE_HEADING.subn("", body)
    body, counts["question_form_headings"] = re.subn(
        r"(?im)^\s*#{1,6}\s*(?:Question about pandas|Research|Link to question on StackOverflow|Code Sample[^\n]*|Output of .*pd\.show_versions[^\n]*)\s*$",
        "",
        body,
    )
    body, counts["question_form_notes"] = re.subn(
        r"(?ims)^\*\*Note\*\*: If you'd still like to submit a question,.*?reproduce your question\.\s*",
        "",
        body,
    )
    body, counts["placeholder_lines"] = re.subn(
        r"(?im)^\s*(?:ADD ISSUE DESCRIPTION HERE|# Your code here, if applicable|_No response_|Does this issue occur when all extensions are disabled\?[^\n]*)\s*$",
        "",
        body,
    )
    body, counts["empty_fences"] = re.subn(r"```[a-z]*\s*\n\s*```", "", body)
    # Reporter metadata adds long repeated tails and identifies the template.
    body, counts["reporter_metadata"] = re.subn(
        r"(?im)^\s*(?:VS Code version|OS version|Modes|Remote OS version):[^\n]*(?:\n|$)",
        "",
        body,
    )
    body, counts["attachment_markup"] = re.subn(
        r"<img\b[^>]*>|!\[[^\]]*\]\([^)]*\)", "[ATTACHMENT]", body, flags=re.I
    )
    # Tracking/query strings can contain e-mail addresses and user identifiers.
    body, counts["tracking_urls"] = re.subn(
        r'https?://[a-zA-Z0-9.-]*safelinks\.protection\.outlook\.com/[^\s<>")]+',
        "[LINK]",
        body,
    )
    body = re.sub(r"\n[ \t]+\n", "\n\n", body.replace("\r\n", "\n"))
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return title.strip(), body, counts


def english_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0


def main():
    started = time.perf_counter()
    params = load_params()
    cfg, paths = params["collect"], params["paths"]
    source = []
    for file in source_files(params):
        for line in file.open(encoding="utf-8"):
            source.append(json.loads(line))
    if len({row["id"] for row in source}) != len(source):
        raise SystemExit("collect: duplicate source ids")
    reviews = json.loads(Path(cfg["reviews"]).read_text(encoding="utf-8"))["records"]
    review_map = {r["id"]: r for r in reviews}
    if len(review_map) != len(reviews):
        raise SystemExit("collect: duplicate review ids")
    reasons, transformations = Counter(), Counter()
    eligible = []
    for row in source:
        mapping = cfg["repositories"].get(row["repository"])
        if mapping is None:
            reasons["unknown_repository"] += 1
            continue
        if row["is_pull_request"]:
            reasons["pull_request"] += 1
            continue
        if row["is_bot"]:
            reasons["bot"] += 1
            continue
        labels = set(row["labels"])
        targets = [name for name, label in mapping.items() if label in labels]
        if len(targets) != 1:
            reasons["ambiguous_or_missing_label"] += 1
            continue
        category = targets[0]
        if not row["body"].strip():
            reasons["empty_body"] += 1
            continue
        declared = []
        match = PREFIX.match(row["title"])
        if match:
            declared.append(CATEGORY[(match.group(1) or match.group(2)).lower()])
        match = TYPE_LINE.search(row["body"])
        if match:
            declared.append(CATEGORY[match.group(1).lower()])
        for heading in re.findall(
            r"(?im)^\s*#{1,6}\s*(feature request|bug report|question)\s*:?[ \t]*$",
            row["body"],
        ):
            declared.append(CATEGORY[heading.lower()])
        if any(value != category for value in declared):
            reasons["declared_type_disagrees_with_label"] += 1
            continue
        review = review_map.get(row["id"])
        if review and review["decision"] == "exclude":
            reasons["review_excluded"] += 1
            continue
        title, body, counts = strip_template(row["title"], row["body"])
        transformations.update(counts)
        if not body or not re.search(r"[A-Za-z]{3}", body.replace("[ATTACHMENT]", "")):
            reasons["no_text_after_template_removal"] += 1
            continue
        if english_share(title + " " + body) < cfg["min_english_word_share"]:
            reasons["non_english_heuristic"] += 1
            continue
        if category == "question" and cfg["question_require_explicit_intent"]:
            # Keep explicit requests for usage help. Questions closed as feature
            # requests / unresolved complaints are too noisy for this task.
            prose = re.sub(
                r"```.*?```|`[^`\n]+`|https?://\S+",
                " ",
                title + "\n" + body,
                flags=re.S,
            )
            help_intent = re.search(
                r"(?i)\b(?:how (?:can|do|does|to|should)|why (?:is|does|do)|what am i|would like to know|need help|help me|please help|any help|anyone can help|if anyone can help|could (?:someone|you) (?:please )?help|(?:could|can) let me know|idk|is it intended)\b",
                prose,
            )
            if "?" not in prose and not help_intent:
                reasons["question_without_explicit_help_intent"] += 1
                continue
        eligible.append({**row, "clean_title": title, "clean_body": body, "category": category})
    # A group is a family of related issues, not the target class or repository.
    # Families use exact substantial titles, similar >=5-word titles and explicit
    # duplicate references. The complete eligible snapshot is grouped before sampling.
    families = UnionFind([row["id"] for row in eligible])
    title_buckets = defaultdict(list)
    long_rows, long_sets = [], []
    for row in eligible:
        words = set(re.findall(r"\w+", normalize_text(row["clean_title"])))
        if len(words) >= 3:
            title_buckets[(row["repository"], normalize_text(row["clean_title"]))].append(row["id"])
        if len(words) >= cfg["title_group_min_words"]:
            long_rows.append(row)
            long_sets.append(words)
        for linked_repo, url_number, local_number in re.findall(
            r"(?i)\b(?:duplicate\s+of|same\s+(?:issue\s+)?as)\s+(?:https://github\.com/([^/\s]+/[^/\s]+)/issues/(\d+)|#(\d+))",
            row["body"],
        ):
            related = f"{linked_repo or row['repository']}#{url_number or local_number}"
            if related not in families.parent:
                families.parent[related] = related
            families.union(row["id"], related)
    exact_title_links = 0
    for ids in title_buckets.values():
        for identifier in ids[1:]:
            families.union(ids[0], identifier)
            exact_title_links += 1
    similar_titles = similar_set_pairs(long_sets, cfg["title_group_threshold"])
    similar_titles = [
        (i, j)
        for i, j in similar_titles
        if long_rows[i]["repository"] == long_rows[j]["repository"]
    ]
    for i, j in similar_titles:
        families.union(long_rows[i]["id"], long_rows[j]["id"])
    buckets = defaultdict(list)
    for row in eligible:
        buckets[(row["repository"], row["category"])].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: rank(row["id"], cfg["seed"]))
    repository_caps = {
        repo: min(len(buckets[(repo, category)]) for category in cfg["repositories"][repo])
        for repo in cfg["repositories"]
    }
    by_class = {}
    for category in sorted({key[1] for key in buckets}):
        queues = [
            buckets[(repo, category)][: repository_caps[repo]]
            for repo in sorted(cfg["repositories"])
        ]
        order = []
        for index in range(max(map(len, queues))):
            for queue in queues:
                if index < len(queue):
                    order.append(queue[index])
        by_class[category] = order
    available = min(map(len, by_class.values()))
    base_quota = cfg["per_class"]["v1"]
    quota = min(cfg["per_class"][cfg["version"]], available)
    if available <= base_quota:
        raise SystemExit(
            f"Need more reviewed source questions: only {available} per class; cannot build expanded v2"
        )
    selected = []
    for start, stop in ((0, min(base_quota, quota)), (base_quota, quota)):
        if start >= stop:
            continue
        for category in sorted(by_class):
            selected.extend(by_class[category][start:stop])
    examples, metadata = [], []
    for row in selected:
        identifier = row["id"]
        topic = "issue-family:" + families.find(identifier)
        prompt = cfg["system_prompts"][
            int(rank(identifier, cfg["seed"]), 16) % len(cfg["system_prompts"])
        ]
        examples.append(
            {
                "id": identifier,
                "topic": topic,
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"Title: {row['clean_title']}\n\nDescription:\n{row['clean_body']}",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"category": row["category"]}),
                    },
                ],
            }
        )
        metadata.append(
            {
                key: row[key]
                for key in (
                    "id",
                    "repository",
                    "url",
                    "number",
                    "created_at",
                    "closed_at",
                    "retrieved_at",
                    "labels",
                    "category",
                )
            }
        )
    write_jsonl(paths["raw"], examples)
    write_jsonl(paths["metadata"], metadata)
    metrics = {
        "version": cfg["version"],
        "source_rows": len(source),
        "eligible_rows": len(eligible),
        "rows_written": len(examples),
        "excluded": dict(reasons),
        "not_selected_by_quota": len(eligible) - len(selected),
        "transformations": dict(transformations),
        "eligible_by_repository_class": {
            f"{repo}/{cat}": len(rows) for (repo, cat), rows in sorted(buckets.items())
        },
        "selected_by_class": dict(Counter(row["category"] for row in selected)),
        "selected_by_repository_class": dict(
            Counter(row["repository"] + "/" + row["category"] for row in selected)
        ),
        "selected_by_repository": dict(Counter(row["repository"] for row in selected)),
        "groups": len({row["topic"] for row in examples}),
        "exact_title_links": exact_title_links,
        "similar_title_links": len(similar_titles),
        "system_prompt_variants": len({row["messages"][0]["content"] for row in examples}),
        "reviewed_source_records": len(reviews),
    }
    write_json(paths["metrics_collect"], metrics)
    print(
        f"collect: {cfg['version']}, {len(source)} source -> {len(eligible)} eligible -> {len(examples)} selected; {time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
