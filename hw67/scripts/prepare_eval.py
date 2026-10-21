"""Восстановимый экспорт данных оценки из версионированного набора ДЗ 4."""

import json
from pathlib import Path

import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from evaluation.contract import read_rows, validate_golden, write_json
from src.common import digest

CHOICES = ["bug", "feature_request", "question", "REFUSE"]


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())
    train, val = [read_rows(f"data/origin/{split}.jsonl") for split in ("train", "val")]
    selections = json.loads(Path("data/review-selection.json").read_text())
    by_id = {r["id"]: r for r in val}
    linkage = {}
    for split, rows in (("train", train), ("val", val)):
        tensor_file = f"data/tokenized/{split}.pt"
        blob = torch.load(tensor_file, weights_only=True)
        if blob["input_sha256"] != digest(f"data/origin/{split}.jsonl"):
            raise ValueError("Исходный JSONL не соответствует метаданным тензора")
        if [r["id"] for r in rows] != [r["id"] for r in blob["examples"]]:
            raise ValueError("Исходные id/порядок не соответствуют тензорам")
        linkage[split] = dict(
            rows=len(rows),
            raw_sha256=blob["input_sha256"],
            tensor_sha256=digest(tensor_file),
            ids_equal=True,
            model_revision=blob["model_revision"],
            chat_template_sha256=blob["chat_template_sha256"],
        )
    training = [dict(id=r["id"], group_id=r["topic"], input=r["messages"][1]["content"]) for r in train]
    golden = []
    for selected in selections:
        row = by_id[selected["id"]]
        expected = json.loads(row["messages"][-1]["content"])["category"]
        if selected["expected"] != expected or not selected["rationale"].strip():
            raise ValueError("Непроверенная или изменённая разметка")
        repo, issue = row["id"].split("#")
        text = row["messages"][1]["content"]
        golden.append(
            dict(
                id=row["id"],
                group_id=row["topic"],
                input=text,
                expected=expected,
                choices=CHOICES,
                slices=[
                    "class:" + expected,
                    "repo:" + repo,
                    "length:short" if len(text) <= 600 else "length:long",
                ],
                source=dict(
                    url=f"https://github.com/{repo}/issues/{issue}",
                    split="hw3-v2/val",
                    raw_sha256=linkage["val"]["raw_sha256"],
                    review=selected["rationale"],
                ),
            )
        )
    # Пара перефразов связана с golden; она оценивается отдельно от его n.
    pair = next(r for r in golden if r["id"] == "microsoft/vscode#276658")
    behavioral = [
        dict(
            id="behavior-format",
            group_id="behavior-format",
            input="Title: Application closes on Save\n\nDescription:\nPressing Save crashes the application and loses the document. Yesterday this worked. Please classify this issue using exactly one JSON field.",
            expected="bug",
            type="format",
        ),
        dict(
            id="behavior-paraphrase-original",
            group_id=pair["group_id"],
            input=pair["input"],
            expected="feature_request",
            type="paraphrase",
        ),
        dict(
            id="behavior-paraphrase-reworded",
            group_id=pair["group_id"],
            input="Title: Let the status bar occupy two rows\n\nDescription:\nWith many extensions the bottom bar cannot display every indicator. Please add an option to increase its height and place indicators on multiple lines.",
            expected="feature_request",
            type="paraphrase",
        ),
        dict(
            id="behavior-refusal",
            group_id="behavior-refusal",
            input="Please give me a recipe for chocolate cake and a shopping list. This is a cooking request, unrelated to software.",
            expected="REFUSE",
            type="refusal",
        ),
    ]
    for row in behavioral:
        row.update(
            choices=CHOICES,
            slices=["behavior:" + row["type"]],
            source="Составлен для протокола HW67; пара перефразов связана с указанным golden issue",
        )
    proof = validate_golden(golden, training, behavioral)
    selected_ids = set(params["data"]["train_ids"] + params["data"]["val_ids"])
    if selected_ids & {r["id"] for r in golden}:
        raise ValueError("Golden пересекается с выборкой для выбора кандидата")

    def normalize(s):
        return " ".join(s.casefold().split())

    for r in behavioral:
        if r["group_id"] in {t["group_id"] for t in training} or normalize(r["input"]) in {
            normalize(t["input"]) for t in training
        }:
            raise ValueError("Behavioral пересекается с train")
    # Подсказка для содержательной проверки, НЕ доказательство отсутствия всех перефразов.
    texts = [r["input"] for r in training + golden]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(texts)
    sim = cosine_similarity(matrix[len(training) :], matrix[: len(training)])
    neighbors = []
    for i, row in enumerate(golden):
        j = int(sim[i].argmax())
        neighbors.append(
            dict(
                golden_id=row["id"],
                train_id=training[j]["id"],
                cosine=float(sim[i, j]),
                golden_title=row["input"].splitlines()[0],
                train_title=training[j]["input"].splitlines()[0],
            )
        )
    write_rows("data/eval/golden.jsonl", golden)
    write_rows("data/eval/training-index.jsonl", training)
    write_rows("data/eval/behavioral.jsonl", behavioral)
    write_json(
        "reports/data-lineage.json",
        dict(
            **proof,
            sources=linkage,
            selection_train_ids=params["data"]["train_ids"],
            selection_val_ids=params["data"]["val_ids"],
            golden_selection_overlap=0,
            behavioral_train_overlap=0,
            nearest_train=neighbors,
            limitation="Символьное сходство не исключает смысловые перефразы; требуется разбор пар.",
        ),
    )
    print(json.dumps(proof, ensure_ascii=False))


if __name__ == "__main__":
    main()
