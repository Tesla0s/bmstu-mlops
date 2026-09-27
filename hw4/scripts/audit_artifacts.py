"""Проверка каждого сохранённого примера и каждой позиции батча."""

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from src.collate import DynamicPaddingCollator
from src.config import load_params
from src.prompt import build_chat_text, load_tokenizer
from src.tokenize_data import render_report


def main():
    params = load_params()
    tok = load_tokenizer(params)
    metrics = json.loads(Path("metrics/tokenize.json").read_text())
    assert metrics["passed"]
    assert Path("docs/tokenize_report.md").read_text() == render_report(metrics)
    result = {"passed": True, "splits": {}, "generated_report_matches_metrics": True}
    for name in ("train", "val"):
        path = Path(params["data"][f"{name}_jsonl"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == params["provenance"]["sha256"][name]
        with path.open() as stream:
            rows = [json.loads(line) for line in stream]
        blob = torch.load(
            Path(params["data"]["out_dir"]) / f"{name}.pt", map_location="cpu", weights_only=True
        )
        examples = blob["examples"]
        assert len(rows) == len(examples)
        assert [r["id"] for r in rows] == [e["id"] for e in examples]
        assert blob["input_sha256"] == digest
        assert blob["padding_side"] == "left"
        assert blob["model_revision"] == params["model"]["revision"]
        assert blob["pad_token_id"] == tok.pad_token_id
        lengths, supervised = [], 0
        for row, e in zip(rows, examples, strict=True):
            full = build_chat_text(tok, row["messages"], params, False)
            prefix = build_chat_text(tok, row["messages"], params, True)
            assert full.encode().startswith(prefix.encode())
            ids = tok(full, add_special_tokens=False)["input_ids"]
            prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            assert ids[: len(prefix_ids)] == prefix_ids
            assert e["input_ids"] == ids
            assert e["attention_mask"] == [1] * len(ids)
            assert len(e["labels"]) == len(ids) <= params["tokenize"]["max_seq_len"]
            assert e["labels"][: len(prefix_ids)] == [-100] * len(prefix_ids)
            assert e["labels"][len(prefix_ids) :] == ids[len(prefix_ids) :]
            answer = tok.decode(ids[len(prefix_ids) :]).strip()
            assert answer == row["messages"][-1]["content"] + tok.eos_token
            lengths.append(len(ids))
            supervised += len(ids) - len(prefix_ids)
        stats = metrics["splits"][name]
        assert stats["total_tokens"] == sum(lengths)
        assert stats["supervised_tokens"] == supervised
        assert stats["truncated"] == stats["dropped_no_supervision"] == 0
        assert stats["complete_answers"] == len(rows)
        for p in (50, 90, 95, 99):
            assert stats["length_tokens"][f"p{p}"] == math.ceil(float(np.percentile(lengths, p)))
        shuffled = examples.copy()
        random.Random(params["batch"]["seed"]).shuffle(shuffled)
        collator = DynamicPaddingCollator(tok.pad_token_id)
        positions = 0
        for start in range(0, len(shuffled), params["batch"]["size"]):
            batch = shuffled[start : start + params["batch"]["size"]]
            out = collator(batch)
            width = max(len(e["input_ids"]) for e in batch)
            assert tuple(out["input_ids"].shape) == (len(batch), width)
            positions += out["input_ids"].numel()
            for i, e in enumerate(batch):
                n = width - len(e["input_ids"])
                assert out["input_ids"][i, :n].tolist() == [tok.pad_token_id] * n
                assert out["labels"][i, :n].tolist() == [-100] * n
                assert out["attention_mask"][i, :n].tolist() == [0] * n
                assert out["input_ids"][i, n:].tolist() == e["input_ids"]
                assert out["labels"][i, n:].tolist() == e["labels"]
                assert out["attention_mask"][i, n:].tolist() == e["attention_mask"]
        assert positions == stats["padding"]["strategies"]["dynamic_shuffled"]["positions"]
        result["splits"][name] = {
            "examples_checked": len(rows),
            "tokens_checked": sum(lengths),
            "supervised_tokens_checked": supervised,
            "all_answers_and_eos_intact": True,
            "byte_prefix_and_token_boundary_match": True,
            "all_batch_padding_positions_correct": True,
            "input_sha256": digest,
            "tokenized_sha256": hashlib.sha256(
                (Path(params["data"]["out_dir"]) / f"{name}.pt").read_bytes()
            ).hexdigest(),
        }
        print(f"{name}: проверены {len(rows)} примеров, маски, конец ответа и все батчи")
    original_hash = "f2002c8c9519"
    fingerprint = hashlib.sha256(Path("tests/check.sh").read_bytes()).hexdigest()
    assert fingerprint.startswith(original_hash)
    result["teacher_check_sha256"] = fingerprint
    Path("reports").mkdir(exist_ok=True)
    Path("reports/artifact_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Проверка готовых файлов пройдена; отпечаток tests/check.sh: {fingerprint[:12]}")


if __name__ == "__main__":
    main()
