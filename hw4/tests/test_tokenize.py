"""Проверки границ маски, потери ответа и дополнения реальных последовательностей."""

import copy
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from src.collate import DynamicPaddingCollator
from src.config import load_params
from src.prompt import build_chat_text, load_tokenizer, prompt_token_len
from src.tokenize_data import (
    encode_example,
    mask_prompt,
    padding_cost,
    process_split,
    read_jsonl,
    truncation_stats,
)


def record(text="Why does the function return an empty list?", id_="example"):
    return {
        "id": id_,
        "topic": "test-family",
        "messages": [
            {"role": "system", "content": "Classify the issue. Return a category as JSON."},
            {"role": "user", "content": text},
            {"role": "assistant", "content": '{"category": "question"}'},
        ],
    }


class TokenizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_params()
        cls.tokenizer = load_tokenizer(cls.params)

    def test_unanswered_inference_matches_training_prompt(self):
        m = record()["messages"]
        tok, p = self.tokenizer, self.params
        prompt = build_chat_text(tok, m, p, True)
        self.assertEqual(prompt, build_chat_text(tok, m[:-1], p, True))
        self.assertTrue(build_chat_text(tok, m, p, False).encode().startswith(prompt.encode()))

    def test_supervision_is_answer_and_eos_not_empty_reasoning(self):
        row = record()
        e = encode_example(self.tokenizer, row, self.params, 2176)
        n = e["_meta"]["prompt_len"]
        self.assertGreater(n, 0)
        self.assertEqual(e["labels"][:n], [-100] * n)
        self.assertEqual(e["labels"][n:], e["input_ids"][n:])
        text = self.tokenizer.decode(e["input_ids"][n:]).strip()
        self.assertEqual(text, row["messages"][-1]["content"] + self.tokenizer.eos_token)
        self.assertFalse(e["_meta"]["bpe_fallback"])

    def test_thinking_mode_does_not_silently_train_on_reasoning(self):
        p = copy.deepcopy(self.params)
        p["model"]["enable_thinking"] = True
        with self.assertRaisesRegex(ValueError, "не только ответ"):
            encode_example(self.tokenizer, record(), p, 2176)

    def test_real_bpe_merge_masks_crossing_token(self):
        prompt, full = "Приве", "Привет, мир"
        enc = self.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        n, fallback = prompt_token_len(
            self.tokenizer, prompt, enc["input_ids"], enc["offset_mapping"]
        )
        self.assertTrue(fallback)
        self.assertGreaterEqual(enc["offset_mapping"][n][0], len(prompt))
        self.assertLess(enc["offset_mapping"][n - 1][0], len(prompt))

    def test_fully_merged_answer_has_no_fake_supervision(self):
        class MergedTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [123]}

        n, fallback = prompt_token_len(MergedTokenizer(), "ab", [456], [(0, 3)])
        self.assertEqual((n, fallback), (1, True))
        self.assertEqual(mask_prompt([456], n), [-100])

    def test_truncation_at_prompt_boundary_has_no_labels(self):
        row = record()
        full = encode_example(self.tokenizer, row, self.params, 2176)
        boundary = full["_meta"]["prompt_len"]
        e = encode_example(self.tokenizer, row, self.params, boundary)
        self.assertEqual(e["_meta"]["supervised"], 0)
        self.assertTrue(e["_meta"]["truncated"])
        self.assertEqual(e["labels"], [-100] * boundary)

    def test_partial_answer_is_counted_separately(self):
        row = record()
        full = encode_example(self.tokenizer, row, self.params, 2176)
        e = encode_example(self.tokenizer, row, self.params, full["_meta"]["prompt_len"] + 2)
        self.assertEqual(e["_meta"]["supervised"], 2)
        self.assertFalse(e["_meta"]["answer_complete"])

    def test_dropped_rows_remain_in_truncation_denominator(self):
        rows = [record(id_="short"), record("word " * 1000, id_="long")]
        p = copy.deepcopy(self.params)
        p["tokenize"]["max_seq_len"] = 128
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with warnings.catch_warnings(record=True) as caught:
                examples, stats, _ = process_split(self.tokenizer, "fixture", path, p)
        self.assertEqual([r["id"] for r in examples], ["short"])
        self.assertEqual(stats["examples_in"], 2)
        self.assertEqual(stats["dropped_no_supervision"], 1)
        self.assertEqual(stats["truncated_ratio"], 0.5)
        self.assertFalse(stats["truncation_passed"])
        self.assertEqual(len(caught), 1)

    def test_equal_threshold_is_allowed(self):
        p = copy.deepcopy(self.params)
        p["tokenize"]["truncated_warn_ratio"] = 0.5
        with warnings.catch_warnings(record=True) as caught:
            result = truncation_stats([{"truncated": True}, {"truncated": False}], "test", p)
        self.assertTrue(result["truncation_passed"])
        self.assertFalse(caught)


class PaddingTests(unittest.TestCase):
    def test_left_padding_does_not_mask_real_eos_when_pad_equals_eos(self):
        features = [
            {"input_ids": [1, 7], "attention_mask": [1, 1], "labels": [-100, 7]},
            {"input_ids": [9, 2, 7], "attention_mask": [1, 1, 1], "labels": [-100, 2, 7]},
        ]
        b = DynamicPaddingCollator(7)(features)
        self.assertEqual(b["input_ids"].tolist(), [[7, 1, 7], [9, 2, 7]])
        self.assertEqual(b["attention_mask"].tolist(), [[0, 1, 1], [1, 1, 1]])
        self.assertEqual(b["labels"].tolist(), [[-100, -100, 7], [-100, 2, 7]])
        self.assertEqual(b["labels"].dtype, torch.long)

    def test_rejects_right_padding_and_malformed_batch(self):
        with self.assertRaises(ValueError):
            DynamicPaddingCollator(7, "right")
        with self.assertRaises(ValueError):
            DynamicPaddingCollator(7)([])
        with self.assertRaises(ValueError):
            DynamicPaddingCollator(7)([{"input_ids": [1], "attention_mask": [1, 1], "labels": [1]}])

    def test_padding_cost_keeps_last_incomplete_batch(self):
        examples = [{"input_ids": [1] * size} for size in (2, 3, 5)]
        p = {"batch": {"size": 2, "seed": 42}, "tokenize": {"max_seq_len": 8}}
        report = padding_cost(examples, p)
        self.assertEqual(report["batches"], 2)
        self.assertEqual(report["useful_tokens"], 10)
        self.assertEqual(report["strategies"]["dynamic_sorted"]["positions"], 11)
        self.assertEqual(report["strategies"]["fixed_max_seq_len"]["positions"], 24)


class InputTests(unittest.TestCase):
    def test_error_reports_physical_line_even_with_unicode_separator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(record("line\u2028break"), ensure_ascii=False) + "\n{}\n")
            with self.assertRaisesRegex(ValueError, r"bad.jsonl:2:"):
                read_jsonl(path)

    def test_repeated_ids_and_empty_dataset_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text((json.dumps(record()) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "повторный id"):
                read_jsonl(path)
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "пустой датасет"):
                read_jsonl(path)
