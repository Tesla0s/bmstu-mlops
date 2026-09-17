"""Focused regressions beyond the unchanged instructor checks."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

from src.collect import strip_template
from src.dedup import _pairs, similar_set_pairs
from src.diversity import measure, violations
from src.pii import scrub
from src.schema import Example, SchemaError, iter_examples
from src.split import assign_groups


def example(
    identifier="one",
    topic="group",
    category="bug",
    user="A sufficiently detailed issue description.",
):
    return {
        "id": identifier,
        "topic": topic,
        "messages": [
            {"role": "system", "content": "Classify the issue."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps({"category": category})},
        ],
    }


class JaccardTests(unittest.TestCase):
    def test_prefix_index_equals_exhaustive_search(self):
        tokens = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        sets = [set(combo) for size in range(7) for combo in itertools.combinations(tokens, size)]
        for threshold in (0.3, 0.5, 0.85, 1.0):
            expected = []
            for j, right in enumerate(sets):
                for i, left in enumerate(sets[:j]):
                    union = left | right
                    score = len(left & right) / len(union) if union else 1.0
                    if score + 1e-12 >= threshold:
                        expected.append((i, j))
            self.assertEqual(set(similar_set_pairs(sets, threshold)), set(expected))

    def test_cross_index_equals_exhaustive_search(self):
        sets = [set(str(i) for i in range(n)) for n in range(12)]
        left, right = sets[::2], sets[1::2]
        for threshold in (0.5, 0.85, 1.0):
            expected = {
                (i, j)
                for i, a in enumerate(left)
                for j, b in enumerate(right)
                if len(a & b) / len(a | b) + 1e-12 >= threshold
            }
            self.assertEqual(set(_pairs(left, right, threshold, False)), expected)

    def test_rejects_unrelated_lsh_candidates(self):
        sets = [set("abcdefg"), set("abcxyzt"), set("abcdefg")]
        self.assertEqual(similar_set_pairs(sets, 0.85), [(0, 2)])

    def test_threshold_boundary(self):
        a = set(range(20))
        b = set(range(17))
        self.assertEqual(similar_set_pairs([a, b], 0.85), [(0, 1)])
        self.assertEqual(similar_set_pairs([a, b], 0.85001), [])


class SchemaTests(unittest.TestCase):
    def test_error_reports_actual_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "raw.jsonl"
            bad = example("bad")
            bad["messages"][2]["role"] = "user"
            file.write_text(
                "\n".join(json.dumps(row) for row in (example(), example("two"), bad)) + "\n"
            )
            with self.assertRaisesRegex(SchemaError, r"raw\.jsonl:3"):
                list(iter_examples(file))

    def test_unicode_line_separator_is_part_of_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "raw.jsonl"
            row = example(user="Text before\u2028text after the Unicode separator.")
            file.write_text(json.dumps(row, ensure_ascii=False) + "\n")
            self.assertEqual(len(list(iter_examples(file))), 1)

    def test_blank_topic_is_rejected(self):
        with self.assertRaises(ValueError):
            Example.model_validate(example(topic=" "))


class CleaningTests(unittest.TestCase):
    def test_pii_masks_real_contacts_preserves_code(self):
        text = (
            "Contact person@example.com, +7 (999) 123-45-67, дата рождения 12.03.1975. "
            "Ping @some-person. Version 2.3.3.\n"
            "```python\n@dataclass\nclass Sample: pass\n```\n"
            "Use `@pytest.fixture` and @scope/package.\nRegards,\nAlice Smith\n"
        )
        cleaned, hits = scrub(text)
        self.assertNotIn("person@example.com", cleaned)
        self.assertNotIn("12.03.1975", cleaned)
        self.assertIn("[PHONE]", cleaned)
        self.assertNotIn("@some-person", cleaned)
        self.assertIn("@dataclass", cleaned)
        self.assertIn("@pytest.fixture", cleaned)
        self.assertIn("@scope/package", cleaned)
        self.assertIn("2.3.3", cleaned)
        self.assertNotIn("Alice Smith", cleaned)
        self.assertGreaterEqual(sum(hits.values()), 5)

    def test_connection_password_and_query_token_are_masked(self):
        cleaned, hits = scrub(
            "postgresql://demo:privatepass@localhost:5432/db https://api.example.org/?token=abcd&mode=fast"
        )
        self.assertNotIn("privatepass", cleaned)
        self.assertNotIn("token=abcd", cleaned)
        self.assertIn("mode=fast", cleaned)
        self.assertIn("@localhost:5432/db", cleaned)
        self.assertEqual(hits["url_credentials"], 1)
        self.assertEqual(hits["query_secret"], 1)

    def test_templates_removed_without_losing_functional_query(self):
        title, body, counts = strip_template(
            "[BUG] Request fails",
            "Type: <b>Bug</b>\n\n"
            "Calling https://example.org/api?mode=fast returns an error.\n"
            "### Installed Versions\npackage inventory only",
        )
        self.assertEqual(title, "Request fails")
        self.assertNotIn("Type:", body)
        self.assertNotIn("package inventory", body)
        self.assertIn("api?mode=fast", body)
        self.assertEqual(counts["title_prefixes"], 1)

    def test_unfilled_question_form_is_removed(self):
        _, body, _ = strip_template(
            "Question",
            "#### Question about pandas\n"
            "**Note**: If you'd still like to submit a question, please read a guide "
            "detailing how to provide the necessary information for us to reproduce your question.\n"
            "```python\n# Your code here, if applicable\n```",
        )
        self.assertEqual(body.strip(), "")


class SplitTests(unittest.TestCase):
    def test_related_rows_never_split_and_time_orders_groups(self):
        rows = []
        meta = {}
        for category in ("bug", "feature_request", "question"):
            for group in range(20):
                for member in range(2):
                    identifier = f"{category}:{group}:{member}"
                    ex = Example.model_validate(
                        example(identifier, f"{category}:family:{group}", category)
                    )
                    rows.append(ex)
                    meta[identifier] = {
                        "repository": "test/project",
                        "created_at": f"2025-01-{group + 1:02}T00:00:00Z",
                    }
        buckets, _, _ = assign_groups(rows, meta, {"train": 0.7, "val": 0.15, "test": 0.15})
        assigned = {}
        for name, items in buckets.items():
            for ex in items:
                self.assertEqual(assigned.setdefault(ex.topic, name), name)
        self.assertEqual(sum(map(len, buckets.values())), len(rows))
        for category in ("bug", "feature_request", "question"):
            dates = {
                name: [
                    meta[ex.id]["created_at"]
                    for ex in items
                    if json.loads(ex.assistant)["category"] == category
                ]
                for name, items in buckets.items()
            }
            self.assertLessEqual(max(dates["train"]), min(dates["val"]))
            self.assertLessEqual(max(dates["val"]), min(dates["test"]))


class DiversityTests(unittest.TestCase):
    def test_balanced_lengths_do_not_hide_missing_class(self):
        from src.config import load_params

        cfg = load_params()["diversity"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one-class.jsonl"
            rows = [
                example(str(i), str(i), "bug", f"Unique report {i} about software failure.")
                for i in range(1200)
            ]
            for i, row in enumerate(rows):
                row["messages"][0]["content"] = f"Instruction variation {i % 5}"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            problems = violations(measure(str(path), "topic"), cfg)
            self.assertTrue(any("категории" in reason for reason in problems))


if __name__ == "__main__":
    unittest.main()
