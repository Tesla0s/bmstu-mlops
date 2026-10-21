"""Дополнительные проверки; исходный каталог tests/ не изменяется."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluation.solution import quality_gate, score


def items(n=60):
    return [
        dict(id=str(i), choices=["bug", "question"], expected="bug", slices=[f"class:{i % 3}", "all"])
        for i in range(n)
    ]


def report():
    rows = items()
    outputs = [dict(id=x["id"], raw='{"answer":"bug"}', error=None) for x in rows]
    out = score(rows, outputs)
    behavior = items(4)
    out.update(
        status="complete",
        model=dict(kind="base", name="fixture", revision="a" * 40),
        protocol_sha256="a" * 64,
        golden_sha256="b" * 64,
        baseline_spec_sha256="c" * 64,
        behavioral=score(behavior, outputs[:4]),
    )
    return out


class AdditionalChecks(unittest.TestCase):
    def test_duplicate_json_fields_and_nonfinite_are_invalid(self):
        for raw in (
            '{"answer":"bug","answer":"bug"}',
            '{"answer":NaN}',
            '{"answer":["bug"]}',
            '{"answer":"bug"} trailing',
        ):
            with self.subTest(raw=raw):
                self.assertEqual(score(items(1), [dict(id="0", raw=raw, error=None)])["overall"]["valid"], 0)

    def test_error_with_plausible_answer_is_still_failure(self):
        result = score(items(1), [dict(id="0", raw='{"answer":"bug"}', error="timeout")])
        self.assertEqual(
            result["overall"], dict(n=1, correct=0, valid=0, errors=1, accuracy=0, format_rate=0)
        )

    def test_overlapping_slices_do_not_increase_overall_n(self):
        result = report()
        self.assertEqual(result["overall"]["n"], 60)
        self.assertEqual(result["slices"]["all"]["n"], 60)
        self.assertEqual(sum(x["n"] for x in result["slices"].values()), 120)

    def test_extra_missing_duplicate_task_and_output_ids(self):
        rows = items(2)
        out = [dict(id=x["id"], raw='{"answer":"bug"}', error=None) for x in rows]
        for a, b in (
            (rows + rows[:1], out),
            (rows, out + out[:1]),
            (rows, out + [dict(id="extra", raw="", error=None)]),
            (rows, out[:1]),
        ):
            with self.assertRaises(ValueError):
                score(a, b)

    def test_tampered_counts_and_group_membership_rejected(self):
        mutations = [
            lambda x: x["overall"].update(n=True),
            lambda x: x["slices"]["all"].update(errors=1),
            lambda x: x["per_item"][0].update(slices=["all"]),
            lambda x: x["behavioral"]["per_item"][0].update(id="other"),
            lambda x: x.update(model={}),
            lambda x: x.update(protocol_sha256=""),
        ]
        for mutate in mutations:
            candidate = report()
            mutate(candidate)
            with self.assertRaises(ValueError):
                quality_gate(report(), candidate, ["class:0"])

    def test_baseline_generation_errors_also_block_release(self):
        baseline = report()
        row = baseline["per_item"][0]
        row.update(error="timeout", valid=False, correct=False)
        # Даже подделка согласованных средних не должна обходить запрет ошибок.
        with self.assertRaises(ValueError):
            quality_gate(baseline, report(), ["class:0"])

    def test_loss_gate_before_any_registry_mutation(self):
        from src.promote import promote

        run = SimpleNamespace(
            info=SimpleNamespace(status="FINISHED", run_id="fixture"),
            data=SimpleNamespace(metrics=dict(base_val_loss=1.0, final_val_loss=0.2)),
        )
        for threshold in (0.0, float("nan"), -1.0):
            client = Mock()
            with patch("src.promote.mlflow.register_model") as register, self.assertRaises(ValueError):
                promote(client, run, "fixture", "candidate", threshold)
            register.assert_not_called()
            self.assertEqual(client.mock_calls, [])

    def test_negative_or_missing_loss_rejected(self):
        from src.promote import gate

        for metrics in (
            dict(base_val_loss=1, final_val_loss=-1),
            dict(base_val_loss=0, final_val_loss=0),
            dict(base_val_loss=1),
            dict(base_val_loss=1, final_val_loss=float("inf")),
        ):
            run = SimpleNamespace(
                info=SimpleNamespace(status="FINISHED"), data=SimpleNamespace(metrics=metrics)
            )
            with self.assertRaises(ValueError):
                gate(run, 1.0)


if __name__ == "__main__":
    unittest.main()
