"""Дополнительные проверки; исходный каталог tests/ не изменяется."""

import copy
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

    def test_exact_model_identity_required(self):
        adapter = dict(kind="adapter", name="issues", version="1", run_id="a" * 32, uri="models:/issues/1")
        candidate = report()
        candidate["model"] = adapter
        self.assertTrue(quality_gate(report(), candidate, ["class:0"])["passed"])
        bad_specs = [
            {"kind": "base"},
            {"kind": "adapter"},
            dict(kind="base", name="fixture", revision="main"),
            dict(kind="base", name=" ", revision="a" * 40),
            dict(kind="unknown", name="fixture"),
        ]
        for key in adapter:
            spec = dict(adapter)
            del spec[key]
            bad_specs.append(spec)
        for key, value in (("version", "latest"), ("run_id", ""), ("uri", "models:/issues@candidate")):
            spec = dict(adapter)
            spec[key] = value
            bad_specs.append(spec)
        for spec in bad_specs:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                candidate = report()
                candidate["model"] = spec
                quality_gate(report(), candidate, ["class:0"])

    def test_consistent_critical_regression_despite_average_growth(self):
        baseline, candidate = report(), report()
        rows = items()
        # База теряет два ответа вне class:0; кандидат — один внутри class:0.
        for target, wrong_ids in ((baseline, {"1", "2"}), (candidate, {"0"})):
            outputs = [
                dict(
                    id=r["id"],
                    raw='{"answer":"question"}' if r["id"] in wrong_ids else '{"answer":"bug"}',
                    error=None,
                )
                for r in rows
            ]
            target.update(score(rows, outputs))
        self.assertGreater(candidate["overall"]["accuracy"], baseline["overall"]["accuracy"])
        with self.assertRaisesRegex(ValueError, "критического среза class:0"):
            quality_gate(baseline, candidate, ["class:0"])

    def test_consistent_behavior_failure_rejected(self):
        candidate = report()
        behavior = items(4)
        outputs = [dict(id=r["id"], raw='{"answer":"bug"}', error=None) for r in behavior]
        outputs[0]["raw"] = '{"answer":"question"}'
        candidate["behavioral"] = score(behavior, outputs)
        with self.assertRaisesRegex(ValueError, "behavioral"):
            quality_gate(report(), candidate, ["class:0"])

    def test_consistent_generation_error_rejected(self):
        rows = items()
        for role in ("baseline", "candidate"):
            baseline, candidate = report(), report()
            outputs = [dict(id=r["id"], raw='{"answer":"bug"}', error=None) for r in rows]
            outputs[0].update(raw="", error="timeout")
            target = baseline if role == "baseline" else candidate
            target.update(score(rows, outputs))
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "ошибки генерации"):
                quality_gate(baseline, candidate, ["class:0"])

    def test_same_size_but_different_memberships_rejected(self):
        baseline, candidate = report(), report()
        rows = copy.deepcopy(items())
        rows[0]["slices"], rows[1]["slices"] = rows[1]["slices"], rows[0]["slices"]
        candidate.update(score(rows, [dict(id=r["id"], raw='{"answer":"bug"}', error=None) for r in rows]))
        self.assertEqual(baseline["overall"], candidate["overall"])
        self.assertEqual(baseline["slices"], candidate["slices"])
        with self.assertRaisesRegex(ValueError, "состав срезов"):
            quality_gate(baseline, candidate, ["class:0"])

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
