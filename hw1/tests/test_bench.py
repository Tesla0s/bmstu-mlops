"""Регрессия: загрузка и прогрев не должны попадать в скорость генерации."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.bench import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_load_warmup_and_gpu_wait_are_accounted_separately(self):
        clock = [0.0]
        model = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float32,
            config=SimpleNamespace(_commit_hash="test"),
            parameters=lambda: [],
        )
        tokenizer = Mock()
        tokenizer.decode.return_value = "Ответ."
        params = {
            "model": {"name": "test-model"},
            "generate": {},
            "bench": {"prompt": "Вопрос", "warmup_runs": 1, "measure_runs": 3},
        }
        # Прогрев очень дорогой. Длины ответов различаются: считаем реальные токены.
        calls = iter([(20.0, 200), (1.0, 12), (3.0, 25), (2.0, 30)])

        def load(_):
            clock[0] += 50.0
            return tokenizer, model

        def prepare(*_):
            clock[0] += 4.0
            return {"input_ids": torch.ones((1, 4), dtype=torch.long)}

        def generate(*_):
            duration, count = next(calls)
            clock[0] += duration
            return torch.ones(count, dtype=torch.long)

        def wait(_):
            # Имитация незавершённой работы ускорителя.
            clock[0] += 0.25

        with (
            patch("src.bench.load_model", side_effect=load),
            patch("src.bench.prepare_inputs", side_effect=prepare),
            patch("src.bench.generate_tokens", side_effect=generate) as gen,
            patch("src.bench.synchronize", side_effect=wait),
            patch("src.bench.time.perf_counter", side_effect=lambda: clock[0]),
        ):
            report = benchmark(params)

        self.assertEqual(report["load_time_sec"], 50.25)
        self.assertEqual(gen.call_count, 4)
        self.assertEqual([r["elapsed_sec"] for r in report["runs"]], [1.25, 3.25, 2.25])
        self.assertEqual([r["new_tokens"] for r in report["runs"]], [12, 25, 30])
        self.assertEqual(report["tokens_per_sec_all"], [12 / 1.25, 25 / 3.25, 30 / 2.25])
        self.assertAlmostEqual(report["tokens_per_sec"], 9.6)
        tokenizer.decode.assert_called_once()

    def test_missing_warmup_fails_before_model_download(self):
        params = {"bench": {"warmup_runs": 0, "measure_runs": 3}}
        with patch("src.bench.load_model") as load:
            with self.assertRaisesRegex(ValueError, "прогрев"):
                benchmark(params)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
