"""Независимые проверки исправленных дефектов без скачивания модели."""
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.inspect_model import (
    PeakMemory, activation_norms, group_table, memory_profile, parameter_rows,
)


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "sample"

    def __call__(self, *args, **kwargs):
        return Inputs(input_ids=torch.ones(1, 3, dtype=torch.long))


class TinyModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5))
            for _ in range(3)
        ])
        self.device = torch.device("cpu")
        self.fail = fail

    def get_decoder(self):
        return self

    def forward(self, input_ids, **kwargs):
        hidden = torch.ones(*input_ids.shape, 4)
        for layer in self.layers:
            hidden = layer(hidden)
            if self.fail:
                raise RuntimeError("искусственная ошибка forward")
        return hidden


class AnatomyTests(unittest.TestCase):
    def test_shared_weight_is_counted_once_and_keeps_both_names(self):
        model = torch.nn.Module()
        model.embed_tokens = torch.nn.Embedding(5, 4)
        model.lm_head = torch.nn.Linear(4, 5, bias=False)
        model.lm_head.weight = model.embed_tokens.weight
        rows = parameter_rows(model)
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(r["params"] for r in group_table(rows)), 20)
        head = next(r for r in rows if r["name"] == "lm_head.weight")
        self.assertTrue(head["tied"])
        self.assertEqual(head["shared_with"], "embed_tokens.weight")

    def test_hooks_repeat_and_preserve_preexisting_hook_and_training_state(self):
        model = TinyModel()
        model.layers[1].eval()  # Сохраняется даже смешанный исходный режим.
        states = [m.training for m in model.modules()]
        handle = model.layers[0].register_forward_hook(lambda *args: None)
        params = {"hooks": {"prompt": "sample"}, "generate": {}}
        try:
            first = activation_norms(Tokenizer(), model, params)
            second = activation_norms(Tokenizer(), model, params)
            self.assertEqual(first["norms"], second["norms"])
            self.assertEqual(first["hooks_before"], 1)
            self.assertEqual(second["hooks_after"], 1)
            self.assertEqual(states, [m.training for m in model.modules()])
        finally:
            handle.remove()

    def test_hooks_and_state_are_restored_when_forward_raises(self):
        model = TinyModel(fail=True)
        states = [m.training for m in model.modules()]
        with self.assertRaisesRegex(RuntimeError, "искусственная"):
            activation_norms(Tokenizer(), model, {"hooks": {"prompt": "sample"}, "generate": {}})
        self.assertEqual(sum(len(m._forward_hooks) for m in model.modules()), 0)
        self.assertEqual(states, [m.training for m in model.modules()])

    def test_peak_keeps_transient_allocation_after_it_was_freed(self):
        with patch("src.inspect_model.device_allocated_bytes", side_effect=[10, 90, 20]):
            with PeakMemory(torch.device("cpu")) as peak:
                peak.sample()
            self.assertEqual(peak.used, 90)

    def test_every_mode_and_repeat_uses_a_new_process_and_passed_config(self):
        calls = []
        params = {"memory": {"repeats": 2, "seq_len": 17}, "model": {"name": "from-memory"}}

        def run(args, **kwargs):
            calls.append((args, json.loads(kwargs["input"])))
            self.assertIn("--params-stdin", args)
            self.assertEqual(kwargs["timeout"], 300)
            payload = {"peak_bytes": len(calls) * 1024, "peak_mb": len(calls), "pid": len(calls)}
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

        with patch("src.inspect_model.subprocess.run", side_effect=run):
            result = memory_profile(params)
        self.assertEqual(len(calls), 6)
        self.assertEqual([c[0][4] for c in calls], ["inference"] * 2 + ["full_ft"] * 2 + ["lora"] * 2)
        self.assertTrue(all(c[1] == params for c in calls))
        self.assertEqual([r["peak_mb"] for r in result], [2, 4, 6])

    def test_failed_child_is_not_reported_as_a_measurement(self):
        with patch("src.inspect_model.subprocess.run", return_value=SimpleNamespace(
                returncode=1, stderr="device failure", stdout="")):
            with self.assertRaisesRegex(RuntimeError, "device failure"):
                memory_profile({"memory": {"repeats": 1}})


if __name__ == "__main__":
    unittest.main()
