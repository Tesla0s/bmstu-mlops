"""Проверки потерь, градиентов, последней группы и состава адаптеров."""

import unittest

import torch
from peft import get_peft_model
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

from src.compare import generate
from src.config import load_params
from src.data import accumulation_groups, batches, pad_batch
from src.loss import supervised_loss_sum
from src.runtime import set_seed
from src.train import lora_config


def tiny_model():
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=37,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
    )


def examples():
    return [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1] * 5,
            "labels": [-100, -100, -100, 4, 5],
        },
        {"input_ids": [6, 7, 8], "attention_mask": [1] * 3, "labels": [-100, 7, 8]},
    ]


class LossTests(unittest.TestCase):
    def test_selected_positions_match_full_loss_and_gradients(self):
        set_seed(42)
        model = tiny_model().eval()
        batch = pad_batch(examples(), 0)
        full = model(**batch).loss
        full.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        selected, count = supervised_loss_sum(model, batch)
        self.assertEqual(count, 4)
        torch.testing.assert_close(selected / count, full, rtol=1e-6, atol=1e-6)
        (selected / count).backward()
        for n, p in model.named_parameters():
            torch.testing.assert_close(p.grad, gradients[n], rtol=2e-5, atol=1e-6)

    def test_accumulated_token_weighted_gradient_matches_whole_batch(self):
        set_seed(42)
        model = tiny_model().eval()
        rows = examples()
        rows[0]["labels"][2] = 3
        full, count = supervised_loss_sum(model, pad_batch(rows, 0))
        (full / count).backward()
        expected = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        for row in rows:
            value, _ = supervised_loss_sum(model, pad_batch([row], 0))
            (value / count).backward()
        for n, p in model.named_parameters():
            torch.testing.assert_close(p.grad, expected[n], rtol=2e-5, atol=1e-6)

    def test_no_supervision_is_rejected(self):
        model = tiny_model().eval()
        batch = pad_batch(examples(), 0)
        batch["labels"].fill_(-100)
        with self.assertRaises(ValueError):
            supervised_loss_sum(model, batch)


class DataTests(unittest.TestCase):
    def test_last_incomplete_accumulation_group_is_retained(self):
        rows = examples() * 5 + examples()[:1]
        groups = list(accumulation_groups(batches(rows, 2, 0, False, 0), 4))
        self.assertEqual([sum(len(b["input_ids"]) for b in g) for g in groups], [8, 3])

    def test_padding_never_masks_a_real_eos_equal_to_pad(self):
        batch = pad_batch(examples(), 8)
        self.assertEqual(batch["input_ids"][1].tolist(), [8, 8, 6, 7, 8])
        self.assertEqual(batch["labels"][1].tolist(), [-100, -100, -100, 7, 8])
        self.assertEqual(batch["attention_mask"][1].tolist(), [0, 0, 1, 1, 1])


class AdapterTests(unittest.TestCase):
    def test_layer_selection_reduces_trainable_parameters(self):
        p = load_params()
        totals = []
        for freeze in (0, 1):
            model = get_peft_model(tiny_model(), lora_config(p, 2, freeze))
            names = [n for n, v in model.named_parameters() if v.requires_grad]
            self.assertTrue(all("lora_" in n for n in names))
            if freeze:
                self.assertTrue(all(".layers.1." in n for n in names))
            totals.append(sum(v.numel() for v in model.parameters() if v.requires_grad))
        self.assertEqual(totals[0], totals[1] * 2)

    def test_seed_controls_initialization_and_dropout(self):
        values = []
        for _ in range(2):
            set_seed(42)
            layer = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Dropout(0.5)).train()
            values.append(layer(torch.ones(2, 5)))
        torch.testing.assert_close(*values, rtol=0, atol=0)


class GenerationTests(unittest.TestCase):
    def test_model_defaults_cannot_enable_random_sampling(self):
        class TinyTokenizer:
            pad_token_id = 0

            def apply_chat_template(self, *_args, **_kwargs):
                return "fixed input"

            def __call__(self, *_args, **_kwargs):
                return BatchEncoding(
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "attention_mask": torch.ones((1, 3), dtype=torch.long),
                    }
                )

            def decode(self, ids, **_kwargs):
                return " ".join(str(x) for x in ids.tolist())

        set_seed(42)
        model = tiny_model().eval()
        model.generation_config.do_sample = True
        model.generation_config.transformers_version = "4.57.6"
        params = {
            "model": {"attn_implementation": "eager", "enable_thinking": False},
            "compare": {"max_new_tokens": 2},
        }
        before = torch.get_rng_state().clone()
        first = generate(
            model, TinyTokenizer(), ["question"], "system", params, torch.device("cpu")
        )
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        set_seed(999)
        second = generate(
            model, TinyTokenizer(), ["question"], "system", params, torch.device("cpu")
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
