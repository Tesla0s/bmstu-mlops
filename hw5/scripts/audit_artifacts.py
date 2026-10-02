"""Независимая проверка сохранённых адаптеров, кривых и всех входных примеров."""

import hashlib
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from src.config import load_params
from src.train import inputs_fingerprint


def main():
    p = load_params()
    inputs = {}
    for split, path in p["data"].items():
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == p["provenance"]["tokenized_sha256"][split]
        blob = torch.load(path, weights_only=True, map_location="cpu")
        rows = blob["examples"]
        inputs[split] = {
            "examples": len(rows),
            "input_tokens": sum(len(r["input_ids"]) for r in rows),
            "supervised_tokens": sum(sum(x != -100 for x in r["labels"][1:]) for r in rows),
            "sha256": digest,
        }
    runs = {}
    for variant in p["variants"]:
        name, freeze = variant["name"], variant["freeze_first"]
        m = json.loads(Path(f"metrics/train_{name}.json").read_text())
        assert m["inputs_fingerprint"] == inputs_fingerprint(p)
        assert m["max_steps"] is None and m["val_limit"] is None
        assert (
            m["steps"]
            == m["expected_steps"]
            == math.ceil(inputs["train"]["examples"] / m["effective_batch"]) * p["train"]["epochs"]
        )
        for field, source in [
            ("examples_seen", "examples"),
            ("input_tokens_seen", "input_tokens"),
            ("supervised_tokens_seen", "supervised_tokens"),
        ]:
            assert m[field] == inputs["train"][source] * p["train"]["epochs"]
        assert m["val_examples"] == inputs["val"]["examples"]
        assert [row[0] for row in m["curve_train"]] == list(range(1, m["steps"] + 1))
        expected_steps = sorted(
            {
                0,
                m["steps"],
                *range(p["train"]["eval_every"], m["steps"] + 1, p["train"]["eval_every"]),
            }
        )
        assert [row[0] for row in m["curve_val"]] == expected_steps
        assert all(math.isfinite(v) and v >= 0 for _, v in m["curve_train"] + m["curve_val"])
        assert m["base_val_loss"] > m["final_val_loss"]
        assert m["curve_val"][0][1] == m["base_val_loss"]
        assert m["curve_val"][-1][1] == m["final_val_loss"]
        assert not m["diverged"]
        assert sum(s["examples"] for s in m["step_trace"]) == m["examples_seen"]
        assert sum(s["input_tokens"] for s in m["step_trace"]) == m["input_tokens_seen"]
        assert sum(s["supervised_tokens"] for s in m["step_trace"]) == m["supervised_tokens_seen"]
        expected_last = inputs["train"]["examples"] % m["effective_batch"] or m["effective_batch"]
        assert m["step_trace"][-1]["examples"] == expected_last
        ad = Path(m["adapter_dir"])
        cfg = json.loads((ad / "adapter_config.json").read_text())
        assert cfg["layers_to_transform"] == list(range(freeze, 28))
        assert cfg["revision"] == p["model"]["revision"]
        assert not cfg.get("modules_to_save")
        count, layers, matrices = 0, set(), 0
        actual_slots = set()
        with safe_open(ad / "adapter_model.safetensors", framework="pt", device="cpu") as sf:
            for key in sf.keys():  # noqa: SIM118 — safe_open не является словарём.
                assert "lora_A" in key or "lora_B" in key
                assert "lm_head" not in key and "embed_tokens" not in key
                match = re.search(
                    r"\.layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.(lora_[AB])\.weight$", key
                )
                assert match, key
                layer = int(match[1])
                slot = (layer, match[2], match[3])
                assert slot not in actual_slots
                actual_slots.add(slot)
                layers.add(layer)
                t = sf.get_tensor(key)
                assert torch.isfinite(t).all()
                count += t.numel()
                matrices += 1
        assert layers == set(range(freeze, 28))
        expected_slots = {
            (layer, module, matrix)
            for layer in range(freeze, 28)
            for module in p["lora"]["target_modules"]
            for matrix in ("lora_A", "lora_B")
        }
        assert actual_slots == expected_slots
        assert matrices == (28 - freeze) * 7 * 2
        assert count == m["trainable_params"]
        actual_mib = round(sum(x.stat().st_size for x in ad.rglob("*") if x.is_file()) / 1048576, 2)
        assert actual_mib == m["adapter_size_mb"] < 100
        metadata = json.loads((ad / "training_metadata.json").read_text())
        assert metadata["inputs_fingerprint"] == m["inputs_fingerprint"]
        assert metadata["input_sha256"] == m["input_sha256"]
        assert metadata["model_revision"] == p["model"]["revision"]
        assert metadata["chat_template_sha256"] == p["provenance"]["chat_template_sha256"]
        assert json.loads((ad / "run_config.json").read_text()) == p
        tok = AutoTokenizer.from_pretrained(ad, local_files_only=True)
        assert (
            hashlib.sha256(tok.chat_template.encode()).hexdigest()
            == p["provenance"]["chat_template_sha256"]
        )
        runs[name] = {
            "passed": True,
            "steps": m["steps"],
            "val_points": len(m["curve_val"]),
            "parameters_in_saved_matrices": count,
            "layers": sorted(layers),
            "examples_seen": m["examples_seen"],
            "final_val_loss": m["final_val_loss"],
        }
    assert (
        runs["all_layers"]["parameters_in_saved_matrices"]
        == 2 * runs["freeze14"]["parameters_in_saved_matrices"]
    )
    comparison = json.loads(Path("metrics/compare_all_layers.json").read_text())
    assert comparison["prompts"] == p["compare"]["prompts"]
    assert len(comparison["base"]) == len(comparison["adapter"]) == 5
    result = {"passed": True, "inputs": inputs, "runs": runs, "five_prompts_present": True}
    out = Path(p["paths"]["metrics"]) / "artifact_audit.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, item in runs.items():
        print(
            f"{name}: {item['examples_seen']} примеров, {item['steps']} шагов, "
            f"{item['val_points']} точек val; сохранённые матрицы и слои проверены"
        )
    print(f"Аудит пройден: {out}")


if __name__ == "__main__":
    main()
