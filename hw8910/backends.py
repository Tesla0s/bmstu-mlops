"""The HTTP layer uses the same interface for explicit fixtures and the real Qwen adapter."""

from importlib.metadata import version
import time
from pathlib import Path
import random
import release
from artifacts import artifact_root, verify_files

LIBRARIES = ("torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors", "huggingface-hub")


def accepted_record(root, manifest):
    for path in (root / "accepted-history").glob("*.json"):
        record = release.read(path)
        if record["manifest"] == manifest:
            release.validate_release(manifest, record["report"])
            return record
    raise ValueError("Accepted pointer has no verified decision in release history; recover the publisher")


class StubBackend:
    def __init__(self, manifest, root):
        self.loaded_id = manifest["model_id"]
        self.runtime_sha256 = manifest["runtime_sha256"]
        self.calls = 0

    def warmup(self):
        pass

    def validate(self, prompt):
        return prompt

    def generate(self, prompt, limit):
        self.calls += 1
        time.sleep(0.01)
        return {"answer": f"STUB ONLY: {self.loaded_id}", "output_tokens": 1}


class TransformersBackend:
    def __init__(self, manifest, root):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        runtime = manifest["runtime"]
        for name, expected in runtime["libraries"].items():
            if version(name) != expected:
                raise ValueError(f"Runtime library changed: {name}; re-evaluate before release")
        self.torch = torch
        self.runtime = runtime
        self.loaded_id = manifest["model_id"]
        self.runtime_sha256 = manifest["runtime_sha256"]
        self.calls = 0
        self.device = runtime["device"]
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("This release was evaluated on MPS; evaluate another runtime for this device")
        torch.set_num_threads(4)
        random.seed(runtime["generation"]["seed"])
        torch.manual_seed(runtime["generation"]["seed"])
        import numpy as np

        np.random.seed(runtime["generation"]["seed"])
        adapter = artifact_root(manifest, root)
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            runtime["base_model"],
            revision=runtime["base_revision"],
            dtype=getattr(torch, runtime["dtype"]),
            attn_implementation=runtime["attention"],
        )
        self.model = PeftModel.from_pretrained(base, adapter).to(self.device).eval()

    def validate(self, prompt):
        cfg = self.runtime["generation"]
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": cfg["system"]}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg["enable_thinking"],
        )
        batch = self.tokenizer(text, return_tensors="pt")
        if batch["input_ids"].shape[1] > 2048:
            raise ValueError("Input exceeds 2048 tokens including the system message")
        return batch.to(self.device)

    def synchronize(self):
        if self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def generate(self, batch, limit):
        self.calls += 1
        with self.torch.inference_mode():
            out = self.model.generate(
                **batch,
                max_new_tokens=limit,
                do_sample=False,
                use_model_defaults=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        self.synchronize()
        tokens = out[0, batch["input_ids"].shape[1] :]
        return {
            "answer": self.tokenizer.decode(tokens, skip_special_tokens=True),
            "output_tokens": len(tokens),
        }

    def warmup(self):
        batch = self.validate("Title: Editor crashes. Description: Saving a file closes the editor.")
        self.generate(batch, self.runtime["generation"]["max_new_tokens"])


def load_backend(root):
    root = Path(root)
    manifest = release.read(root / "accepted.json")
    accepted_record(root, manifest)
    verify_files(manifest, root)
    if manifest.get("backend") == "transformers":
        return TransformersBackend(manifest, root)
    if manifest["runtime"].get("base_revision") != "fixture-only":
        raise ValueError("Unknown backend")
    return StubBackend(manifest, root)
