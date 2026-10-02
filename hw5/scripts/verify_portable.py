"""Проверить переданную папку без импортов исходного проекта и без сети."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    folder = Path(args.adapter).resolve()
    p = json.loads((folder / "run_config.json").read_text())
    cfg = json.loads((folder / "adapter_config.json").read_text())
    device = p["model"]["device"]
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    dtype = getattr(torch, p["model"]["dtype"])
    torch.set_num_threads(p["train"]["cpu_threads"])
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    template_hash = hashlib.sha256(tok.chat_template.encode()).hexdigest()
    assert template_hash == p["provenance"]["chat_template_sha256"]
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_name_or_path"],
        revision=cfg["revision"],
        local_files_only=True,
        dtype=dtype,
        attn_implementation=p["model"]["attn_implementation"],
    ).to(device)
    model = PeftModel.from_pretrained(base, folder, local_files_only=True).to(device).eval()
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = False
    generation.temperature = generation.top_p = generation.top_k = None
    answers = []
    for question in p["compare"]["prompts"]:
        messages = [
            {"role": "system", "content": p["compare"]["system"]},
            {"role": "user", "content": question},
        ]
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=p["model"]["enable_thinking"],
        )
        inputs = tok(text, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            result = model.generate(
                **inputs,
                generation_config=generation,
                do_sample=False,
                use_model_defaults=False,
                pad_token_id=tok.pad_token_id,
                max_new_tokens=p["compare"]["max_new_tokens"],
            )
        answers.append(
            tok.decode(result[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        )
    reference = json.loads(Path(args.reference).read_text())
    assert len(answers) == len(reference["adapter"]) == 5
    assert answers == reference["adapter"], (answers, reference["adapter"])
    report = {
        "passed": True,
        "answers_identical": 5,
        "adapter": answers,
        "offline": True,
        "no_src_imports": True,
        "template_sha256": template_hash,
        "model_revision": cfg["revision"],
        "device": device,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Папка адаптера автономна: пять ответов дословно совпали, сеть отключена.")


if __name__ == "__main__":
    main()
