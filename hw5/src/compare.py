"""Пять фиксированных запросов к базе и адаптеру."""

import argparse
import copy
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_params
from src.runtime import resolve_device, resolve_dtype, set_seed


def load_adapter_tokenizer(adapter_dir: Path):
    tok = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    if not tok.chat_template:
        raise ValueError("В папке адаптера нет шаблона диалога")
    return tok


@torch.no_grad()
def generate(model, tok, prompts: list[str], system: str, params: dict, device) -> list[str]:
    model.set_attn_implementation(params["model"]["attn_implementation"])
    model.eval()
    out = []
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = False
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    for user in prompts:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=params["model"]["enable_thinking"],
        )
        ids = tok(text, add_special_tokens=False, return_tensors="pt").to(device)
        generated = model.generate(
            **ids,
            generation_config=generation,
            do_sample=False,
            use_model_defaults=False,
            max_new_tokens=params["compare"]["max_new_tokens"],
            pad_token_id=tok.pad_token_id,
        )
        out.append(
            tok.decode(generated[0, ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        )
    return out


def categories(answers, *, allow_markdown=False):
    result = []
    for text in answers:
        if allow_markdown and text.startswith("```json\n") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(text)
            category = parsed.get("category") if set(parsed) == {"category"} else None
            result.append(category if category in ("bug", "feature_request", "question") else None)
        except (ValueError, TypeError, AttributeError):
            result.append(None)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all_layers")
    ap.add_argument("--adapter-dir")
    ap.add_argument("--only-adapter", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    params = load_params()
    torch.set_num_threads(params["train"]["cpu_threads"])
    set_seed(params["train"]["seed"])
    device, dtype = (
        resolve_device(params["model"]["device"]),
        resolve_dtype(params["model"]["dtype"]),
    )
    adapter_dir = Path(
        args.adapter_dir or Path(params["paths"]["models"]) / f"adapter_{args.variant}"
    )
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg["revision"] != params["model"]["revision"]:
        raise ValueError("Версия базы в адаптере отличается от конфигурации")
    tok = load_adapter_tokenizer(adapter_dir)
    prompts, system = params["compare"]["prompts"], params["compare"]["system"]
    base = (
        AutoModelForCausalLM.from_pretrained(
            cfg["base_model_name_or_path"],
            revision=cfg["revision"],
            dtype=dtype,
            attn_implementation=params["model"]["attn_implementation"],
        )
        .to(device)
        .eval()
    )
    before = [] if args.only_adapter else generate(base, tok, prompts, system, params, device)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
    after = generate(model, tok, prompts, system, params, device)
    expected = params["compare"]["expected_categories"]
    result = {
        "variant": args.variant,
        "adapter_dir": str(adapter_dir),
        "model_revision": cfg["revision"],
        "prompts": prompts,
        "system": system,
        "base": before,
        "adapter": after,
        "expected_categories": expected,
        "base_categories": categories(before),
        "adapter_categories": categories(after),
        "base_semantic_categories": categories(before, allow_markdown=True),
        "adapter_semantic_categories": categories(after, allow_markdown=True),
        "prompt_origin": params["compare"]["prompt_origin"],
    }
    out = (
        Path(args.out)
        if args.out
        else Path(params["paths"]["metrics"]) / f"compare_{args.variant}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.out:
        print(f"-> {out}")
        return
    fence = chr(96) * 4
    lines = [
        "# Базовая модель и LoRA: пять заранее заданных обращений GitHub",
        "",
        (
            f"Вариант: {args.variant}. Генерация без случайного выбора, до "
            f"{params['compare']['max_new_tokens']} новых токенов. Версия базы: {cfg['revision']}."
        ),
        "",
        (
            "Пять искусственных обращений зафиксированы в params.yaml до обучения. "
            "Это демонстрация подключения адаптера и формата ответа, не оценка качества "
            "на репрезентативной выборке. Итоговая тестовая часть ДЗ3 не использована."
        ),
        "",
        "Оба варианта получают одну и ту же системную инструкцию и шаблон из папки адаптера.",
        "",
        "**Системная инструкция:**",
        "",
        "\n".join("> " + line if line else ">" for line in system.split("\n")),
        "",
    ]
    for i, (question, b, a, target) in enumerate(
        zip(prompts, before, after, expected, strict=True), 1
    ):
        lines += [
            f"## {i}. Ожидаемая категория: {target}",
            "",
            "\n".join("> " + line if line else ">" for line in question.split("\n")),
            "",
            "**Базовая модель:**",
            "",
            fence + "text",
            b,
            fence,
            "",
            "**С адаптером:**",
            "",
            fence + "text",
            a,
            fence,
            "",
        ]
    b_correct = sum(x == y for x, y in zip(categories(before), expected))
    a_correct = sum(x == y for x, y in zip(categories(after), expected))
    b_semantic = sum(x == y for x, y in zip(categories(before, allow_markdown=True), expected))
    a_semantic = sum(x == y for x, y in zip(categories(after, allow_markdown=True), expected))
    lines += [
        "## Что показывает эта проверка",
        "",
        (
            f"Правильная категория вместе со строгим форматом ответа (чистый JSON без Markdown): "
            f"база {b_correct}/5, адаптер {a_correct}/5."
        ),
        (
            f"Если убрать только обрамление блока кода Markdown, категории совпадают с ожидаемыми: "
            f"база {b_semantic}/5, адаптер {a_semantic}/5. Это отдельная проверка содержания: "
            "нарушение формата не следует выдавать за ошибку выбора категории."
        ),
        (
            "Пять примеров не доказывают статистически значимого улучшения. "
            "Изменение val loss на всех 281 отложенном примере показано в отчёте обучения."
        ),
        "",
    ]
    path = Path(params["paths"]["compare"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    for b, a in zip(before, after):
        print(f"база: {b!r}\nадаптер: {a!r}")
    print(f"-> {out}, {path}")


if __name__ == "__main__":
    main()
