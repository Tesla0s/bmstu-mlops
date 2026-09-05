"""Общая загрузка модели, подготовка вопроса и генерация."""

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Выбрана cuda, но графический процессор NVIDIA недоступен.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Выбрана mps, но графический процессор Apple недоступен.")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    """Дождаться вычислений устройства перед чтением таймера."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_model(params: dict):
    set_seed(params["generate"]["seed"])
    name = params["model"]["name"]
    device = resolve_device(params["model"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=getattr(torch, params["model"]["dtype"]),
        device_map={"": str(device)},
    )
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, params: dict, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=params["generate"]["enable_thinking"],
    )


def prepare_inputs(tokenizer, model, params: dict, text: str):
    prompt = build_prompt(tokenizer, params, text)
    # Шаблон диалога уже содержит специальные токены.
    return tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)


def generate_tokens(model, params: dict, inputs) -> torch.Tensor:
    """Вернуть только новые токены; перевод в текст выполняется отдельно."""
    temperature = params["generate"]["temperature"]
    options = {
        "max_new_tokens": params["generate"]["max_new_tokens"],
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        options["temperature"] = temperature
    else:
        # Убираем параметры случайного выбора, унаследованные от модели.
        # Нейтральные значения берём из библиотеки, не меняя EOS/PAD модели.
        defaults = GenerationConfig()
        for key in ("temperature", "top_p", "top_k"):
            options[key] = getattr(defaults, key)
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    return output[0, inputs["input_ids"].shape[1]:]


def generate(tokenizer, model, params: dict, text: str) -> tuple[str, int]:
    inputs = prepare_inputs(tokenizer, model, params, text)
    new_tokens = generate_tokens(model, params, inputs)
    return tokenizer.decode(new_tokens.cpu(), skip_special_tokens=True), len(new_tokens)
