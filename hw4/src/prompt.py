"""Единственное место сборки диалога для обучения и получения ответа."""

from typing import Any

from transformers import AutoTokenizer


def load_tokenizer(params: dict):
    tokenizer = AutoTokenizer.from_pretrained(
        params["model"]["name"],
        revision=params["model"]["revision"],
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Для границы маски нужны символьные смещения быстрого токенизатора")
    if not tokenizer.chat_template:
        raise ValueError("У токенизатора нет шаблона диалога")
    tokenizer.padding_side = params["tokenize"]["padding_side"]
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("У токенизатора нет ни pad, ни eos")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def split_messages(messages: list[dict]) -> tuple[list[dict], dict]:
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Последняя реплика обучающего диалога должна принадлежать ассистенту")
    return messages[:-1], messages[-1]


def build_chat_text(
    tokenizer: Any,
    messages: list[dict],
    params: dict,
    add_generation_prompt: bool,
) -> str:
    """При генерации принимает как полный пример, так и ещё не отвеченный запрос."""
    if not messages:
        raise ValueError("Пустой диалог")
    if add_generation_prompt:
        if messages[-1]["role"] == "assistant":
            messages, _ = split_messages(messages)
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("Промпт должен заканчиваться сообщением пользователя")
    else:
        split_messages(messages)
    value = params["model"].get("enable_thinking")
    kwargs = {} if value is None else {"enable_thinking": value}
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def prompt_token_len(
    tokenizer: Any,
    prompt_text: str,
    full_ids: list[int],
    full_offsets: list[tuple[int, int]],
) -> tuple[int, bool]:
    """Проверить префикс токенов; при склейке на границе использовать смещения."""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    n = len(prompt_ids)
    if full_ids[:n] == prompt_ids:
        return n, False
    if len(full_offsets) != len(full_ids):
        raise ValueError("Число символьных смещений не совпадает с числом токенов")
    boundary = len(prompt_text)
    for i, (start, end) in enumerate(full_offsets):
        if start >= boundary and end > start:
            return i, True
    # Весь ответ мог склеиться с последним токеном промпта.
    # Такой пример не должен получать выдуманную границу или обучающий сигнал.
    return len(full_ids), True
