"""JSONL из ДЗ 3 -> токены, маска ответа, измерения и автоматически созданный отчёт."""

import hashlib
import importlib.metadata
import io
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import torch

from src.collate import LABEL_PAD_ID
from src.config import load_params
from src.prompt import build_chat_text, load_tokenizer, prompt_token_len

METRICS_PATH = Path("metrics/tokenize.json")
REPORT_PATH = Path("docs/tokenize_report.md")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Нет {path}: выполните make prepare или восстановите файлы через DVC")
    records, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {"id", "topic", "messages"}:
                    raise ValueError("ожидаются только id, topic, messages")
                for key in ("id", "topic"):
                    if not isinstance(row[key], str) or not row[key].strip():
                        raise ValueError(f"{key} должен быть непустой строкой")
                messages = row["messages"]
                if not isinstance(messages, list) or len(messages) != 3:
                    raise ValueError("нужны три сообщения: system, user, assistant")
                for message, role in zip(messages, ("system", "user", "assistant"), strict=True):
                    if not isinstance(message, dict) or set(message) != {"role", "content"}:
                        raise ValueError("у сообщения должны быть role и content")
                    if message["role"] != role:
                        raise ValueError("порядок ролей должен быть system, user, assistant")
                    if not isinstance(message["content"], str) or not message["content"].strip():
                        raise ValueError("содержимое сообщения должно быть непустой строкой")
                if row["id"] in seen:
                    raise ValueError(f"повторный id: {row['id']}")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            seen.add(row["id"])
            records.append(row)
    if not records:
        raise ValueError(f"{path}: пустой датасет")
    return records


def mask_prompt(input_ids: list[int], n_prompt: int) -> list[int]:
    if n_prompt < 0:
        raise ValueError("Граница промпта не может быть отрицательной")
    n = min(n_prompt, len(input_ids))
    return [LABEL_PAD_ID] * n + list(input_ids[n:])


def encode_example(tokenizer, record: dict, params: dict, max_seq_len: int) -> dict:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len должен быть положительным")
    messages = record["messages"]
    full_text = build_chat_text(tokenizer, messages, params, add_generation_prompt=False)
    prompt_text = build_chat_text(tokenizer, messages, params, add_generation_prompt=True)
    if not full_text.encode("utf-8").startswith(prompt_text.encode("utf-8")):
        raise ValueError(f"{record['id']}: промпт не является побайтовым префиксом диалога")
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encoded["input_ids"]
    n_prompt, fallback = prompt_token_len(
        tokenizer, prompt_text, full_ids, encoded["offset_mapping"]
    )
    full_answer_ids = full_ids[n_prompt:]
    wanted = messages[-1]["content"].strip()
    decoded = tokenizer.decode(full_answer_ids, skip_special_tokens=True).strip()
    # При BPE-склейке первый токен ответа может частично остаться под маской.
    # В штатном шаблоне текущего набора такого нет; доля отдельно измеряется.
    if not fallback and decoded != wanted:
        raise ValueError(
            f"{record['id']}: под маской находится не только ответ ассистента; "
            "проверьте enable_thinking и шаблон"
        )
    if tokenizer.eos_token_id not in full_answer_ids:
        raise ValueError(f"{record['id']}: у полного ответа отсутствует токен конца реплики")
    input_ids = full_ids[:max_seq_len]
    labels = mask_prompt(input_ids, n_prompt)
    supervised = [token for token, label in zip(input_ids, labels, strict=True) if label != -100]
    return {
        "id": record["id"],
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "_meta": {
            "id": record["id"],
            "full_len": len(full_ids),
            "prompt_len": n_prompt,
            "answer_len": len(full_ids) - n_prompt,
            "truncated": len(full_ids) > max_seq_len,
            "bpe_fallback": fallback,
            "supervised": len(supervised),
            "answer_complete": (
                tokenizer.eos_token_id in supervised
                and tokenizer.decode(supervised, skip_special_tokens=True).strip() == wanted
            ),
        },
    }


def describe(values: list[int]) -> dict:
    if not values:
        raise ValueError("Нельзя посчитать распределение пустого набора")
    a = np.asarray(values)
    # Для выбора предела целое значение округляется вверх, а не занижается.
    return {
        "count": len(values),
        "min": int(a.min()),
        "mean": round(float(a.mean()), 2),
        **{f"p{p}": int(math.ceil(float(np.percentile(a, p)))) for p in (50, 90, 95, 99)},
        "max": int(a.max()),
    }


def truncation_stats(metas: list[dict], name: str, params: dict) -> dict:
    total = len(metas)
    count = sum(m["truncated"] for m in metas)
    ratio = count / total if total else 0.0
    threshold = params["tokenize"]["truncated_warn_ratio"]
    if ratio > threshold:
        warnings.warn(
            f"{name}: обрезано {count}/{total} ({ratio:.2%}), порог {threshold:.2%}",
            stacklevel=2,
        )
    return {"truncated": count, "truncated_ratio": ratio, "truncation_passed": ratio <= threshold}


def padding_cost(examples: list[dict], params: dict) -> dict:
    size = params["batch"]["size"]
    if not isinstance(size, int) or size <= 0:
        raise ValueError("batch.size должен быть положительным целым числом")
    lengths = [len(e["input_ids"]) for e in examples]
    useful = sum(lengths)
    shuffled = lengths.copy()
    random.Random(params["batch"]["seed"]).shuffle(shuffled)

    def dynamic(order):
        return sum(
            max(order[i : i + size]) * len(order[i : i + size]) for i in range(0, len(order), size)
        )

    slots = {
        "fixed_max_seq_len": len(lengths) * params["tokenize"]["max_seq_len"],
        "dynamic_shuffled": dynamic(shuffled),
        "dynamic_sorted": dynamic(sorted(lengths)),
    }
    return {
        "batch_size": size,
        "seed": params["batch"]["seed"],
        "batches": math.ceil(len(lengths) / size),
        "useful_tokens": useful,
        "strategies": {
            name: {
                "positions": n,
                "padding": n - useful,
                "padding_ratio": (n - useful) / n if n else 0.0,
            }
            for name, n in slots.items()
        },
    }


def process_split(tokenizer, name: str, path: Path, params: dict):
    records = read_jsonl(path)
    examples, metas = [], []
    for record in records:
        example = encode_example(tokenizer, record, params, params["tokenize"]["max_seq_len"])
        meta = example.pop("_meta")
        metas.append(meta)
        if meta["supervised"] > 0:
            examples.append(example)
    stats = {
        "examples_in": len(records),
        "examples_kept": len(examples),
        "dropped_no_supervision": len(records) - len(examples),
        "partial_answers_kept": sum(
            m["supervised"] > 0 and not m["answer_complete"] for m in metas
        ),
        "complete_answers": sum(m["answer_complete"] for m in metas),
        "length_tokens": describe([m["full_len"] for m in metas]),
        "prompt_tokens": describe([m["prompt_len"] for m in metas]),
        "answer_tokens": describe([m["answer_len"] for m in metas]),
        "bpe_boundary_fallback": sum(m["bpe_fallback"] for m in metas),
        "bpe_boundary_fallback_ratio": sum(m["bpe_fallback"] for m in metas) / len(metas),
        "tokens_before_truncation": sum(m["full_len"] for m in metas),
        "answer_tokens_before_truncation": sum(m["answer_len"] for m in metas),
        "total_tokens": sum(len(e["input_ids"]) for e in examples),
        "supervised_tokens": sum(m["supervised"] for m in metas),
        **truncation_stats(metas, name, params),
        "padding": padding_cost(examples, params),
        "packing": None,
    }
    print(
        f"{name}: {len(records)} -> {len(examples)} примеров; "
        f"токенов {stats['total_tokens']}, в лосс {stats['supervised_tokens']}; "
        f"p50/p90/p99/max = "
        f"{stats['length_tokens']['p50']}/{stats['length_tokens']['p90']}/"
        f"{stats['length_tokens']['p99']}/{stats['length_tokens']['max']}; "
        f"обрезано {stats['truncated']} ({stats['truncated_ratio']:.2%})"
    )
    return examples, stats, metas


def estimate_train_time(total_tokens: int, params: dict) -> dict:
    cfg = params["train_estimate"]
    path = Path(cfg["benchmark_json"])
    benchmark = json.loads(path.read_text())
    if benchmark["model"] != params["model"]["name"]:
        raise ValueError("Модель измерения ДЗ 1 не совпадает с выбранной моделью")
    if benchmark["model_revision"] != params["model"]["revision"]:
        raise ValueError("Версия модели отличается от измерения ДЗ 1")
    tps = benchmark["tokens_per_sec"]
    if not isinstance(tps, (float, int)) or not math.isfinite(tps) or tps <= 0:
        raise ValueError("Скорость из ДЗ 1 должна быть положительным конечным числом")
    lo, hi = cfg["training_slowdown_min"], cfg["training_slowdown_max"]
    if not 1 <= lo <= hi:
        raise ValueError("Ожидаются положительные упорядоченные множители замедления")
    seconds = total_tokens * cfg["epochs"] / tps
    return {
        "epochs": cfg["epochs"],
        "tokens_per_sec": tps,
        "tokens_per_epoch": total_tokens,
        "seconds": round(seconds, 2),
        "hours": round(seconds / 3600, 3),
        "training_slowdown_assumption": [lo, hi],
        "training_hours_range": [round(seconds * x / 3600, 3) for x in (lo, hi)],
        "benchmark_path": str(path),
        "benchmark_sha256": sha256(path),
        "benchmark_device": benchmark["device"],
        "benchmark_measured_at": benchmark["measured_at"],
        "note": (
            "Скорость — фактический замер ГЕНЕРАЦИИ из ДЗ 1. Умножение времени "
            "на 2–3 — допущение из задания, не измерение обучения. Прогноз использует "
            "все полезные входные токены, не только позиции с метками. "
            "Паддинг, проверка модели, загрузка и сохранение дополнительно требуют времени. "
            "Оценка относится к машине и модели ДЗ 1, не к GPU Kaggle. "
            "Реальное время дообучения будет измерено в ДЗ 5."
        ),
    }


def render_report(metrics: dict) -> str:
    lines = [
        "# Отчёт стадии tokenize",
        "",
        "Создан программой из входных JSONL и параметров; вручную не редактируется.",
        "",
        f"- Модель: {metrics['model']}.",
        f"- Закреплённая версия токенизатора: {metrics['model_revision']}.",
        f"- Режим рассуждений: {metrics['enable_thinking']}; дополнение батчей: слева.",
        f"- Предел длины: {metrics['max_seq_len']} токенов; допустимая доля обрезки: "
        f"{metrics['truncated_warn_ratio']:.1%}.",
        f"- Итог проверки стадии: {'пройдена' if metrics['passed'] else 'НЕ ПРОЙДЕНА'}.",
        "",
        "## Вход и связь с ДЗ 3",
        "",
        f"Версия данных: {metrics['source']['git_tag']}, коммит "
        f"{metrics['source']['git_commit']}. Файлы копируются без изменения байтов. "
        "Итоговая тестовая часть ДЗ 3 здесь не используется.",
        "",
    ]
    for name, digest in metrics["input_sha256"].items():
        lines += [f"- SHA-256 {name}: {digest}."]
    lines += [
        "",
        "## Длины до обрезки",
        "",
        "Перцентили рассчитаны по всем входным записям, включая записи, "
        "которые не поместились бы в предел. Значения округлены вверх до целого токена.",
        "",
        "| Часть | Примеров | p50 | p90 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in metrics["splits"].items():
        d = s["length_tokens"]
        lines.append(
            f"| {name} | {s['examples_in']} | {d['p50']} | {d['p90']} | "
            f"{d['p95']} | {d['p99']} | {d['max']} |"
        )
    lines += [
        "",
        "## Обрезка и сохранность ответов",
        "",
        "| Часть | Обрезано | Доля | Без ответа, удалено | Неполных ответов сохранено | Осталось |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in metrics["splits"].items():
        lines.append(
            f"| {name} | {s['truncated']} | {s['truncated_ratio']:.3%} | "
            f"{s['dropped_no_supervision']} | {s['partial_answers_kept']} | "
            f"{s['examples_kept']} |"
        )
    lines += [
        "",
        "Обрезка применяется справа. Если после неё нет ни одной обучаемой позиции, "
        "пример удаляется; все такие случаи остаются в знаменателе доли обрезки. "
        "При превышении порога стадия завершается с ошибкой и не записывает новые тензоры.",
        "",
        "## Какие позиции дают обучающую ошибку",
        "",
        "| Часть | Всего токенов | В лосс | Доля в лосс | Средний промпт | Средний ответ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in metrics["splits"].items():
        share = s["supervised_tokens"] / s["total_tokens"] if s["total_tokens"] else 0.0
        lines.append(
            f"| {name} | {s['total_tokens']} | {s['supervised_tokens']} | {share:.3%} | "
            f"{s['prompt_tokens']['mean']} | {s['answer_tokens']['mean']} |"
        )
    lines += [
        "",
        "На промпте labels = −100. Обучаемые позиции содержат ответ ассистента, "
        "токен конца реплики и завершающий перевод строки штатного шаблона. "
        "Служебный пустой блок рассуждений входит в промпт и замаскирован. "
        "Модель при обучении читает все входные токены; маска не убирает их из вычисления.",
        "",
        "## Проверка единого шаблона и границы",
        "",
        "Оба пути используют один модуль src/prompt.py. Для каждого диалога проверяется, "
        "что строка запроса является побайтовым префиксом строки обучения. "
        "Затем сравниваются токены префикса. Если алгоритм объединения частых фрагментов "
        "(BPE) склеил границу, используется начало токена в исходной строке; "
        "пересекающий границу токен остаётся под маской.",
        "",
    ]
    for name, s in metrics["splits"].items():
        lines.append(
            f"- {name}: запасной путь {s['bpe_boundary_fallback']} из {s['examples_in']} "
            f"({s['bpe_boundary_fallback_ratio']:.3%})."
        )
    lines += [
        "",
        "Нулевое число срабатываний на этом наборе не означает, что запасной путь "
        "не нужен: его отдельно проверяют на примере со склейкой внутри слова.",
        "",
        "## Цена дополнения батчей",
        "",
        "Это подсчёт позиций, а не замер ускорения. Полная сортировка по длине показана "
        "для сравнения; она меняет порядок примеров. При обучении порядок следует "
        "перемешивать или группировать по длине внутри перемешанного буфера. "
        "Последний неполный батч не выбрасывается.",
        "",
        "| Часть | Способ | Позиций | Из них дополнение | Доля дополнения |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        "fixed_max_seq_len": "До заданного предела",
        "dynamic_shuffled": "Динамически, перемешивание",
        "dynamic_sorted": "Динамически, сортировка",
    }
    for name, s in metrics["splits"].items():
        for method, p in s["padding"]["strategies"].items():
            lines.append(
                f"| {name} | {labels[method]} | {p['positions']} | {p['padding']} | "
                f"{p['padding_ratio']:.2%} |"
            )
    lines += [
        "",
        f"Размер батча: {metrics['batch']['size']}, seed: {metrics['batch']['seed']}. "
        "Коллатор добавляет pad_token_id слева, attention_mask = 0 и labels = −100. "
        "Тензоры в файле имеют исходную переменную длину; дополнение происходит при сборке батча.",
        "",
        "## Прогноз времени обучения",
        "",
    ]
    est = metrics["train_time_estimate"]
    lines += [
        f"Формула: {est['tokens_per_epoch']} токенов × {est['epochs']} эпохи / "
        f"{est['tokens_per_sec']:.5f} токена/с = {est['seconds']:.2f} с "
        f"({est['hours']:.3f} ч) по скорости генерации.",
        "",
        f"При допущении замедления обучения в {est['training_slowdown_assumption'][0]}–"
        f"{est['training_slowdown_assumption'][1]} раза: "
        f"{est['training_hours_range'][0]:.3f}–{est['training_hours_range'][1]:.3f} ч.",
        "",
        est["note"],
        "",
        f"Исходный замер: {est['benchmark_path']}, устройство {est['benchmark_device']}, "
        f"дата {est['benchmark_measured_at']}.",
        "",
        "## Дополнительная упаковка",
        "",
        "Отключена. Несколько диалогов не объединяются в одну последовательность. "
        "Маска, запрещающая внимание между такими диалогами, не реализована; "
        "одного перезапуска номеров позиций для такой изоляции недостаточно.",
        "",
        "## Воспроизводимость",
        "",
        f"SHA-256 шаблона: {metrics['chat_template_sha256']}.",
        "",
        "Версии библиотек закреплены в uv.lock; исходные JSONL и подготовленные файлы "
        "хранятся через DVC вне Git. Веса модели не загружаются. "
        "Этот этап не обучает модель и не измеряет качество классификации.",
        "",
    ]
    return "\n".join(lines)


def main():
    params = load_params()
    tokenizer = load_tokenizer(params)
    input_hashes = {}
    for name in ("train", "val"):
        path = Path(params["data"][f"{name}_jsonl"])
        if not path.is_file():
            raise ValueError(f"Нет {path}; выполните make prepare или make pull-data")
        input_hashes[name] = sha256(path)
        if input_hashes[name] != params["provenance"]["sha256"][name]:
            raise ValueError(f"{path}: SHA-256 отличается от закреплённой версии ДЗ 3")
    splits, outputs = {}, {}
    for name in ("train", "val"):
        examples, stats, _metas = process_split(
            tokenizer, name, Path(params["data"][f"{name}_jsonl"]), params
        )
        splits[name], outputs[name] = stats, examples
    overlap = {e["id"] for e in outputs["train"]} & {e["id"] for e in outputs["val"]}
    if overlap:
        raise ValueError(f"Обнаружены общие id train и val: {len(overlap)}")
    passed = all(s["truncation_passed"] and s["examples_kept"] > 0 for s in splits.values())
    template_hash = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    metrics = {
        "model": params["model"]["name"],
        "model_revision": params["model"]["revision"],
        "enable_thinking": params["model"].get("enable_thinking"),
        "chat_template_sha256": template_hash,
        "tokenizer_class": type(tokenizer).__name__,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_seq_len": params["tokenize"]["max_seq_len"],
        "padding_side": tokenizer.padding_side,
        "truncated_warn_ratio": params["tokenize"]["truncated_warn_ratio"],
        "passed": passed,
        "input_sha256": input_hashes,
        "source": {k: params["provenance"][k] for k in ("git_tag", "git_commit")},
        "software": {
            k: importlib.metadata.version(k)
            for k in ("torch", "transformers", "tokenizers", "numpy")
        },
        "batch": params["batch"],
        "splits": splits,
        "train_time_estimate": estimate_train_time(splits["train"]["total_tokens"], params),
    }
    for path in (METRICS_PATH, REPORT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    REPORT_PATH.write_text(render_report(metrics))
    if not passed:
        raise SystemExit("Проверка длины не пройдена; новые файлы токенов не записаны. См. отчёт.")
    out_dir = Path(params["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in outputs.items():
        blob = {
            "format_version": 1,
            "examples": examples,
            "model": params["model"]["name"],
            "model_revision": params["model"]["revision"],
            "max_seq_len": params["tokenize"]["max_seq_len"],
            "padding_side": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "enable_thinking": params["model"].get("enable_thinking"),
            "chat_template_sha256": template_hash,
            "input_sha256": input_hashes[name],
        }
        buffer = io.BytesIO()
        torch.save(blob, buffer)
        (out_dir / f"{name}.pt").write_bytes(buffer.getvalue())
    print(f"Готово: {out_dir}, {METRICS_PATH}, {REPORT_PATH}")


if __name__ == "__main__":
    main()
