"""Два воспроизводимых прогона LoRA на неизменённых тензорах ДЗ4."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.config import load_params
from src.data import LABEL_PAD_ID, accumulation_groups, batches, load_split
from src.loss import supervised_loss_sum
from src.runtime import (
    MemoryMonitor,
    memory_metric,
    resolve_device,
    resolve_dtype,
    set_seed,
    synchronize,
)

TRAIN_CODE = (
    "src/train.py",
    "src/data.py",
    "src/runtime.py",
    "src/config.py",
    "src/loss.py",
    "pyproject.toml",
    "uv.lock",
)
TRAIN_PARAMS = ("model", "data", "lora", "train", "variants", "provenance")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs_fingerprint(params: dict) -> str:
    h = hashlib.sha256()
    for name in TRAIN_CODE:
        h.update(name.encode())
        h.update(Path(name).read_bytes())
    h.update(json.dumps({k: params.get(k) for k in TRAIN_PARAMS}, sort_keys=True).encode())
    for split, name in sorted(params["data"].items()):
        h.update(split.encode())
        h.update(sha256(name).encode())
    return h.hexdigest()[:12]


def lora_config(params: dict, n_layers: int, freeze_first: int) -> LoraConfig:
    if not 0 <= freeze_first < n_layers:
        raise ValueError("freeze_first вне диапазона слоёв")
    cfg = params["lora"]
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        modules_to_save=None,
        layers_to_transform=list(range(freeze_first, n_layers)),
        layers_pattern="layers",
        revision=params["model"]["revision"],
        task_type="CAUSAL_LM",
        bias="none",
    )


@torch.no_grad()
def evaluate(model, examples, pad_id, device, batch_size: int) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for batch in batches(examples, batch_size, pad_id, shuffle=False, seed=0):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, n = supervised_loss_sum(model, batch)
            total += loss.item()
            count += n
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("Пустая проверочная выборка")
    value = total / count
    if not math.isfinite(value):
        raise ValueError("Val loss не конечен")
    if device.type == "mps":
        torch.mps.empty_cache()
    return value


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all_layers")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--out")
    ap.add_argument("--val-limit", type=int)
    args = ap.parse_args()
    params = load_params()
    variants = {v["name"]: v for v in params["variants"]}
    if args.variant not in variants:
        raise ValueError(f"Неизвестный вариант: {args.variant}")
    tcfg = params["train"]
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps должен быть положительным")
    if args.val_limit is not None and args.val_limit < 1:
        raise ValueError("val_limit должен быть положительным")
    torch.set_num_threads(tcfg["cpu_threads"])
    set_seed(tcfg["seed"])  # До загрузки модели, инициализации LoRA и любого dropout.
    device, dtype = (
        resolve_device(params["model"]["device"]),
        resolve_dtype(params["model"]["dtype"]),
    )
    blobs = {s: load_split(path) for s, path in params["data"].items()}
    data_hashes = {s: sha256(path) for s, path in params["data"].items()}
    if data_hashes != params["provenance"]["tokenized_sha256"]:
        raise ValueError("Вход отличается от сохранённого результата ДЗ4")
    for blob in blobs.values():
        if blob["model_revision"] != params["model"]["revision"] or blob["padding_side"] != "left":
            raise ValueError("Версия модели или дополнение не совпадают с ДЗ4")
    if {e["id"] for e in blobs["train"]["examples"]} & {e["id"] for e in blobs["val"]["examples"]}:
        raise ValueError("Train и val пересекаются")
    examples = blobs["train"]["examples"]
    validation = blobs["val"]["examples"]
    if args.val_limit is not None:
        validation = validation[: args.val_limit]
    pad_id = blobs["train"]["pad_token_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        params["model"]["name"], revision=params["model"]["revision"]
    )
    tokenizer.padding_side = "left"
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    if template_hash != params["provenance"]["chat_template_sha256"]:
        raise ValueError("Шаблон отличается от ДЗ4")
    if tokenizer.pad_token_id != pad_id or blobs["val"]["pad_token_id"] != pad_id:
        raise ValueError("Разные токены дополнения")
    model = AutoModelForCausalLM.from_pretrained(
        params["model"]["name"],
        revision=params["model"]["revision"],
        dtype=dtype,
        attn_implementation=params["model"]["attn_implementation"],
    ).to(device)
    n_layers = model.config.num_hidden_layers
    freeze = variants[args.variant]["freeze_first"]
    if tcfg["gradient_checkpointing"]:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(params, n_layers, freeze))
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_names or any("lora_" not in n for n in trainable_names):
        raise ValueError("Обучаться должны только матрицы LoRA")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if "lora_B" in n):
        raise ValueError("Начальная поправка должна быть нулевой")
    steps_per_epoch = math.ceil(math.ceil(len(examples) / tcfg["batch_size"]) / tcfg["grad_accum"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        foreach=False,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(total_steps * tcfg["warmup_ratio"])),
        total_steps,
    )
    print(
        f"[{args.variant}] {device}, {trainable:,}/{total:,} обучаемых; "
        f"{len(examples)} train, {len(validation)} val; {total_steps} шагов",
        flush=True,
    )
    monitor = MemoryMonitor(device)
    monitor.start()
    synchronize(device)
    started = time.perf_counter()
    base_val = evaluate(model, validation, pad_id, device, tcfg["eval_batch_size"])
    synchronize(device)
    eval_seconds = time.perf_counter() - started
    curve_train, curve_val = [], [[0, round(base_val, 6)]]
    print(f"  шаг 0: val {base_val:.6f}", flush=True)
    step = seen_examples = seen_tokens = seen_supervised = 0
    trace = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(tcfg["epochs"]):
        iterator = batches(examples, tcfg["batch_size"], pad_id, True, tcfg["seed"] + epoch)
        for group in accumulation_groups(iterator, tcfg["grad_accum"]):
            denominator = sum(int((b["labels"][:, 1:] != LABEL_PAD_ID).sum()) for b in group)
            group_loss, group_examples, group_tokens = 0.0, 0, 0
            step_started = time.perf_counter()
            lr_used = optimizer.param_groups[0]["lr"]
            for cpu_batch in group:
                group_examples += cpu_batch["input_ids"].shape[0]
                group_tokens += int(cpu_batch["attention_mask"].sum())
                batch = {k: v.to(device) for k, v in cpu_batch.items()}
                loss_sum, _ = supervised_loss_sum(model, batch)
                value = loss_sum.item()
                if not math.isfinite(value):
                    raise ValueError(f"Неконечная ошибка до шага {step + 1}")
                (loss_sum / denominator).backward()
                group_loss += value
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                tcfg["max_grad_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            step += 1
            seen_examples += group_examples
            seen_tokens += group_tokens
            seen_supervised += denominator
            curve_train.append([step, round(group_loss / denominator, 6)])
            trace.append(
                {
                    "step": step,
                    "examples": group_examples,
                    "input_tokens": group_tokens,
                    "supervised_tokens": denominator,
                    "lr": lr_used,
                    "gradient_norm": float(norm),
                    "seconds": round(time.perf_counter() - step_started, 4),
                }
            )
            monitor.sample()
            print(
                f"  шаг {step}/{total_steps}: train {curve_train[-1][1]:.6f}, "
                f"{trace[-1]['seconds']:.2f} с, {group_examples} примеров",
                flush=True,
            )
            if step % tcfg["eval_every"] == 0 or step == total_steps:
                evaluation_started = time.perf_counter()
                val = evaluate(model, validation, pad_id, device, tcfg["eval_batch_size"])
                synchronize(device)
                eval_seconds += time.perf_counter() - evaluation_started
                curve_val.append([step, round(val, 6)])
                print(f"  оценка шага {step}: val {val:.6f}", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    synchronize(device)
    wall_seconds = time.perf_counter() - started
    seconds = wall_seconds - eval_seconds
    monitor.stop()
    if max_steps is None and seen_examples != len(examples) * tcfg["epochs"]:
        raise ValueError("Не все примеры использованы за заданные эпохи")
    out_root = Path(args.out) if args.out else Path(params["paths"]["models"])
    adapter_dir = out_root / f"adapter_{args.variant}"
    if adapter_dir.exists() and any(adapter_dir.iterdir()):
        raise ValueError(f"{adapter_dir} уже существует: сохраните предыдущий опыт отдельно")
    model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "run_config.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2) + "\n"
    )
    metrics = {
        "variant": args.variant,
        "freeze_first": freeze,
        "model": params["model"]["name"],
        "model_revision": params["model"]["revision"],
        "device": device.type,
        "dtype": params["model"]["dtype"],
        "seed": tcfg["seed"],
        "lr": tcfg["lr"],
        "effective_batch": tcfg["batch_size"] * tcfg["grad_accum"],
        "steps": step,
        "expected_steps": total_steps,
        "epochs": tcfg["epochs"],
        "max_steps": max_steps,
        "val_limit": args.val_limit,
        "train_examples": len(examples),
        "val_examples": len(validation),
        "examples_seen": seen_examples,
        "input_tokens_seen": seen_tokens,
        "supervised_tokens_seen": seen_supervised,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_share": round(trainable / total, 6),
        "trainable_names": trainable_names,
        "base_val_loss": curve_val[0][1],
        "final_val_loss": curve_val[-1][1],
        "diverged": False,
        "curve_train": curve_train,
        "curve_val": curve_val,
        "step_trace": trace,
        "seconds": round(seconds, 3),
        "eval_seconds": round(eval_seconds, 3),
        "wall_seconds": round(wall_seconds, 3),
        "seconds_per_step": round(seconds / step, 3),
        "train_tokens_per_sec": round(seen_tokens / seconds, 3),
        "peak_memory_mb": round(monitor.peak / 1048576, 1),
        "peak_rss_mb": round(monitor.rss_peak / 1048576, 1),
        "memory_metric": memory_metric(device),
        "adapter_dir": str(adapter_dir),
        "adapter_weights_mb": round(
            (adapter_dir / "adapter_model.safetensors").stat().st_size / 1048576, 2
        ),
        "adapter_size_mb": dir_size_mb(adapter_dir),
        "inputs_fingerprint": inputs_fingerprint(params),
        "input_sha256": data_hashes,
        "chat_template_sha256": template_hash,
        "software": {
            n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "numpy")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    mdir = out_root / "metrics" if args.out else Path(params["paths"]["metrics"])
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"train_{args.variant}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (adapter_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "inputs_fingerprint",
                    "model_revision",
                    "input_sha256",
                    "chat_template_sha256",
                )
            },
            indent=2,
        )
        + "\n",
    )
    # Последний файл входит в измеренный размер папки.
    metrics["adapter_size_mb"] = dir_size_mb(adapter_dir)
    (mdir / f"train_{args.variant}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"[{args.variant}] {step} шагов; train {seconds:.1f} с, val {eval_seconds:.1f} с; "
        f"память {metrics['peak_memory_mb']} МиБ; папка {metrics['adapter_size_mb']} МиБ",
        flush=True,
    )


if __name__ == "__main__":
    main()
