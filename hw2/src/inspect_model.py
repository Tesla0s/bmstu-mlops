"""Анатомия модели: параметры по слоям, нормы активаций, профиль памяти.

    python -m src.inspect_model            полный разбор, отчёт в docs/
    python -m src.inspect_model --probe M  один режим замера памяти (служебный
                                           вызов из отдельного процесса)

Файл называется inspect_model.py, а не inspect.py: имя inspect занято
модулем стандартной библиотеки, и его перекрытие ломает импорты в чужом коде.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import subprocess
import threading
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model

from src.config import load_params
from src.model import build_prompt, load_model, set_seed

# Пик RSS снимается разными механизмами на разных ОС, поэтому оба импорта
# необязательные: resource есть на macOS и Linux, но его нет на Windows;
# psutil нужен на Windows, где peak_wset — единственный high-water mark,
# который отдаёт система. Код, написанный под одну ОС, у соседа не запустится.
try:
    import resource
except ImportError:
    resource = None

try:
    import psutil
except ImportError:
    psutil = None

# transformers читает safetensors в несколько потоков, и на связке
# pyo3 OnceLock + GIL загрузка иногда встаёт намертво: на этой машине
# примерно один процесс из шести не доживал до конца from_pretrained.
# Замер обязан быть воспроизводимым, поэтому читаем последовательно —
# на модели 0.6B это не стоит ничего (4.4 с против 4.5 с).
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

MODES = ("inference", "full_ft", "lora")

# Порядок задаёт порядок строк в таблице. Проверка идёт сверху вниз,
# поэтому «norm» стоит после проекций: в их именах слова norm нет.
GROUPS = (
    ("embed", ("embed_tokens",)),
    ("q_proj", ("q_proj",)),
    ("k_proj", ("k_proj",)),
    ("v_proj", ("v_proj",)),
    ("o_proj", ("o_proj",)),
    ("gate_proj", ("gate_proj",)),
    ("up_proj", ("up_proj",)),
    ("down_proj", ("down_proj",)),
    ("norm", ("norm",)),
    ("lm_head", ("lm_head",)),
)


def resolve_device(params: dict) -> torch.device:
    """Развернуть device: auto в конкретное устройство — ровно один раз.

    Строка «auto» уходит в device_map и включает диспетчер accelerate,
    который для шага обучения только мешает. Решаем здесь и передаём дальше
    уже конкретное имя.
    """
    name = params["model"]["device"]
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# 1. Параметры по типам модулей
# --------------------------------------------------------------------------

def group_of(name: str) -> str:
    """Тип модуля по имени параметра."""
    for group, marks in GROUPS:
        if any(mark in name for mark in marks):
            return group
    return "прочее"


def parameter_rows(model) -> list[dict]:
    """Все тензоры параметров модели.

    remove_duplicate=False — иначе в таблицу не попадёт lm_head.
    """
    rows = []
    seen = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        rows.append({
            "name": name,
            "shape": tuple(param.shape),
            "numel": param.numel(),
            "tied": id(param) in seen,
            "shared_with": seen.get(id(param)),
        })
        seen.setdefault(id(param), name)
    return rows


def group_table(rows: list[dict]) -> list[dict]:
    """Свод «тип модуля → shape → параметров → доля от всей модели».

    В params попадают только уникальные тензоры, в tied_params — то,
    что модуль переиспользует у соседа.
    """
    total = sum(r["numel"] for r in rows if not r["tied"])
    agg: dict[str, dict] = {}
    for row in rows:
        group = group_of(row["name"])
        item = agg.setdefault(group, {
            "group": group, "modules": 0, "shapes": [], "params": 0, "tied_params": 0,
        })
        item["modules"] += 1
        shape = "×".join(map(str, row["shape"]))
        if shape not in item["shapes"]:
            item["shapes"].append(shape)
        if row["tied"]:
            item["tied_params"] += row["numel"]
        else:
            item["params"] += row["numel"]

    order = [g for g, _ in GROUPS] + ["прочее"]
    table = [agg[g] for g in order if g in agg]
    for item in table:
        # В группе norm форм две (по голове и по hidden), показываем обе.
        item["shape"] = ", ".join(item.pop("shapes"))
        item["share"] = item["params"] / total
    return table


# --------------------------------------------------------------------------
# 2. Forward-hooks и нормы активаций
# --------------------------------------------------------------------------

def decoder_layers(model):
    """Список декодер-блоков. У Qwen3 это model.model.layers."""
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    return decoder.layers


def hook_targets(model) -> dict[str, int]:
    """Первый, средний и последний блок — по номерам, а не по именам."""
    n_layers = len(decoder_layers(model))
    return {"первый": 0, "средний": n_layers // 2, "последний": n_layers - 1}


@contextmanager
def forward_hooks(modules: dict):
    """Временно собирать нормы. Снять каждый обработчик даже при исключении."""
    store = {}
    handles = []

    def make_hook(label):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store[label] = hidden[0].detach().float().norm(dim=-1).cpu().tolist()
        return hook

    try:
        for label, module in modules.items():
            handles.append(module.register_forward_hook(make_hook(label)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


def activation_norms(tokenizer, model, params: dict) -> dict:
    """Нормы по позициям токенов без изменения режима модели и чужих хуков."""
    layers = decoder_layers(model)
    targets = hook_targets(model)
    prompt = build_prompt(tokenizer, params, params["hooks"]["prompt"])
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    states = {module: module.training for module in model.modules()}
    hooks_before = sum(len(m._forward_hooks) for m in model.modules())
    try:
        model.eval()
        with forward_hooks({label: layers[i] for label, i in targets.items()}) as store:
            with torch.inference_mode():
                model(**inputs, use_cache=False)
    finally:
        for module, state in states.items():
            module.training = state
    return {
        "layers": targets,
        "norms": {label: store[label] for label in targets},
        "n_tokens": inputs["input_ids"].shape[1],
        "hooks_before": hooks_before,
        "hooks_after": sum(len(m._forward_hooks) for m in model.modules()),
    }


# --------------------------------------------------------------------------
# 3. Сколько параметров добавляет LoRA
# --------------------------------------------------------------------------

def lora_config(params: dict, cfg: dict) -> LoraConfig:
    """LoraConfig из params.yaml — ни r, ни target_modules в коде не зашиты."""
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=params["lora"]["alpha_ratio"] * cfg["r"],
        lora_dropout=params["lora"]["dropout"],
        target_modules=list(cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def lora_params_formula(model, r: int, target_modules) -> int:
    """Своя формула: на каждый целевой Linear ровно r * (in_features + out_features).

    A имеет форму (r, in), B — (out, r), смещений у них нет. Вся арифметика
    LoRA умещается в эту строчку, и она обязана сойтись с peft до штуки.
    """
    targets = set(target_modules)
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            total += r * (module.in_features + module.out_features)
    return total


def lora_report(model, params: dict) -> list[dict]:
    """Посчитать адаптеры, затем вернуть исходную модель без их следов."""
    base_params = sum(p.numel() for p in model.parameters())
    original_grad = {id(p): p.requires_grad for p in model.parameters()}
    training_states = {m: m.training for m in model.modules()}
    result = []
    for cfg in params["lora"]["configs"]:
        expected = lora_params_formula(model, cfg["r"], cfg["target_modules"])
        targets = [{"name": name, "input": m.in_features, "output": m.out_features,
                    "params": cfg["r"] * (m.in_features + m.out_features)}
                   for name, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear)
                   and name.rsplit(".", 1)[-1] in cfg["target_modules"]]
        wrapped = get_peft_model(model, lora_config(params, cfg))
        try:
            wrapped.print_trainable_parameters()
            trainable, total = wrapped.get_nb_trainable_parameters()
        finally:
            model = wrapped.unload()
            # PEFT 0.17 оставляет служебную конфигурацию после unload().
            if hasattr(model, "peft_config"):
                delattr(model, "peft_config")
            for parameter in model.parameters():
                parameter.requires_grad_(original_grad[id(parameter)])
            for module, state in training_states.items():
                module.training = state
        result.append({
            "name": cfg["name"], "r": cfg["r"],
            "target_modules": list(cfg["target_modules"]),
            "target_count": len(targets), "targets": targets,
            "formula": expected, "peft": trainable, "match": expected == trainable,
            "total_with_adapter": total, "share_of_base": trainable / base_params,
        })
    return result


# --------------------------------------------------------------------------
# 4. Память в трёх режимах
# --------------------------------------------------------------------------

def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def device_allocated_bytes(device: torch.device) -> int:
    """Текущая память устройства; на CPU — максимум RSS процесса."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    return peak_rss()[0]


def device_metric_source(device: torch.device) -> str:
    if device.type == "mps":
        return "torch.mps.driver_allocated_memory"
    if device.type == "cuda":
        return "torch.cuda.max_memory_allocated"
    return peak_rss()[1]


def peak_rss() -> tuple[int, str]:
    """Пик RSS процесса в байтах И метка источника метрики.

    Метка возвращается не для красоты: «пик 1001 МБ» без указания, чем это
    снято, — не результат, а повод для спора. Тем более что RSS и память
    ускорителя — разные величины (см. PeakMemory ниже).

    Три ОС меряют по-разному:

    * macOS и Linux — `resource.getrusage(RUSAGE_SELF).ru_maxrss`, high-water
      mark процесса; на macOS он в байтах, на Linux в килобайтах;
    * Windows — `psutil.Process().memory_info().peak_wset`: модуля `resource`
      там нет вовсе. Обратное тоже верно — поля `peak_wset` нет на macOS и
      Linux, и код, написанный только под него, у соседа падает.

    Если недоступно ничего — исключение. Тихий ноль хуже отсутствия числа:
    ноль попадает в отчёт и его выдают за результат.
    """
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (peak if sys.platform == "darwin" else peak * 1024), "ru_maxrss"
    if psutil is not None and hasattr(psutil.Process().memory_info(), "peak_wset"):
        return int(psutil.Process().memory_info().peak_wset), "peak_wset"
    raise RuntimeError(
        f"нечем снять пик RSS на платформе {sys.platform}: модуля resource нет, "
        "а psutil не установлен либо не отдаёт peak_wset. Выполните uv sync."
    )


class PeakMemory:
    """Максимум за весь замер; MPS опрашивается потоком и на границах этапов."""

    def __init__(self, device, interval=0.002):
        if interval <= 0:
            raise ValueError("Интервал опроса должен быть положительным")
        self.device, self.interval = device, interval
        self.used, self.samples = 0, 0
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.thread = None
        self.stages = []

    def sample(self):
        value = device_allocated_bytes(self.device)
        with self.lock:
            self.used = max(self.used, value)
            self.samples += 1
        return value

    def checkpoint(self, stage):
        synchronize(self.device)
        value = self.sample()
        self.stages.append({"stage": stage, "allocated_bytes": value})

    def _poll(self):
        try:
            while not self.stop.wait(self.interval):
                self.sample()
        except Exception as error:
            self.error = error

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()
        if self.device.type == "mps":
            self.thread = threading.Thread(target=self._poll, name="memory-sampler", daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        try:
            self.checkpoint("end")
        finally:
            self.stop.set()
            if self.thread:
                self.thread.join()
        if self.device.type == "cuda":
            self.used = max(self.used, torch.cuda.max_memory_allocated(self.device))
        if self.error is not None and exc[0] is None:
            raise RuntimeError("Опрос памяти завершился ошибкой") from self.error
        return False

    def result(self):
        rss, rss_source = peak_rss()
        accelerator = self.device.type in ("mps", "cuda")
        peak = self.used if accelerator else rss
        return {
            "peak_mb": round(peak / 1024**2, 1),
            "peak_bytes": peak,
            "peak_device_mb": round(self.used / 1024**2, 1),
            "peak_rss_mb": round(rss / 1024**2, 1),
            "metric": f"память устройства {self.device.type}" if accelerator else "RSS процесса",
            "metric_source": device_metric_source(self.device),
            "rss_source": rss_source,
            "sample_interval_sec": self.interval if self.device.type == "mps" else None,
            "samples": self.samples,
            "stages": self.stages,
        }


def tensor_bytes(tensors):
    seen = set()
    total = 0
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and id(tensor) not in seen:
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return total


def measure_mode(mode: str, params: dict) -> dict:
    """Один forward или один настоящий шаг AdamW в отдельном процессе."""
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим: {mode}")
    device = resolve_device(params)
    params["model"]["device"] = str(device)
    set_seed(params["generate"]["seed"])
    started = time.perf_counter()
    loss = None
    gradient_bytes = optimizer_bytes = 0
    memory = params["memory"]
    with PeakMemory(device, memory.get("sample_interval_sec", 0.002)) as peak:
        _, model = load_model(params)
        base_weights_bytes = tensor_bytes(model.parameters())
        peak.checkpoint("loaded")
        ids = torch.randint(0, model.config.vocab_size,
                            (memory["batch_size"], memory["seq_len"]), device=model.device)
        if mode == "lora":
            model = get_peft_model(model, lora_config(params, params["lora"]["configs"][0]))
        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable_params = sum(p.numel() for p in trainable) if mode != "inference" else 0
        parameter_bytes = tensor_bytes(model.parameters())
        if mode == "inference":
            model.eval()
            with torch.inference_mode():
                output = model(input_ids=ids, use_cache=memory.get("use_cache", False))
            peak.checkpoint("forward")
        else:
            model.train()
            optimizer = torch.optim.AdamW(trainable, lr=float(memory["lr"]), foreach=False)
            output = model(input_ids=ids, labels=ids, use_cache=memory.get("use_cache", False))
            peak.checkpoint("forward")
            output.loss.backward()
            peak.checkpoint("backward")
            gradient_bytes = tensor_bytes(p.grad for p in trainable)
            optimizer.step()
            peak.checkpoint("optimizer_step")
            optimizer_bytes = tensor_bytes(v for state in optimizer.state.values() for v in state.values())
            loss = float(output.loss.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            peak.checkpoint("zero_grad")
    result = peak.result()
    result.update(
        mode=mode, device=str(device), dtype=params["model"]["dtype"],
        seq_len=memory["seq_len"], batch_size=memory["batch_size"],
        seconds=round(time.perf_counter() - started, 3), loss=loss,
        pid=os.getpid(), measured_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        seed=params["generate"]["seed"], use_cache=memory.get("use_cache", False),
        optimizer="AdamW(foreach=False)" if mode != "inference" else None,
        trainable_params=trainable_params, base_weights_bytes=base_weights_bytes,
        parameter_bytes=parameter_bytes, gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_bytes,
    )
    if result["peak_bytes"] < base_weights_bytes:
        raise RuntimeError("Пик памяти меньше фактического размера весов: проверьте метрику")
    return result


def memory_profile(params: dict) -> list[dict]:
    """Новый процесс для каждого режима и повтора; сохраняем все исходные прогоны."""
    repeats = int(params["memory"].get("repeats", 1))
    if repeats < 1:
        raise ValueError("Нужен хотя бы один прогон")
    results = []
    for mode in MODES:
        runs = []
        for _ in range(repeats):
            completed = subprocess.run(
                [sys.executable, "-m", "src.inspect_model", "--probe", mode, "--params-stdin"],
                input=json.dumps(params), capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[1], timeout=300,
            )
            if completed.returncode:
                raise RuntimeError(f"Замер {mode} завершился с кодом {completed.returncode}: {completed.stderr}")
            runs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        worst = dict(max(runs, key=lambda item: item["peak_bytes"]))
        worst.update(repeats=repeats, peak_mb_runs=[r["peak_mb"] for r in runs], runs=runs)
        results.append(worst)
    return results


# --------------------------------------------------------------------------
# 5. Условия, без которых цифры замера ничего не значат
# --------------------------------------------------------------------------

def environment(params: dict, memory: list[dict]) -> dict:
    """Всё, что нужно, чтобы чужой замер можно было сравнить со своим.

    Расхождение в полтора раза между двумя машинами — норма, а не ошибка,
    но только если написано, чем эти машины отличались. Метрики памяти берутся
    из самих замеров, а не из предположений: что реально сработало в дочернем
    процессе, то и уходит в отчёт.
    """
    def unique(field: str) -> str:
        values = dict.fromkeys(str(item.get(field) or "") for item in memory)
        return ", ".join(value for value in values if value)

    return {
        "platform": platform.platform(),
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "processor": platform.processor(),
        "seed": params["generate"]["seed"],
        "use_cache": params["memory"].get("use_cache", False),
        "sample_interval_sec": params["memory"].get("sample_interval_sec", 0.002),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "0"),
        "system": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": params["model"]["device"],
        "dtype": params["model"]["dtype"],
        "seq_len": params["memory"]["seq_len"],
        "batch_size": params["memory"]["batch_size"],
        "repeats": max(1, int(params["memory"].get("repeats", 1))),
        "memory_metric": unique("metric_source"),
        "rss_metric": unique("rss_source"),
    }


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Разбор модели: параметры, активации, память")
    parser.add_argument("--probe", choices=MODES, help="служебный режим: замерить память и выйти")
    parser.add_argument("--params-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    params = json.load(sys.stdin) if args.params_stdin else load_params()
    set_seed(params["generate"]["seed"])
    params["model"]["device"] = str(resolve_device(params))

    if args.probe:
        print(json.dumps(measure_mode(args.probe, params), ensure_ascii=False))
        return

    # Импорт здесь, а не наверху: matplotlib не нужен в служебных --probe
    # процессах, а тянется он заметно дольше остального.
    from src.report import write_report

    # Замеры идут до загрузки родительской модели, чтобы не держать лишние веса.
    memory = memory_profile(params)
    tokenizer, model = load_model(params)
    rows = parameter_rows(model)
    table = group_table(rows)
    total = sum(item["params"] for item in table)
    activations = activation_norms(tokenizer, model, params)
    again = activation_norms(tokenizer, model, params)
    activations["repeat_identical"] = activations["norms"] == again["norms"]

    report = {
        "model": params["model"]["name"],
        "dtype": params["model"]["dtype"],
        "device": params["model"]["device"],
        "environment": environment(params, memory),
        "config": {
            key: getattr(model.config, key)
            for key in ("num_hidden_layers", "hidden_size", "intermediate_size",
                        "num_attention_heads", "num_key_value_heads", "head_dim",
                        "vocab_size", "tie_word_embeddings")
        },
        "model_revision": getattr(model.config, "_commit_hash", None),
        "params_rows": rows,
        "params_total": total,
        "params_direct": sum(p.numel() for p in model.parameters()),
        "params_by_group": table,
        "activations": activations,
        "lora": lora_report(model, params),
        "memory": memory,
    }

    Path(params["report"]["json"]).parent.mkdir(exist_ok=True)
    Path(params["report"]["json"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(report, params)

    env = report["environment"]
    print(f"\nПараметров: {total:,} (по таблице) / {report['params_direct']:,} (напрямую)"
          .replace(",", " "))
    print(f"Условия: {env['platform']}, device {env['device']}, dtype {env['dtype']}, "
          f"seq_len {env['seq_len']}, прогонов на режим {env['repeats']}, "
          f"torch {env['torch']}, transformers {env['transformers']}")
    for mode in report["memory"]:
        print(f"  {mode['mode']:<10} пик {mode['peak_mb']:>8.1f} МБ  "
              f"({mode['metric']}: {mode['metric_source']})")
    print(f"\nОтчёт: {params['report']['markdown']}, график: {params['hooks']['plot']}")


if __name__ == "__main__":
    main()
