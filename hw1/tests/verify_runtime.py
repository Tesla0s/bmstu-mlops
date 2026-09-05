"""Проверка реальных запусков и сохранённых результатов make bench."""

import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

from src.config import load_params


def main() -> None:
    params = load_params()
    assert params["generate"]["temperature"] == 0, "Для проверки нужна temperature=0."
    runs = [
        subprocess.run(
            [sys.executable, "-m", "src.generate"],
            capture_output=True,
            check=True,
        )
        for _ in range(2)
    ]
    assert runs[0].stdout == runs[1].stdout, "Два успешных запуска дали разный вывод."
    output = runs[0].stdout.decode("utf-8")
    assert output.startswith(f"Модель: {params['model']['name']}\n")
    match = re.search(r"\[(\d+) токенов\]\s*$", output)
    assert match, "В выводе нет числа новых токенов."
    n_tokens = int(match.group(1))
    assert 0 < n_tokens <= params["generate"]["max_new_tokens"]
    answer = output[output.index("\n") + 1:match.start()].strip()
    assert answer, "Ответ пустой."
    assert not answer.startswith(params["bench"]["prompt"]), "В вывод попал входной вопрос."
    if not params["generate"]["enable_thinking"]:
        assert "<think>" not in answer and "</think>" not in answer

    report = json.loads(Path("docs/bench.json").read_text())
    assert report["config"] == params, "Замер сделан с другой конфигурацией."
    assert report["answer"].strip() == answer, "Генерация и замер дали разные ответы."
    assert len(report["runs"]) == params["bench"]["measure_runs"]
    for run in report["runs"]:
        assert math.isfinite(run["elapsed_sec"]) and run["elapsed_sec"] > 0
        assert 0 < run["new_tokens"] <= params["generate"]["max_new_tokens"]
        assert math.isclose(
            run["tokens_per_sec"], run["new_tokens"] / run["elapsed_sec"]
        )
    speeds = [run["tokens_per_sec"] for run in report["runs"]]
    assert speeds == report["tokens_per_sec_all"]
    assert math.isclose(report["tokens_per_sec"], statistics.median(speeds))
    for key in ("load_time_sec", "peak_rss_mb"):
        assert math.isfinite(report[key]) and report[key] > 0

    for i, run in enumerate(runs, 1):
        Path(f"docs/generate_run{i}.txt").write_bytes(run.stdout)
    result = {
        "generation_exit_codes": [run.returncode for run in runs],
        "byte_identical": True,
        "stdout_sha256": hashlib.sha256(runs[0].stdout).hexdigest(),
        "new_tokens": n_tokens,
        "answer_nonempty": True,
        "prompt_excluded": True,
        "thinking_disabled": not params["generate"]["enable_thinking"],
        "benchmark_matches_config": True,
        "benchmark_answer_matches_generation": True,
        "benchmark_arithmetic_valid": True,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    Path("docs/verification.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
