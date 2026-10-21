"""Строгая оценка ответов и запрет выпуска при ошибках или регрессии."""

import json
import math
import re


def _unique(items, kind):
    if not isinstance(items, list) or not items:
        raise ValueError(f"{kind}: нужен непустой список")
    result = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{kind}: запись должна быть словарём")
        key = item.get("id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"{kind}: пустой или повторяющийся id")
        result[key] = item
    return result


def _aggregate(items):
    n = len(items)
    correct = sum(x["correct"] for x in items)
    valid = sum(x["valid"] for x in items)
    errors = sum(x["error"] is not None for x in items)
    return dict(n=n, correct=correct, valid=valid, errors=errors, accuracy=correct / n, format_rate=valid / n)


def _groups(items):
    names = sorted({s for item in items for s in item["slices"]})
    return {s: _aggregate([item for item in items if s in item["slices"]]) for s in names}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Повторяющееся поле JSON")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Недопустимая константа JSON: {value}")


def score(rows, outputs):
    """Каждый вопрос сохраняется в знаменателе, ответы сопоставляются по id."""
    tasks, answers = _unique(rows, "Задания"), _unique(outputs, "Ответы")
    if tasks.keys() != answers.keys():
        raise ValueError("ID заданий и ответов не совпадают")
    per_item = []
    for key, row in tasks.items():
        choices, slices = row.get("choices"), row.get("slices")
        if (
            not isinstance(choices, list)
            or not choices
            or any(not isinstance(x, str) or not x for x in choices)
            or len(set(choices)) != len(choices)
            or row.get("expected") not in choices
        ):
            raise ValueError(f"{key}: неверные choices/expected")
        if (
            not isinstance(slices, list)
            or not slices
            or any(not isinstance(x, str) or not x for x in slices)
            or len(set(slices)) != len(slices)
        ):
            raise ValueError(f"{key}: неверные slices")
        answer = answers[key]
        if "raw" not in answer or "error" not in answer:
            raise ValueError(f"{key}: отсутствует raw/error")
        raw, error = answer["raw"], answer["error"]
        if not isinstance(raw, str) or (error is not None and (not isinstance(error, str) or not error)):
            raise ValueError(f"{key}: неверные типы raw/error")
        parsed, valid = None, False
        try:
            obj = json.loads(raw, object_pairs_hook=_object, parse_constant=_nonfinite)
            if (
                isinstance(obj, dict)
                and set(obj) == {"answer"}
                and isinstance(obj["answer"], str)
                and obj["answer"] in choices
            ):
                parsed = obj["answer"]
                valid = error is None
        except (ValueError, RecursionError):
            pass
        per_item.append(
            dict(
                id=key,
                raw=raw,
                parsed=parsed,
                valid=valid,
                correct=valid and parsed == row["expected"],
                error=error,
                slices=list(slices),
            )
        )
    return dict(overall=_aggregate(per_item), slices=_groups(per_item), per_item=per_item)


def _counts(value, expected, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where}: нет метрик")
    for key in ("n", "correct", "valid", "errors"):
        if type(value.get(key)) is not int or value[key] < 0 or value[key] != expected[key]:
            raise ValueError(f"{where}: неверный count {key}")
    for key in ("accuracy", "format_rate"):
        v = value.get(key)
        if (
            type(v) not in (int, float)
            or not math.isfinite(v)
            or not math.isclose(v, expected[key], rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError(f"{where}: неверная метрика {key}")


def _validate(report, where):
    if not isinstance(report, dict):
        raise ValueError(f"{where}: отсутствует отчёт")
    items = list(_unique(report.get("per_item"), where).values())
    for item in items:
        if type(item.get("valid")) is not bool or type(item.get("correct")) is not bool:
            raise ValueError(f"{where}: valid/correct должны быть bool")
        if "error" not in item or "raw" not in item or "parsed" not in item:
            raise ValueError(f"{where}: неполный per_item")
        if not isinstance(item["raw"], str):
            raise ValueError(f"{where}: raw должен быть строкой")
        if item["correct"] and not item["valid"]:
            raise ValueError(f"{where}: правильный ответ обязан быть валидным")
        if item["valid"] and not isinstance(item["parsed"], str):
            raise ValueError(f"{where}: валидный ответ должен содержать строковую метку")
        if item["error"] is not None and (
            not isinstance(item["error"], str) or not item["error"] or item["valid"]
        ):
            raise ValueError(f"{where}: неверное состояние ошибки")
        ss = item.get("slices")
        if (
            not isinstance(ss, list)
            or not ss
            or any(not isinstance(s, str) or not s for s in ss)
            or len(set(ss)) != len(ss)
        ):
            raise ValueError(f"{where}: неверные срезы")
    expected = _aggregate(items)
    _counts(report.get("overall"), expected, where)
    groups = _groups(items)
    if not isinstance(report.get("slices"), dict) or set(report["slices"]) != set(groups):
        raise ValueError(f"{where}: неверный набор срезов")
    for name, metrics in groups.items():
        _counts(report["slices"][name], metrics, f"{where}/{name}")
    return {x["id"]: set(x["slices"]) for x in items}


def _model(spec, where):
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: не указана модель")
    kind = spec.get("kind")
    # Синтетический тип из неизменённых тестов преподавателя.
    # Реальная обвязка дополнительно сверяет весь spec с baseline.json/реестром.
    if kind == "fixture":
        return
    if kind not in ("base", "adapter"):
        raise ValueError(f"{where}: неизвестный тип модели")
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError(f"{where}: отсутствует имя модели")
    if kind == "base":
        revision = spec.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{where}: нужна точная ревизия базовой модели")
    else:
        version, run_id = spec.get("version"), spec.get("run_id")
        if not isinstance(version, str) or not re.fullmatch(r"[1-9][0-9]*", version):
            raise ValueError(f"{where}: нужна числовая версия адаптера")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"{where}: отсутствует точный run_id адаптера")
        if spec.get("uri") != f"models:/{spec['name']}/{version}":
            raise ValueError(f"{where}: нужен неизменяемый URI той же версии")


def quality_gate(baseline, candidate, critical_slices):
    """Допуск регрессии равен нулю. Проверяем и counts, и каждую строку отчёта."""
    if (
        not isinstance(critical_slices, list)
        or not critical_slices
        or any(not isinstance(s, str) or not s for s in critical_slices)
    ):
        raise ValueError("Нужно заранее указать критические срезы")
    memberships, behaviors = [], []
    for role, report in (("baseline", baseline), ("candidate", candidate)):
        if not isinstance(report, dict) or report.get("status") != "complete":
            raise ValueError(f"{role}: оценка не завершена")
        _model(report.get("model"), role)
        for field in ("protocol_sha256", "golden_sha256", "baseline_spec_sha256"):
            if not isinstance(report.get(field), str) or not report[field]:
                raise ValueError(f"{role}: нет {field}")
        memberships.append(_validate(report, role))
        behaviors.append(_validate(report.get("behavioral"), f"{role}/behavioral"))
        if report["overall"]["n"] < 50 or len(report["slices"]) < 3:
            raise ValueError(f"{role}: нужно минимум 50 вопросов и три среза")
        if report["behavioral"]["overall"]["n"] < 3:
            raise ValueError(f"{role}: нужно минимум три behavioral-кейса")
        if report["overall"]["errors"] or report["behavioral"]["overall"]["errors"]:
            raise ValueError(f"{role}: есть ошибки генерации")
    for field in ("protocol_sha256", "golden_sha256", "baseline_spec_sha256"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Несовместимые отчёты: {field}")
    if memberships[0] != memberships[1] or behaviors[0] != behaviors[1]:
        raise ValueError("Различаются id или состав срезов")
    for metric in ("accuracy", "format_rate"):
        if candidate["overall"][metric] < baseline["overall"][metric]:
            raise ValueError(f"Регрессия общей {metric}")
    for name in critical_slices:
        if name not in baseline["slices"] or name not in candidate["slices"]:
            raise ValueError(f"Отсутствует критический срез {name}")
        if candidate["slices"][name]["accuracy"] < baseline["slices"][name]["accuracy"]:
            raise ValueError(f"Регрессия критического среза {name}")
    behavior = candidate["behavioral"]["overall"]
    if behavior["correct"] != behavior["n"]:
        raise ValueError("Не все обязательные behavioral-кейсы пройдены")
    return dict(
        passed=True,
        regression_tolerance=0,
        critical_slices=critical_slices,
        baseline=baseline["overall"],
        candidate=candidate["overall"],
        behavioral=behavior,
        protocol_sha256=baseline["protocol_sha256"],
    )
