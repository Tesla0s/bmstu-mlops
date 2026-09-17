"""Fail-closed diversity checks, including classification-specific safeguards."""

import json
import re
from collections import Counter

from src.config import load_params
from src.io_utils import write_json
from src.schema import iter_examples
from src.stats import spread
from src.textnorm import normalize_group, normalize_text


class DiversityError(ValueError):
    pass


def measure(path, group_key):
    examples = list(iter_examples(path))
    if not examples:
        raise DiversityError(f"{path}: empty dataset")
    systems = Counter(normalize_text(ex.messages[0].content) for ex in examples)
    groups = Counter(normalize_group(getattr(ex, group_key)) for ex in examples)
    answers = Counter(normalize_text(ex.assistant) for ex in examples)
    lengths = Counter(len(ex.assistant) for ex in examples)
    categories, invalid_answers = Counter(), 0
    vocabulary = set()
    prompts_by_category = {}
    for ex in examples:
        vocabulary.update(word.lower() for word in re.findall(r"\b[A-Za-z]{3,}\b", ex.user))
        try:
            answer = json.loads(ex.assistant)
            if (
                not isinstance(answer, dict)
                or set(answer) != {"category"}
                or not isinstance(answer["category"], str)
            ):
                raise ValueError("not a category object")
            category = answer["category"]
            categories[category] += 1
            prompts_by_category.setdefault(category, set()).add(
                normalize_text(ex.messages[0].content)
            )
        except (ValueError, TypeError):
            invalid_answers += 1
    top, count = groups.most_common(1)[0]
    return {
        "examples": len(examples),
        "system_prompts": len(systems),
        "groups": len(groups),
        "largest_group": top,
        "largest_group_share": count / len(examples),
        "largest_system_prompt_share": max(systems.values()) / len(examples),
        "answer_len": spread([len(ex.assistant) for ex in examples]),
        "same_length_share": max(lengths.values()) / len(examples),
        "duplicate_answer_share": sum(n - 1 for n in answers.values()) / len(examples),
        "user_len": spread([len(ex.user) for ex in examples]),
        "unique_user_share": len({normalize_text(ex.user) for ex in examples}) / len(examples),
        "vocabulary": len(vocabulary),
        "class_counts": dict(categories),
        "class_shares": {key: value / len(examples) for key, value in sorted(categories.items())},
        "invalid_answers": invalid_answers,
        "system_prompts_per_class": {
            key: len(values) for key, values in prompts_by_category.items()
        },
    }


def violations(stats, cfg):
    failed = []
    if stats["examples"] < cfg["min_examples"]:
        failed.append(f"мало примеров: {stats['examples']}, нужно ≥ {cfg['min_examples']}")
    if stats["system_prompts"] < cfg["min_system_prompts"]:
        failed.append(
            f"системных промптов {stats['system_prompts']}, нужно ≥ {cfg['min_system_prompts']}"
        )
    if stats["groups"] < cfg["min_groups"]:
        failed.append(f"групп {stats['groups']}, нужно ≥ {cfg['min_groups']}")
    if stats["largest_group_share"] > cfg["max_group_share"]:
        failed.append(
            f"крупнейшая группа {stats['largest_group_share']:.3f} > {cfg['max_group_share']}"
        )
    if stats["answer_len"]["ratio_p90_p10"] < cfg["min_answer_len_ratio"]:
        failed.append(
            f"разброс длин ответа {stats['answer_len']['ratio_p90_p10']} < {cfg['min_answer_len_ratio']}"
        )
    if stats["same_length_share"] > cfg["max_same_length_share"]:
        failed.append(
            f"одну и ту же длину имеют {stats['same_length_share']:.3f} ответов, порог {cfg['max_same_length_share']}"
        )
    if stats["duplicate_answer_share"] > cfg["max_duplicate_answer_share"]:
        failed.append(
            f"повторяющихся ответов {stats['duplicate_answer_share']:.3f} > {cfg['max_duplicate_answer_share']}"
        )
    if "required_categories" in cfg:
        if stats["invalid_answers"]:
            failed.append(f"ответов вне JSON-схемы category: {stats['invalid_answers']}")
        if set(stats["class_counts"]) != set(cfg["required_categories"]):
            failed.append(
                f"категории {sorted(stats['class_counts'])}, нужны {cfg['required_categories']}"
            )
        for category in cfg["required_categories"]:
            share = stats["class_shares"].get(category, 0)
            if not cfg["min_class_share"] <= share <= cfg["max_class_share"]:
                failed.append(
                    f"доля {category}: {share:.3f}, допустимо {cfg['min_class_share']}..{cfg['max_class_share']}"
                )
            if stats["system_prompts_per_class"].get(category, 0) < cfg["min_system_prompts"]:
                failed.append(f"системных промптов внутри {category} недостаточно")
        if stats["user_len"]["ratio_p90_p10"] < cfg["min_user_len_ratio"]:
            failed.append(
                f"разброс длин обращений {stats['user_len']['ratio_p90_p10']} < {cfg['min_user_len_ratio']}"
            )
        if stats["unique_user_share"] < cfg["min_unique_user_share"]:
            failed.append(
                f"доля уникальных обращений {stats['unique_user_share']} < {cfg['min_unique_user_share']}"
            )
        if stats["vocabulary"] < cfg["min_vocabulary"]:
            failed.append(f"разных английских слов {stats['vocabulary']} < {cfg['min_vocabulary']}")
        if stats["largest_system_prompt_share"] > cfg["max_system_prompt_share"]:
            failed.append(
                f"доля крупнейшего системного промпта {stats['largest_system_prompt_share']:.3f} > {cfg['max_system_prompt_share']}"
            )
    return failed


def main():
    params = load_params()
    stats = measure(params["paths"]["clean"], params["split"]["group_key"])
    failed = violations(stats, params["diversity"])
    write_json(
        params["paths"]["metrics_diversity"],
        {
            "version": params["collect"]["version"],
            **stats,
            "thresholds": params["diversity"],
            "violations": failed,
            "passed": not failed,
        },
    )
    if failed:
        raise SystemExit("diversity: набор отклонён\n  " + "\n  ".join(failed))
    print(
        f"diversity: {stats['examples']} примеров, {stats['system_prompts']} инструкций, {stats['groups']} групп; все пороги выполнены"
    )


if __name__ == "__main__":
    main()
