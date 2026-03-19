from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from common import (
    ensure_dir,
    load_config,
    project_root,
    resolve_path,
    write_json,
    write_jsonl,
)


def get_meta_field(record: dict[str, Any], key: str, default: Any = None) -> Any:
    meta = record.get("meta", {})
    if key in meta:
        return meta[key]
    return record.get(key, default)


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def build_feature_content(fact: str) -> str:
    return "\n".join(
        [
            "案件类型: 刑事案件",
            f"案件事实: {fact}",
            "任务要求: 识别涉嫌罪名、相关法条，并说明判断依据",
        ]
    )


def normalize_case(record: dict[str, Any], idx: int, max_fact_chars: int) -> dict[str, Any]:
    fact = str(record.get("fact", "")).strip().replace("\n", "")
    fact = fact[:max_fact_chars]
    charges = normalize_list(get_meta_field(record, "accusation"))
    articles = [str(x) for x in get_meta_field(record, "relevant_articles", []) or []]
    imprisonment = get_meta_field(record, "term_of_imprisonment", {})

    return {
        "id": idx,
        "fact": fact,
        "feature_content": build_feature_content(fact),
        "charges": charges,
        "articles": articles,
        "imprisonment": imprisonment,
    }


def filter_case(
    item: dict[str, Any],
    allowed_charges: set[str] | None,
    min_fact_chars: int,
    single_charge_only: bool,
) -> bool:
    if len(item["fact"]) < min_fact_chars:
        return False
    if not item["charges"]:
        return False
    if single_charge_only and len(item["charges"]) != 1:
        return False
    if allowed_charges is not None and item["charges"][0] not in allowed_charges:
        return False
    return True


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_top_charges(
    train_file: Path,
    max_fact_chars: int,
    min_fact_chars: int,
    single_charge_only: bool,
    top_k: int,
) -> list[str]:
    charge_counter = Counter()
    for idx, row in enumerate(iter_jsonl(train_file)):
        item = normalize_case(row, idx, max_fact_chars)
        if filter_case(item, None, min_fact_chars, single_charge_only):
            charge_counter[item["charges"][0]] += 1
    return [name for name, _ in charge_counter.most_common(top_k)]


def write_filtered_train_candidates(
    train_file: Path,
    output_file: Path,
    max_fact_chars: int,
    min_fact_chars: int,
    single_charge_only: bool,
    top_charge_set: set[str],
) -> int:
    ensure_dir(output_file.parent)
    count = 0
    with output_file.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(iter_jsonl(train_file)):
            item = normalize_case(row, idx, max_fact_chars)
            if not filter_case(item, top_charge_set, min_fact_chars, single_charge_only):
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def select_filtered_cases(
    rows: list[dict[str, Any]],
    max_fact_chars: int,
    min_fact_chars: int,
    single_charge_only: bool,
    top_charge_set: set[str],
) -> list[dict[str, Any]]:
    selected = []
    for idx, row in enumerate(rows):
        item = normalize_case(row, idx, max_fact_chars)
        if filter_case(item, top_charge_set, min_fact_chars, single_charge_only):
            selected.append(item)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "project.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = resolve_path(config["paths"]["raw_dir"]) / "cail2018"
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    build_cfg = config["build"]

    train_file = raw_dir / "train.jsonl"
    valid_rows = list(iter_jsonl(raw_dir / "validation.jsonl"))
    test_file = raw_dir / "test.jsonl"
    test_rows = list(iter_jsonl(test_file)) if test_file.exists() else []

    top_charges = count_top_charges(
        train_file=train_file,
        max_fact_chars=build_cfg["max_fact_chars"],
        min_fact_chars=build_cfg["min_fact_chars"],
        single_charge_only=build_cfg["single_charge_only"],
        top_k=build_cfg["top_k_charges"],
    )
    top_charge_set = set(top_charges)
    train_candidates_count = write_filtered_train_candidates(
        train_file=train_file,
        output_file=processed_dir / "train_candidates.jsonl",
        max_fact_chars=build_cfg["max_fact_chars"],
        min_fact_chars=build_cfg["min_fact_chars"],
        single_charge_only=build_cfg["single_charge_only"],
        top_charge_set=top_charge_set,
    )

    target_dev = select_filtered_cases(
        rows=valid_rows[: build_cfg["max_target_samples"]],
        max_fact_chars=build_cfg["max_fact_chars"],
        min_fact_chars=build_cfg["min_fact_chars"],
        single_charge_only=build_cfg["single_charge_only"],
        top_charge_set=top_charge_set,
    )
    eval_source_rows = test_rows or valid_rows[build_cfg["max_target_samples"] :]
    eval_test = select_filtered_cases(
        rows=eval_source_rows,
        max_fact_chars=build_cfg["max_fact_chars"],
        min_fact_chars=build_cfg["min_fact_chars"],
        single_charge_only=build_cfg["single_charge_only"],
        top_charge_set=top_charge_set,
    )
    write_jsonl(target_dev, processed_dir / "target_dev.jsonl")
    write_jsonl(eval_test, processed_dir / "eval_test.jsonl")
    write_json(
        {
            "top_charges": top_charges,
            "train_candidates": train_candidates_count,
            "target_dev": len(target_dev),
            "eval_test": len(eval_test),
        },
        processed_dir / "charge_vocab.json",
    )

    print(f"Saved train candidates: {train_candidates_count}")
    print(f"Saved target dev: {len(target_dev)}")
    print(f"Saved eval test: {len(eval_test)}")


if __name__ == "__main__":
    main()
