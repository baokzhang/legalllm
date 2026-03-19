from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    load_config,
    project_root,
    read_jsonl,
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

    train_rows = read_jsonl(raw_dir / "train.jsonl")
    valid_rows = read_jsonl(raw_dir / "validation.jsonl")
    test_file = raw_dir / "test.jsonl"
    test_rows = read_jsonl(test_file) if test_file.exists() else []

    normalized_train = [
        normalize_case(row, idx, build_cfg["max_fact_chars"])
        for idx, row in enumerate(train_rows)
    ]

    charge_counter = Counter()
    for item in normalized_train:
        if len(item["charges"]) == 1:
            charge_counter[item["charges"][0]] += 1
    top_charges = [name for name, _ in charge_counter.most_common(build_cfg["top_k_charges"])]
    top_charge_set = set(top_charges)

    normalized_valid = [
        normalize_case(row, idx, build_cfg["max_fact_chars"])
        for idx, row in enumerate(valid_rows)
    ]
    normalized_test = [
        normalize_case(row, idx, build_cfg["max_fact_chars"])
        for idx, row in enumerate(test_rows)
    ]

    train_candidates = [
        item
        for item in normalized_train
        if filter_case(
            item,
            top_charge_set,
            build_cfg["min_fact_chars"],
            build_cfg["single_charge_only"],
        )
    ]
    target_dev = [
        item
        for item in normalized_valid[: build_cfg["max_target_samples"]]
        if filter_case(
            item,
            top_charge_set,
            build_cfg["min_fact_chars"],
            build_cfg["single_charge_only"],
        )
    ]
    eval_test = [
        item
        for item in (normalized_test or normalized_valid[build_cfg["max_target_samples"] :])
        if filter_case(
            item,
            top_charge_set,
            build_cfg["min_fact_chars"],
            build_cfg["single_charge_only"],
        )
    ]

    write_jsonl(train_candidates, processed_dir / "train_candidates.jsonl")
    write_jsonl(target_dev, processed_dir / "target_dev.jsonl")
    write_jsonl(eval_test, processed_dir / "eval_test.jsonl")
    write_json(
        {
            "top_charges": top_charges,
            "train_candidates": len(train_candidates),
            "target_dev": len(target_dev),
            "eval_test": len(eval_test),
        },
        processed_dir / "charge_vocab.json",
    )

    print(f"Saved train candidates: {len(train_candidates)}")
    print(f"Saved target dev: {len(target_dev)}")
    print(f"Saved eval test: {len(eval_test)}")


if __name__ == "__main__":
    main()
