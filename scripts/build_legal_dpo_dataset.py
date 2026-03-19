from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common import load_config, project_root, read_json, read_jsonl, resolve_path, write_jsonl


def build_case_prompt(user_task_prefix: str, feature_content: str) -> str:
    return f"{user_task_prefix}\n\n{feature_content}"


def build_chosen_answer(row: dict[str, Any]) -> dict[str, str]:
    charges = "；".join(row["charges"])
    articles = "；".join([f"刑法第{article}条" for article in row["articles"]]) if row["articles"] else ""
    return {
        "charges": charges,
        "articles": articles,
        "reason": "\n".join(
            [
                "1. 需要结合行为方式、主观故意和危害结果判断涉嫌罪名。",
                f"2. 本案更符合的数据标注罪名为：{charges}。",
                f"3. 对应参考法条为：{articles or '未提供明确法条标注'}。",
            ]
        ),
    }


def build_rejected_answer(
    row: dict[str, Any],
    top_charges: list[str],
    rng: random.Random,
) -> dict[str, str]:
    wrong_charge = row["charges"][0]
    while wrong_charge == row["charges"][0]:
        wrong_charge = rng.choice(top_charges)
    return {
        "charges": wrong_charge,
        "articles": "",
        "reason": "\n".join(
            [
                "1. 仅根据表面案情给出结论，未充分核对行为要件。",
                f"2. 直接判断为：{wrong_charge}。",
                "3. 未引用明确法条，依据较弱。",
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "project.yaml"),
    )
    parser.add_argument(
        "--input_file",
        default=None,
        help="Defaults to aligned_train_selected.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    user_task_prefix = config["prompts"]["legal_user_task"].strip()
    system_prompt = config["prompts"]["legal_system"].strip()
    charge_vocab = read_json(processed_dir / "charge_vocab.json")
    top_charges = charge_vocab["top_charges"]
    rng = random.Random(args.seed)

    input_file = (
        Path(args.input_file)
        if args.input_file
        else processed_dir / "aligned_train_selected.jsonl"
    )
    cases = read_jsonl(input_file)

    dpo_rows = []
    for row in cases:
        question = build_case_prompt(user_task_prefix, row["feature_content"])
        chosen = json.dumps(build_chosen_answer(row), ensure_ascii=False)
        rejected = json.dumps(
            build_rejected_answer(row, top_charges, rng), ensure_ascii=False
        )
        dpo_rows.append(
            {
                "system": system_prompt,
                "history": [],
                "question": question,
                "response_chosen": chosen,
                "response_rejected": rejected,
            }
        )

    write_jsonl(dpo_rows, processed_dir / "dpo" / "legal_dpo_pairs.jsonl")
    print(f"Saved DPO rows: {len(dpo_rows)}")


if __name__ == "__main__":
    main()
