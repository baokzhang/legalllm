from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import ensure_dir, load_config, project_root, read_jsonl, resolve_path, write_jsonl


def build_case_prompt(user_task_prefix: str, feature_content: str) -> str:
    return f"{user_task_prefix}\n\n{feature_content}"


def build_gold_answer(row: dict[str, Any]) -> dict[str, str]:
    charges = "；".join(row["charges"])
    articles = "；".join([f"刑法第{article}条" for article in row["articles"]]) if row["articles"] else ""
    reason_lines = [
        "1. 根据案件事实描述，需围绕行为方式、侵害对象与后果判断涉嫌罪名。",
        f"2. 结合数据标注，该案对应的罪名为：{charges}。",
    ]
    if articles:
        reason_lines.append(f"3. 相关法条包括：{articles}。")
    return {
        "charges": charges,
        "articles": articles,
        "reason": "\n".join(reason_lines),
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
        help="Defaults to eval_test.jsonl",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Defaults to data/eval/legal_generation_eval.jsonl",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    eval_dir = ensure_dir(resolve_path(config["paths"]["eval_dir"]))
    user_task_prefix = config["prompts"]["legal_user_task"].strip()
    system_prompt = config["prompts"]["legal_system"].strip()

    input_file = (
        Path(args.input_file)
        if args.input_file
        else processed_dir / "eval_test.jsonl"
    )
    eval_rows = read_jsonl(input_file)

    records = []
    for row in eval_rows:
        records.append(
            {
                "id": row["id"],
                "fact": row["fact"],
                "feature_content": row["feature_content"],
                "prompt": build_case_prompt(user_task_prefix, row["feature_content"]),
                "system_prompt": system_prompt,
                "gold": build_gold_answer(row),
                "gold_charges": row["charges"],
                "gold_articles": row["articles"],
            }
        )

    output_path = Path(args.output_file) if args.output_file else eval_dir / "legal_generation_eval.jsonl"
    write_jsonl(records, output_path)
    print(f"Saved legal generation eval dataset: {output_path}")
    print(f"Saved generation eval rows: {len(records)}")


if __name__ == "__main__":
    main()
