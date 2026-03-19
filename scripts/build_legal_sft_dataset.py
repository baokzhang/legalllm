from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    ensure_dir,
    load_config,
    project_root,
    read_jsonl,
    resolve_path,
    write_jsonl,
)


def build_case_prompt(user_task_prefix: str, feature_content: str) -> str:
    return f"{user_task_prefix}\n\n{feature_content}"


def build_gold_answer(row: dict[str, Any]) -> str:
    charges = "；".join(row["charges"])
    articles = "；".join([f"刑法第{article}条" for article in row["articles"]]) if row["articles"] else ""
    reason_lines = [
        "1. 根据案件事实描述，需围绕行为方式、侵害对象与后果判断涉嫌罪名。",
        f"2. 结合数据标注，该案对应的罪名为：{charges}。",
    ]
    if articles:
        reason_lines.append(f"3. 相关法条包括：{articles}。")
    return json.dumps(
        {
            "charges": charges,
            "articles": articles,
            "reason": "\n".join(reason_lines),
        },
        ensure_ascii=False,
    )


def convert_disc_qa_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    outputs = []
    for row in rows[:max_samples]:
        question = str(row.get("input", "")).strip()
        answer = str(row.get("output", "")).strip()
        if not question or not answer:
            continue
        outputs.append(
            {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ]
            }
        )
    return outputs


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
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    build_cfg = config["build"]
    user_task_prefix = config["prompts"]["legal_user_task"].strip()
    system_prompt = config["prompts"]["legal_system"].strip()

    input_file = (
        Path(args.input_file)
        if args.input_file
        else processed_dir / "aligned_train_selected.jsonl"
    )
    cases = read_jsonl(input_file)

    sft_rows = []
    for row in cases:
        sft_rows.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": build_case_prompt(user_task_prefix, row["feature_content"]),
                    },
                    {
                        "from": "gpt",
                        "value": build_gold_answer(row),
                    },
                ],
                "system_prompt": system_prompt,
            }
        )

    disc_path = (
        resolve_path(config["paths"]["raw_dir"])
        / "disc_law_sft"
        / config["datasets"]["disc_law_sft_file"]
    )
    if disc_path.exists():
        sft_rows.extend(
            convert_disc_qa_rows(disc_path, build_cfg["mix_disc_law_sft_samples"])
        )

    sft_dir = ensure_dir(processed_dir / "sft")
    write_jsonl(sft_rows, sft_dir / "legal_sft_sharegpt.jsonl")
    print(f"Saved SFT rows: {len(sft_rows)}")


if __name__ == "__main__":
    main()
