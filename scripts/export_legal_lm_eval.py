from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import ensure_dir, load_config, project_root, read_json, read_jsonl, resolve_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "project.yaml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    eval_dir = ensure_dir(resolve_path(config["paths"]["eval_dir"]))
    task_dir = ensure_dir(resolve_path(config["paths"]["lm_eval_task_dir"]))
    build_cfg = config["build"]
    rng = random.Random(args.seed)

    eval_rows = read_jsonl(processed_dir / "eval_test.jsonl")
    charge_vocab = read_json(processed_dir / "charge_vocab.json")
    top_charges = charge_vocab["top_charges"]
    choice_count = build_cfg["lm_eval_choice_count"]

    examples = []
    for row in eval_rows:
        answer = row["charges"][0]
        negatives = [charge for charge in top_charges if charge != answer]
        if len(negatives) < choice_count - 1:
            continue
        sampled = rng.sample(negatives, choice_count - 1)
        choices = sampled + [answer]
        rng.shuffle(choices)
        examples.append(
            {
                "question": (
                    "请阅读以下刑事案件事实，并选择最可能涉嫌的罪名。\n\n"
                    f"{row['fact']}\n\n答案："
                ),
                "choices": choices,
                "answer": choices.index(answer),
            }
        )

    dataset_path = eval_dir / "legal_charge_mc.json"
    write_json(examples, dataset_path)

    yaml_path = task_dir / "legal_charge_mc.yaml"
    yaml_content = f"""task: legal_charge_mc
dataset_path: json
dataset_kwargs:
  data_files:
    test: {dataset_path}
output_type: multiple_choice
doc_to_text: "{{{{question}}}}"
doc_to_target: "{{{{answer}}}}"
doc_to_choice: "choices"
test_split: test
num_fewshot: 0
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
  description: "Chinese legal criminal charge multiple choice benchmark"
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"Saved lm-eval dataset: {dataset_path}")
    print(f"Saved lm-eval task yaml: {yaml_path}")


if __name__ == "__main__":
    main()
