from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ensure_dir, load_config, project_root, read_jsonl, resolve_path, write_json, write_jsonl


ARTICLE_RE = re.compile(r"第?\s*(\d+)\s*条")


def normalize_text(value: str) -> str:
    text = str(value).strip()
    text = text.replace(" ", "")
    return text


def normalize_charge(value: Any) -> str:
    if isinstance(value, list):
        value = "；".join([str(item).strip() for item in value if str(item).strip()])
    return normalize_text(str(value))


def extract_article_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    article_ids: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        matched = ARTICLE_RE.findall(text)
        if matched:
            article_ids.extend(matched)
            continue
        if text.isdigit():
            article_ids.append(text)
    # stable unique
    seen: set[str] = set()
    unique_ids: list[str] = []
    for article_id in article_ids:
        if article_id not in seen:
            seen.add(article_id)
            unique_ids.append(article_id)
    return unique_ids


def extract_json_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_prediction(text: str) -> dict[str, Any] | None:
    json_block = extract_json_block(text)
    if not json_block:
        return None
    try:
        return json.loads(json_block)
    except json.JSONDecodeError:
        return None


def reason_is_structured(reason: Any) -> bool:
    if not isinstance(reason, str):
        return False
    lines = [line.strip() for line in reason.splitlines() if line.strip()]
    numbered = [line for line in lines if re.match(r"^\d+[\.、]", line)]
    return len(numbered) >= 2


def batch_iter(rows: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def build_messages(system_prompt: str, prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    device: str,
) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(row["system_prompt"], row["prompt"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
    outputs: list[str] = []
    for idx, seq in enumerate(generated):
        new_tokens = seq[int(input_lengths[idx]) :]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "project.yaml"),
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_dir = resolve_path(config["paths"]["eval_dir"])
    input_file = Path(args.input_file) if args.input_file else eval_dir / "legal_generation_eval.jsonl"
    rows = read_jsonl(input_file)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    output_dir = ensure_dir(Path(args.output_dir))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map.get(args.torch_dtype, torch.bfloat16),
        trust_remote_code=True,
        device_map={"": args.device},
    )
    model.eval()

    prediction_rows: list[dict[str, Any]] = []
    stats = {
        "total": len(rows),
        "json_valid": 0,
        "charges_exact": 0,
        "articles_exact": 0,
        "articles_contains_gold": 0,
        "reason_structured": 0,
        "all_core_fields_correct": 0,
    }

    for batch in batch_iter(rows, args.batch_size):
        outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            rows=batch,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        for row, raw_output in zip(batch, outputs):
            parsed = parse_prediction(raw_output)
            json_valid = parsed is not None
            gold_charge = normalize_charge(row["gold"]["charges"])
            pred_charge = normalize_charge(parsed.get("charges", "")) if parsed else ""
            gold_articles = extract_article_ids(row["gold_articles"])
            pred_articles = extract_article_ids(parsed.get("articles", "")) if parsed else []
            reason_structured = reason_is_structured(parsed.get("reason")) if parsed else False

            charges_exact = pred_charge == gold_charge
            articles_exact = pred_articles == gold_articles
            articles_contains_gold = set(gold_articles).issubset(set(pred_articles))
            all_core_fields_correct = json_valid and charges_exact and articles_contains_gold and reason_structured

            stats["json_valid"] += int(json_valid)
            stats["charges_exact"] += int(charges_exact)
            stats["articles_exact"] += int(articles_exact)
            stats["articles_contains_gold"] += int(articles_contains_gold)
            stats["reason_structured"] += int(reason_structured)
            stats["all_core_fields_correct"] += int(all_core_fields_correct)

            prediction_rows.append(
                {
                    "id": row["id"],
                    "fact": row["fact"],
                    "prompt": row["prompt"],
                    "gold": row["gold"],
                    "raw_output": raw_output,
                    "parsed_output": parsed,
                    "metrics": {
                        "json_valid": json_valid,
                        "charges_exact": charges_exact,
                        "articles_exact": articles_exact,
                        "articles_contains_gold": articles_contains_gold,
                        "reason_structured": reason_structured,
                        "all_core_fields_correct": all_core_fields_correct,
                    },
                }
            )

    denominator = max(stats["total"], 1)
    summary = {
        "model_path": args.model_path,
        "input_file": str(input_file),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "metrics": {
            "json_valid_rate": stats["json_valid"] / denominator,
            "charges_exact_rate": stats["charges_exact"] / denominator,
            "articles_exact_rate": stats["articles_exact"] / denominator,
            "articles_contains_gold_rate": stats["articles_contains_gold"] / denominator,
            "reason_structured_rate": stats["reason_structured"] / denominator,
            "all_core_fields_correct_rate": stats["all_core_fields_correct"] / denominator,
        },
        "counts": stats,
    }

    write_jsonl(prediction_rows, output_dir / "predictions.jsonl")
    write_json(summary, output_dir / "summary.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
