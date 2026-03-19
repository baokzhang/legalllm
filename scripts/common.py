from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return project_root().parent


def resolve_path(relative_or_abs: str | Path) -> Path:
    path = Path(relative_or_abs)
    if path.is_absolute():
        return path
    return project_root() / path


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(input_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(input_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_json(input_path: str | Path) -> Any:
    with Path(input_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
