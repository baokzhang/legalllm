from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from huggingface_hub import hf_hub_download

from common import ensure_dir, load_config, project_root, resolve_path

MODELSCOPE_IMPORT_ERROR = ""
try:
    from modelscope.msdatasets import MsDataset
except Exception as exc1:  # pragma: no cover - optional dependency in local dev
    try:
        from modelscope import MsDataset  # type: ignore[attr-defined]
        MODELSCOPE_IMPORT_ERROR = f"fallback import used after: {type(exc1).__name__}: {exc1}"
    except Exception as exc2:
        MsDataset = None
        MODELSCOPE_IMPORT_ERROR = (
            f"{type(exc1).__name__}: {exc1}; "
            f"fallback import failed with {type(exc2).__name__}: {exc2}"
        )


def export_dataset_dict(dataset_dict: DatasetDict, output_dir: Path) -> None:
    for split_name, dataset in dataset_dict.items():
        output_file = output_dir / f"{split_name}.jsonl"
        dataset.to_json(str(output_file), force_ascii=False)


def export_rows(rows: Iterable[dict[str, Any]], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_dataset_like(dataset_obj: Any, output_file: Path) -> None:
    ensure_dir(output_file.parent)
    hf_dataset = getattr(dataset_obj, "_hf_ds", dataset_obj)
    if isinstance(hf_dataset, (Dataset, IterableDataset)):
        hf_dataset.to_json(str(output_file), force_ascii=False)
        return
    export_rows(hf_dataset, output_file)


def copy_dir_contents(src_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def env_or_config(env_key: str, config_value: str | None, default: str = "") -> str:
    value = os.getenv(env_key)
    if value is not None:
        return value
    return str(config_value or default)


def load_ms_dataset(dataset_id: str, split: str) -> Any:
    if MsDataset is None:
        raise RuntimeError(
            "ModelScope import failed. "
            "This is often caused by an incompatible Python version in the venv. "
            f"Import detail: {MODELSCOPE_IMPORT_ERROR}"
        )
    return MsDataset.load(dataset_id, split=split)


def download_cail_from_huggingface(repo_id: str, output_dir: Path) -> None:
    dataset = load_dataset(repo_id)
    export_dataset_dict(dataset, output_dir)


def download_cail_from_modelscope(dataset_id: str, output_dir: Path) -> None:
    if not dataset_id:
        raise RuntimeError("CAIL2018 ModelScope dataset id is empty.")
    for split in ("train", "validation", "test"):
        output_file = output_dir / f"{split}.jsonl"
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"Skip existing CAIL split: {output_file}")
            continue
        try:
            dataset = load_ms_dataset(dataset_id, split)
        except Exception:
            if split == "test":
                continue
            raise
        export_dataset_like(dataset, output_file)


def prepare_cail_dataset(config: dict[str, Any], raw_dir: Path) -> None:
    datasets_cfg = config["datasets"]
    cail_dir = ensure_dir(raw_dir / "cail2018")
    source = env_or_config("LEGAL_CAIL_SOURCE", datasets_cfg.get("cail2018_source"), "auto")
    local_dir = env_or_config("LEGAL_CAIL_LOCAL_DIR", datasets_cfg.get("cail2018_local_dir"))
    modelscope_id = env_or_config(
        "LEGAL_CAIL_MODELSCOPE_ID",
        datasets_cfg.get("cail2018_modelscope_id"),
    )
    hf_repo = datasets_cfg["cail2018_repo"]

    if local_dir:
        copy_dir_contents(Path(local_dir), cail_dir)
        print(f"Loaded CAIL2018 from local dir: {local_dir}")
        return

    errors: list[str] = []
    backends = [source] if source != "auto" else ["huggingface", "modelscope"]

    for backend in backends:
        try:
            if backend == "huggingface":
                download_cail_from_huggingface(hf_repo, cail_dir)
            elif backend == "modelscope":
                download_cail_from_modelscope(modelscope_id, cail_dir)
            else:
                raise RuntimeError(f"Unsupported CAIL2018 source: {backend}")
            print(f"Saved CAIL2018 splits to: {cail_dir} (source={backend})")
            return
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            errors.append(f"{backend}: {exc}")

    raise RuntimeError(
        "Unable to download CAIL2018. "
        "You can set LEGAL_CAIL_SOURCE=modelscope with LEGAL_CAIL_MODELSCOPE_ID=<dataset_id>, "
        "or set LEGAL_CAIL_LOCAL_DIR=/path/to/cail2018. "
        f"Errors: {' | '.join(errors)}"
    )


def download_disc_from_huggingface(repo_id: str, filename: str, output_dir: Path) -> None:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )


def download_disc_from_modelscope(dataset_id: str, filename: str, output_dir: Path) -> None:
    if not dataset_id:
        raise RuntimeError("DISC-Law-SFT ModelScope dataset id is empty.")
    output_file = output_dir / filename
    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"Skip existing DISC-Law-SFT file: {output_file}")
        return
    dataset = load_ms_dataset(dataset_id, "train")
    export_dataset_like(dataset, output_file)


def prepare_disc_dataset(config: dict[str, Any], raw_dir: Path) -> None:
    datasets_cfg = config["datasets"]
    disc_dir = ensure_dir(raw_dir / "disc_law_sft")
    filename = datasets_cfg["disc_law_sft_file"]
    source = env_or_config(
        "LEGAL_DISC_SOURCE",
        datasets_cfg.get("disc_law_sft_source"),
        "auto",
    )
    local_file = env_or_config(
        "LEGAL_DISC_LOCAL_FILE",
        datasets_cfg.get("disc_law_sft_local_file"),
    )
    modelscope_id = env_or_config(
        "LEGAL_DISC_MODELSCOPE_ID",
        datasets_cfg.get("disc_law_sft_modelscope_id"),
    )
    hf_repo = datasets_cfg["disc_law_sft_repo"]

    if local_file:
        shutil.copy2(local_file, disc_dir / filename)
        print(f"Loaded DISC-Law-SFT from local file: {local_file}")
        return

    errors: list[str] = []
    backends = [source] if source != "auto" else ["huggingface", "modelscope"]

    for backend in backends:
        try:
            if backend == "huggingface":
                download_disc_from_huggingface(hf_repo, filename, disc_dir)
            elif backend == "modelscope":
                download_disc_from_modelscope(modelscope_id, filename, disc_dir)
            else:
                raise RuntimeError(f"Unsupported DISC-Law-SFT source: {backend}")
            print(f"Saved DISC-Law-SFT file to: {disc_dir} (source={backend})")
            return
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            errors.append(f"{backend}: {exc}")

    raise RuntimeError(
        "Unable to download DISC-Law-SFT. "
        "You can set LEGAL_DISC_SOURCE=modelscope with LEGAL_DISC_MODELSCOPE_ID=<dataset_id>, "
        "or set LEGAL_DISC_LOCAL_FILE=/path/to/DISC-Law-SFT-Pair-QA-released.jsonl. "
        f"Errors: {' | '.join(errors)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "project.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = ensure_dir(resolve_path(config["paths"]["raw_dir"]))

    prepare_cail_dataset(config, raw_dir)
    prepare_disc_dataset(config, raw_dir)


if __name__ == "__main__":
    main()
