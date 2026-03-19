from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download

from common import ensure_dir, load_config, project_root, resolve_path


def export_dataset_dict(dataset_dict: DatasetDict, output_dir: Path) -> None:
    for split_name, dataset in dataset_dict.items():
        output_file = output_dir / f"{split_name}.jsonl"
        dataset.to_json(str(output_file), force_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "project.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = ensure_dir(resolve_path(config["paths"]["raw_dir"]))

    cail_dir = ensure_dir(raw_dir / "cail2018")
    cail_repo = config["datasets"]["cail2018_repo"]
    cail_data = load_dataset(cail_repo)
    export_dataset_dict(cail_data, cail_dir)

    disc_dir = ensure_dir(raw_dir / "disc_law_sft")
    hf_hub_download(
        repo_id=config["datasets"]["disc_law_sft_repo"],
        filename=config["datasets"]["disc_law_sft_file"],
        repo_type="dataset",
        local_dir=str(disc_dir),
        local_dir_use_symlinks=False,
    )

    print(f"Saved CAIL2018 splits to: {cail_dir}")
    print(f"Saved DISC-Law-SFT file to: {disc_dir}")


if __name__ == "__main__":
    main()
