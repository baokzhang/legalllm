from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from common import load_config, project_root


def prefetch_model(repo_id: str) -> str:
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        resume_download=True,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "project.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    base_model = config["models"]["base_model"]
    embedding_model = config["models"]["embedding_model"]

    base_path = prefetch_model(base_model)
    embedding_path = prefetch_model(embedding_model)

    print(f"Prefetched base model to: {base_path}")
    print(f"Prefetched embedding model to: {embedding_path}")


if __name__ == "__main__":
    main()
