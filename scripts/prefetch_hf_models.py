from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

from common import load_config, project_root

try:
    from modelscope import snapshot_download as ms_snapshot_download
except ImportError:  # pragma: no cover - optional dependency in local dev
    ms_snapshot_download = None


def prefetch_model(repo_id: str, source: str) -> str:
    errors: list[str] = []
    backends = [source] if source != "auto" else ["huggingface", "modelscope"]

    for backend in backends:
        try:
            if backend == "huggingface":
                return snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    resume_download=True,
                )
            if backend == "modelscope":
                if ms_snapshot_download is None:
                    raise RuntimeError("ModelScope is not installed.")
                return ms_snapshot_download(repo_id)
            raise RuntimeError(f"Unsupported model source: {backend}")
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            errors.append(f"{backend}: {exc}")

    raise RuntimeError(f"Unable to prefetch model {repo_id}. Errors: {' | '.join(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "project.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    source = os.getenv("LEGAL_MODEL_SOURCE", config["models"].get("download_source", "auto"))
    base_model = config["models"]["base_model"]
    embedding_model = config["models"]["embedding_model"]

    base_path = prefetch_model(base_model, source)
    embedding_path = prefetch_model(embedding_model, source)

    print(f"Prefetched base model to: {base_path}")
    print(f"Prefetched embedding model to: {embedding_path}")


if __name__ == "__main__":
    main()
