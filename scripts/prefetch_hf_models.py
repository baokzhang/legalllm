from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from huggingface_hub import snapshot_download

from common import ensure_dir, load_config, project_root, resolve_path, write_json

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
    processed_dir = ensure_dir(resolve_path(config["paths"]["processed_dir"]))
    manifest_path = processed_dir / "prefetched_model_paths.json"
    shell_path = processed_dir / "prefetched_model_paths.sh"

    write_json(
        {
            "source": source,
            "base_model_id": base_model,
            "base_model_path": base_path,
            "embedding_model_id": embedding_model,
            "embedding_model_path": embedding_path,
        },
        manifest_path,
    )
    shell_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"export PREFETCHED_MODEL_SOURCE={shlex.quote(source)}",
                f"export BASE_MODEL_PATH={shlex.quote(base_path)}",
                f"export EMBEDDING_MODEL_PATH={shlex.quote(embedding_path)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Prefetched base model to: {base_path}")
    print(f"Prefetched embedding model to: {embedding_path}")
    print(f"Saved model path manifest to: {manifest_path}")
    print(f"Saved model path env file to: {shell_path}")


if __name__ == "__main__":
    main()
