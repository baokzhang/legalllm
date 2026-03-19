from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from common import load_config, project_root, read_json, read_jsonl, resolve_path, write_jsonl


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "project.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    build_cfg = config["build"]

    candidates = read_jsonl(processed_dir / "train_candidates.jsonl")
    targets = read_jsonl(processed_dir / "target_dev.jsonl")
    charge_vocab = read_json(processed_dir / "charge_vocab.json")

    embedding_model = config["models"]["embedding_model"]
    model = SentenceTransformer(embedding_model)

    candidate_embeddings = encode_texts(model, [row["feature_content"] for row in candidates])
    target_embeddings = encode_texts(model, [row["feature_content"] for row in targets])

    similarity = np.matmul(candidate_embeddings, target_embeddings.T)
    top_k = min(build_cfg["align_top_k"], similarity.shape[1])
    top_indices = np.argpartition(-similarity, kth=top_k - 1, axis=1)[:, :top_k]

    scored_rows: list[dict] = []
    for idx, candidate in enumerate(candidates):
        scores = similarity[idx, top_indices[idx]]
        avg_score = float(scores.mean())
        scored_rows.append(
            {
                **candidate,
                "avg_score": avg_score,
            }
        )

    scored_rows.sort(key=lambda item: item["avg_score"], reverse=True)
    selected = scored_rows[: build_cfg["select_top_n"]]

    for item in selected:
        item.pop("avg_score", None)

    write_jsonl(selected, processed_dir / "aligned_train_selected.jsonl")
    print(f"Top charges in project: {len(charge_vocab['top_charges'])}")
    print(f"Selected aligned cases: {len(selected)}")


if __name__ == "__main__":
    main()
