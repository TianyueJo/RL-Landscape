import json
from pathlib import Path

import numpy as np


def compute_l2_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    diffs = embeddings[:, None, :] - embeddings[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def main() -> None:
    embeddings_dir = Path("behavior_space_embeddings")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Missing directory: {embeddings_dir}")

    # Only process embedding JSON files (skip *_l2_dist.json)
    json_files = sorted(
        p for p in embeddings_dir.glob("*.json") if not p.name.endswith("_l2_dist.json")
    )
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {embeddings_dir}")

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        # Skip non-embedding JSONs safely
        if "embeddings" not in payload:
            print(f"[Skip] No embeddings field: {json_path}")
            continue

        embeddings = np.asarray(payload.get("embeddings", []), dtype=np.float64)
        if embeddings.ndim != 2 or embeddings.size == 0:
            print(f"[Skip] Invalid embeddings in {json_path}")
            continue

        distance_matrix = compute_l2_distance_matrix(embeddings)

        output_payload = {
            "case_id": payload.get("case_id"),
            "method": payload.get("method"),
            "dim": payload.get("dim"),
            "seeds": payload.get("seeds", []),
            "model_paths": payload.get("model_paths", []),
            "distance_matrix": distance_matrix.tolist(),
        }

        output_path = json_path.with_name(f"{json_path.stem}_l2_dist.json")
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()







