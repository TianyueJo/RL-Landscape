import json
import argparse
from pathlib import Path

import numpy as np


def compute_distance_matrix(embeddings: np.ndarray, norm: str) -> np.ndarray:
    diffs = embeddings[:, None, :] - embeddings[None, :, :]
    norm = str(norm).lower()
    if norm == "l2":
        return np.linalg.norm(diffs, axis=-1)
    if norm == "l1":
        return np.sum(np.abs(diffs), axis=-1)
    raise ValueError(f"Unknown norm={norm!r} (expected 'l1' or 'l2')")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute pairwise distance matrices for embedding JSONs.")
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="behavior_space_embeddings",
        help="Directory containing embedding JSON files (e.g., pca_dim*_case*.json).",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="l2",
        choices=["l1", "l2"],
        help="Distance norm to use.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output files that already exist.",
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Missing directory: {embeddings_dir}")

    # Only process embedding JSON files (skip *_l{1,2}_dist.json)
    json_files = sorted(
        p
        for p in embeddings_dir.glob("*.json")
        if not (p.name.endswith("_l2_dist.json") or p.name.endswith("_l1_dist.json"))
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

        out_path = json_path.with_name(f"{json_path.stem}_{args.norm}_dist.json")
        if args.skip_existing and out_path.exists():
            print(f"[Skip] Exists: {out_path}")
            continue

        distance_matrix = compute_distance_matrix(embeddings, norm=args.norm)

        output_payload = {
            "case_id": payload.get("case_id"),
            "method": payload.get("method"),
            "dim": payload.get("dim"),
            "seeds": payload.get("seeds", []),
            "model_paths": payload.get("model_paths", []),
            "distance_norm": args.norm,
            "distance_matrix": distance_matrix.tolist(),
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()







