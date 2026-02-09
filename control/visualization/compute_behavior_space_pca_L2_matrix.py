import numpy as np
from sklearn.decomposition import PCA

def pca_and_pairwise_l2(
    behavior_vectors: np.ndarray,
    dims=(2, 6, 10),
    center: bool = True,
):
    """
    behavior_vectors: (M, D) numpy array
    returns:
      embeddings_dict: {k: (M, k)}
      distances_dict:  {k: (M, M)}  pairwise L2 distance matrix in PCA space
      pca_dict:        {k: PCA object}
    """
    X = np.asarray(behavior_vectors, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"behavior_vectors must be 2D (M, D), got shape {X.shape}")

    # Optional centering (PCA in sklearn already centers internally,
    # but centering here makes it explicit and helps if you later swap PCA impls).
    if center:
        X = X - X.mean(axis=0, keepdims=True)

    embeddings_dict = {}
    distances_dict = {}
    pca_dict = {}

    for k in dims:
        if k > X.shape[1]:
            raise ValueError(f"n_components={k} > feature_dim={X.shape[1]}")

        pca = PCA(n_components=k, svd_solver="auto", random_state=0)
        Z = pca.fit_transform(X)  # (M, k)

        # Pairwise L2 distances in PCA space: ||z_i - z_j||_2
        # Efficient computation via (a-b)^2 = a^2 + b^2 - 2ab
        sq = np.sum(Z * Z, axis=1, keepdims=True)            # (M, 1)
        dist2 = np.maximum(sq + sq.T - 2.0 * (Z @ Z.T), 0.0) # (M, M)
        Dmat = np.sqrt(dist2)

        embeddings_dict[k] = Z
        distances_dict[k] = Dmat
        pca_dict[k] = pca

        print(f"[PCA{k}] explained_variance_ratio sum = {pca.explained_variance_ratio_.sum():.4f}")

    return embeddings_dict, distances_dict, pca_dict


# ===== Example integration inside your main() after you have behavior_vectors =====
# Suppose you already have:
#   behavior_vectors: List[np.ndarray]  (each is a 1D vector)
#
# Add:
# X = np.stack(behavior_vectors, axis=0)   # (M, D)

# embeddings_dict, distances_dict, pca_dict = pca_and_pairwise_l2(X, dims=(2, 6, 10))

# Optionally save results:
# out_dir = args.output_path.parent if hasattr(args, "output_path") else Path(".")
# out_dir.mkdir(parents=True, exist_ok=True)
# for k in (2, 6, 10):
#     np.save(out_dir / f"policy_embeddings_pca{k}.npy", embeddings_dict[k])
#     np.save(out_dir / f"pairwise_l2_pca{k}.npy", distances_dict[k])
