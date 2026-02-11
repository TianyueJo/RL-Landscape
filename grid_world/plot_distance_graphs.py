import json
from pathlib import Path

import numpy as np
import networkx as nx


def load_eval_returns(eval_path: Path) -> dict:
    with eval_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    eval_returns = {}
    for case_key, case_data in data.items():
        # evaluation JSON may contain metadata keys (e.g., "action_mode").
        if not isinstance(case_key, str) or not case_key.startswith("case_"):
            continue
        parts = case_key.split("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        case_id = int(parts[1])
        case_map = {}
        for seed_key, seed_data in case_data.get("seeds_data", {}).items():
            seed = int(seed_key.split("_")[1])
            case_map[seed] = float(seed_data["mean_return"])
        eval_returns[case_id] = case_map
    return eval_returns


def load_pca2_positions(embeddings_dir: Path, case_id: int) -> dict:
    """
    Load 2D PCA embeddings and return {seed: (x, y)} for node placement.
    """
    pca2_path = embeddings_dir / f"pca_dim2_case{case_id}.json"
    if not pca2_path.exists():
        raise FileNotFoundError(
            f"Missing PCA-2D embeddings for case {case_id}: {pca2_path}"
        )

    with pca2_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if int(payload.get("case_id", -1)) != int(case_id):
        raise ValueError(f"Case mismatch in {pca2_path}")
    if payload.get("method") != "pca" or int(payload.get("dim", -1)) != 2:
        raise ValueError(f"Expected PCA dim=2 embeddings in {pca2_path}")

    seeds = payload.get("seeds", [])
    embeddings = payload.get("embeddings", [])
    if len(seeds) != len(embeddings):
        raise ValueError(f"Invalid seeds/embeddings length in {pca2_path}")

    pos_map = {}
    for s, xy in zip(seeds, embeddings):
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            raise ValueError(f"Invalid 2D embedding for seed {s} in {pca2_path}")
        pos_map[int(s)] = (float(xy[0]), float(xy[1]))

    return pos_map


def build_graph(distances: np.ndarray, threshold: float) -> nx.Graph:
    n = distances.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(distances[i, j])
            if d < threshold:
                g.add_edge(i, j, distance=d)
    return g


def pairwise_distances_upper(distances: np.ndarray) -> np.ndarray:
    """
    Extract the upper-triangular pairwise distances (excluding diagonal).
    Returns shape (M,) where M = n*(n-1)/2.
    """
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(f"distances must be square; got shape={distances.shape}")
    n = distances.shape[0]
    if n < 2:
        return np.asarray([], dtype=np.float64)
    tri = distances[np.triu_indices(n, 1)]
    return np.asarray(tri, dtype=np.float64)


def threshold_by_gmm(pairwise_distances: np.ndarray, n_components: int = 2) -> float:
    """
    Fit a 2-component Gaussian Mixture to the 1D distance distribution and return
    the intersection x where w1*N(mu1,s1) == w2*N(mu2,s2).
    """
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import brentq
    from scipy.stats import norm

    x = np.asarray(pairwise_distances, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError(f"Not enough pairwise distances for GMM: n={x.size}")

    X = x.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=0,
    )
    gmm.fit(X)

    means = gmm.means_.reshape(-1)
    covs = gmm.covariances_.reshape(-1)
    stds = np.sqrt(np.maximum(covs, 1e-12))
    weights = gmm.weights_.reshape(-1)

    if means.size != 2 or stds.size != 2 or weights.size != 2:
        raise ValueError(
            f"Expected 2 components; got means={means.shape}, stds={stds.shape}, weights={weights.shape}"
        )

    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]

    def pdf_diff(t: float) -> float:
        return float(
            weights[0] * norm.pdf(t, means[0], stds[0])
            - weights[1] * norm.pdf(t, means[1], stds[1])
        )

    lo = float(means[0])
    hi = float(means[1])

    # Standard case: intersection between the two means.
    f_lo = pdf_diff(lo)
    f_hi = pdf_diff(hi)
    if np.sign(f_lo) != np.sign(f_hi):
        return float(brentq(pdf_diff, lo, hi))

    # Robust fallback: find any sign change on a grid spanning the data.
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    grid = np.linspace(x_min, x_max, 256, dtype=np.float64)
    vals = np.asarray([pdf_diff(float(t)) for t in grid], dtype=np.float64)
    s = np.sign(vals)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        raise ValueError("Failed to bracket GMM PDF intersection (no sign change found).")
    a = float(grid[int(idx[0])])
    b = float(grid[int(idx[0]) + 1])
    return float(brentq(pdf_diff, a, b))


def plot_graph(
    distances: np.ndarray,
    seeds: list,
    node_returns: list,
    node_xy: np.ndarray,
    title: str,
    output_path: Path,
    threshold_method: str = "mean",
) -> None:
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale

    pairwise = pairwise_distances_upper(distances)
    if pairwise.size == 0:
        raise ValueError("Need at least 2 nodes to build a distance graph.")

    if threshold_method == "gmm":
        try:
            threshold = float(threshold_by_gmm(pairwise, n_components=2))
        except Exception as e:
            threshold = float(np.mean(pairwise))
            print(f"[Warning] GMM threshold failed ({e}); fallback to mean={threshold:.6g}")
    elif threshold_method == "mean":
        threshold = float(np.mean(pairwise))
    else:
        raise ValueError(f"Unknown threshold_method={threshold_method!r} (expected 'mean' or 'gmm')")

    min_dist = float(np.min(pairwise))
    max_dist = float(np.max(pairwise))

    graph = build_graph(distances, threshold)
    n = distances.shape[0]
    if node_xy.shape != (n, 2):
        raise ValueError(f"node_xy must have shape (n,2); got {node_xy.shape}")
    x = node_xy[:, 0]
    y = node_xy[:, 1]

    edge_traces = []
    for u, v, data in graph.edges(data=True):
        d = float(data.get("distance", 0.0))
        norm = 0.0 if max_dist == min_dist else (d - min_dist) / (max_dist - min_dist)
        color = sample_colorscale("Viridis", [norm])[0]
        edge_traces.append(
            go.Scatter(
                x=[x[u], x[v]],
                y=[y[u], y[v]],
                mode="lines",
                line=dict(width=2, color=color),
                hoverinfo="none",
                showlegend=False,
            )
        )

    edge_colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            size=0,
            color=[min_dist, max_dist],
            colorscale="Viridis",
            cmin=min_dist,
            cmax=max_dist,
            colorbar=dict(
                title="L2 Distance",
                titleside="right",
                x=1.02,
                y=0.5,
                len=0.7,
                thickness=14,
                titlefont=dict(size=11),
                tickfont=dict(size=10),
            ),
        ),
        showlegend=False,
        hoverinfo="none",
    )

    node_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=[f"S{seed}" for seed in seeds],
        textposition="top center",
        marker=dict(
            size=18,
            color=node_returns,
            colorscale="Plasma",
            colorbar=dict(
                title="Mean Return",
                titleside="right",
                x=1.14,
                y=0.5,
                len=0.7,
                thickness=14,
                titlefont=dict(size=11),
                tickfont=dict(size=10),
            ),
            line=dict(width=1, color="black"),
        ),
        hovertemplate="Seed: %{text}<br>Return: %{marker.color:.2f}<extra></extra>",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [edge_colorbar_trace, node_trace])
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=850,
        height=780,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.write_html(str(output_path))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot distance graphs from *_l2_dist.json (HTML)."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="behavior_space_embeddings",
        help="Directory containing embedding/distance JSON files.",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default="evaluation_results.json",
        help="Evaluation JSON (used to color nodes by mean return).",
    )
    parser.add_argument(
        "--case-id",
        type=int,
        default=None,
        help="Only plot the specified case_id (omit to plot all).",
    )
    parser.add_argument(
        "--threshold-method",
        type=str,
        default="mean",
        choices=["mean", "gmm"],
        help=(
            "How to choose the edge threshold. "
            "'mean' uses the mean of upper-triangular pairwise distances; "
            "'gmm' fits a 2-component GaussianMixture to pairwise distances and uses "
            "the intersection of the two weighted Gaussian PDFs (fallbacks to mean on failure)."
        ),
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    eval_path = Path(args.eval_path)

    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Missing directory: {embeddings_dir}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {eval_path}")

    eval_returns = load_eval_returns(eval_path)

    dist_files = sorted(embeddings_dir.glob("*_l2_dist.json"))
    if not dist_files:
        raise FileNotFoundError(f"No distance matrix files in {embeddings_dir}")

    # Node positions are always taken from PCA dim=2 (per case), loaded lazily.
    pca2_pos_by_case: dict[int, dict | None] = {}

    for dist_path in dist_files:
        with dist_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        distances = np.asarray(payload.get("distance_matrix", []), dtype=np.float64)
        seeds = payload.get("seeds", [])
        case_id = int(payload.get("case_id"))
        method = payload.get("method")
        dim = payload.get("dim")

        if args.case_id is not None and int(case_id) != int(args.case_id):
            continue

        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError(f"Invalid distance matrix in {dist_path}")

        if case_id not in eval_returns:
            raise ValueError(f"Missing case {case_id} in evaluation results")

        if case_id not in pca2_pos_by_case:
            try:
                pca2_pos_by_case[case_id] = load_pca2_positions(embeddings_dir, case_id)
            except FileNotFoundError:
                pca2_pos_by_case[case_id] = None

        pos_map = pca2_pos_by_case[case_id]
        if pos_map is None:
            raise FileNotFoundError(
                f"Missing PCA dim=2 embeddings for case {case_id}: "
                f"{embeddings_dir / f'pca_dim2_case{case_id}.json'} "
                f"(required for node positions)."
            )

        node_returns = []
        for seed in seeds:
            if seed not in eval_returns[case_id]:
                raise ValueError(f"Missing seed {seed} eval data for case {case_id}")
            node_returns.append(eval_returns[case_id][seed])

        # Align PCA-2D positions to the seeds order of this distance matrix.
        node_xy = []
        for seed in seeds:
            if int(seed) not in pos_map:
                raise ValueError(
                    f"Missing seed {seed} in PCA dim=2 embeddings for case {case_id}"
                )
            node_xy.append(pos_map[int(seed)])
        node_xy = np.asarray(node_xy, dtype=np.float64)

        title = (
            f"Distance Graph (L2, mean threshold; node pos=PCA2) - "
            f"{str(method).upper()} dim={dim} case={case_id}"
        )
        # Avoid clobbering heatmap outputs that use the same base name.
        output_path = dist_path.with_name(f"{dist_path.stem}_graph.html")
        title2 = title.replace("mean threshold", f"{args.threshold_method} threshold")
        plot_graph(
            distances,
            seeds,
            node_returns,
            node_xy,
            title2,
            output_path,
            threshold_method=args.threshold_method,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

