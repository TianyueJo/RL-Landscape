import json
from pathlib import Path

import numpy as np
import networkx as nx


def load_eval_data(eval_path: Path) -> dict:
    with eval_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    eval_data: dict[int, dict[int, dict]] = {}
    for case_key, case_data in data.items():
        # evaluation JSON may contain metadata keys (e.g., "action_mode").
        if not isinstance(case_key, str) or not case_key.startswith("case_"):
            continue
        parts = case_key.split("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        case_id = int(parts[1])
        case_map: dict[int, dict] = {}
        for seed_key, seed_data in case_data.get("seeds_data", {}).items():
            seed = int(seed_key.split("_")[1])
            case_map[seed] = {
                "mean_return": float(seed_data.get("mean_return", 0.0)),
                "std_return": float(seed_data.get("std_return", 0.0)),
                "success_rate": float(seed_data.get("success_rate", 0.0)),
                # Optional (added later): end-position stats
                "most_common_end_position": seed_data.get("most_common_end_position", None),
                "end_position_counts": seed_data.get("end_position_counts", None),
            }
        eval_data[case_id] = case_map
    return eval_data


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
    if int(n_components) != 2:
        raise ValueError("Only n_components=2 is supported.")

    x = np.asarray(pairwise_distances, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError(f"Not enough pairwise distances for GMM: n={x.size}")

    def normal_pdf(xx: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        s = float(max(sigma, 1e-12))
        z = (xx - float(mu)) / s
        return np.exp(-0.5 * z * z) / (s * np.sqrt(2.0 * np.pi))

    def fit_gmm_1d_em(xx: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pure NumPy EM for 1D 2-component Gaussian mixture.
        Returns (weights, means, stds), each shape (2,).
        """
        rng = np.random.default_rng(0)
        xx = np.asarray(xx, dtype=np.float64)
        n = xx.size

        # init means by percentiles (more stable than random)
        m0, m1 = np.percentile(xx, [25, 75]).astype(np.float64)
        if not np.isfinite(m0) or not np.isfinite(m1) or float(m0) == float(m1):
            m0 = float(np.min(xx))
            m1 = float(np.max(xx))
            if float(m0) == float(m1):
                m0 -= 1e-3
                m1 += 1e-3

        means = np.array([m0, m1], dtype=np.float64)
        s = float(np.std(xx))
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        stds = np.array([s, s], dtype=np.float64)
        weights = np.array([0.5, 0.5], dtype=np.float64)

        prev_ll = -np.inf
        for _ in range(int(max_iter)):
            # E-step
            p0 = weights[0] * normal_pdf(xx, means[0], stds[0])
            p1 = weights[1] * normal_pdf(xx, means[1], stds[1])
            denom = p0 + p1 + 1e-300
            r0 = p0 / denom
            r1 = p1 / denom

            # M-step
            n0 = float(np.sum(r0))
            n1 = float(np.sum(r1))
            if n0 <= 1e-12 or n1 <= 1e-12:
                # degenerate: re-seed slightly and continue
                means = means + rng.normal(0.0, 1e-3, size=2)
                stds = np.maximum(stds, 1e-3)
                weights = np.array([0.5, 0.5], dtype=np.float64)
                continue

            weights = np.array([n0 / n, n1 / n], dtype=np.float64)
            means = np.array(
                [np.sum(r0 * xx) / n0, np.sum(r1 * xx) / n1], dtype=np.float64
            )
            v0 = np.sum(r0 * (xx - means[0]) ** 2) / n0
            v1 = np.sum(r1 * (xx - means[1]) ** 2) / n1
            stds = np.sqrt(np.maximum([v0, v1], 1e-12)).astype(np.float64)

            # log-likelihood for convergence
            mix = (
                weights[0] * normal_pdf(xx, means[0], stds[0])
                + weights[1] * normal_pdf(xx, means[1], stds[1])
                + 1e-300
            )
            ll = float(np.sum(np.log(mix)))
            if abs(ll - prev_ll) < float(tol) * (1.0 + abs(prev_ll)):
                break
            prev_ll = ll

        return weights, means, stds

    def intersection_of_weighted_normals(
        w0: float, m0: float, s0: float, w1: float, m1: float, s1: float
    ) -> float:
        """
        Solve w0*N(m0,s0) == w1*N(m1,s1) analytically.
        Returns a threshold, preferring a root between m0 and m1.
        """
        s0 = float(max(s0, 1e-12))
        s1 = float(max(s1, 1e-12))
        w0 = float(max(w0, 1e-300))
        w1 = float(max(w1, 1e-300))

        # Solve:
        # log(w0) - log(s0) - (x-m0)^2/(2s0^2) = log(w1) - log(s1) - (x-m1)^2/(2s1^2)
        a = (1.0 / (2.0 * s1 * s1)) - (1.0 / (2.0 * s0 * s0))
        b = (m0 / (s0 * s0)) - (m1 / (s1 * s1))
        c = (
            (m1 * m1) / (2.0 * s1 * s1)
            - (m0 * m0) / (2.0 * s0 * s0)
            + np.log(w1) - np.log(w0)
            + np.log(s0) - np.log(s1)
        )

        if abs(a) < 1e-14:
            # linear
            if abs(b) < 1e-14:
                return float(0.5 * (m0 + m1))
            return float(-c / b)

        disc = b * b - 4.0 * a * c
        disc = float(max(disc, 0.0))
        rdisc = float(np.sqrt(disc))
        x1 = (-b - rdisc) / (2.0 * a)
        x2 = (-b + rdisc) / (2.0 * a)

        lo = float(min(m0, m1))
        hi = float(max(m0, m1))
        cand = []
        for xx in (x1, x2):
            if np.isfinite(xx):
                cand.append(float(xx))
        if not cand:
            return float(0.5 * (m0 + m1))

        between = [xx for xx in cand if lo <= xx <= hi]
        if between:
            # if two, pick the one closer to the midpoint
            mid = 0.5 * (lo + hi)
            return float(min(between, key=lambda t: abs(t - mid)))

        # otherwise pick the root closest to the interval
        def dist_to_interval(t: float) -> float:
            if t < lo:
                return lo - t
            if t > hi:
                return t - hi
            return 0.0

        return float(min(cand, key=dist_to_interval))

    weights, means, stds = fit_gmm_1d_em(x)
    order = np.argsort(means)
    weights, means, stds = weights[order], means[order], stds[order]

    return intersection_of_weighted_normals(
        float(weights[0]),
        float(means[0]),
        float(stds[0]),
        float(weights[1]),
        float(means[1]),
        float(stds[1]),
    )


def plot_graph(
    distances: np.ndarray,
    seeds: list,
    node_returns: list,
    node_endpoints: list | None,
    node_success_rates: list | None,
    node_endpoint_counts: list | None,
    node_xy: np.ndarray,
    title: str,
    output_path: Path,
    threshold_method: str = "mean",
    image_scale: float = 2.0,
    distance_label: str = "L2",
    color_by: str = "return",
) -> None:
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    from plotly.colors import qualitative

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
                title=f"{distance_label} Distance",
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

    node_traces = []
    if color_by == "return":
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
        node_traces.append(node_trace)
    elif color_by == "endpoint":
        if node_endpoints is None:
            raise ValueError("color_by='endpoint' requires node_endpoints.")
        if node_success_rates is None:
            node_success_rates = [None] * len(seeds)

        # Assign a discrete color to each distinct endpoint label.
        # Endpoint label format: "x,y" (or "unknown").
        endpoints = [str(e) if e is not None else "unknown" for e in node_endpoints]
        uniq = sorted(set(endpoints))
        palette = list(qualitative.Dark24) + list(qualitative.Set3) + list(qualitative.Plotly)
        color_map = {ep: palette[i % len(palette)] for i, ep in enumerate(uniq)}

        # Create one trace per endpoint to get a legend.
        seed_labels = [f"S{seed}" for seed in seeds]
        for ep in uniq:
            idx = [i for i, e in enumerate(endpoints) if e == ep]
            if not idx:
                continue
            hovertext = []
            for i in idx:
                ret = float(node_returns[i]) if node_returns is not None else float("nan")
                sr = node_success_rates[i]
                if sr is None:
                    hovertext.append(
                        f"Seed: {seed_labels[i]}<br>Return: {ret:.2f}<br>End: {ep}"
                    )
                else:
                    hovertext.append(
                        f"Seed: {seed_labels[i]}<br>Return: {ret:.2f}<br>End: {ep}<br>Success: {float(sr)*100:.1f}%"
                    )

            node_traces.append(
                go.Scatter(
                    x=[x[i] for i in idx],
                    y=[y[i] for i in idx],
                    mode="markers+text",
                    text=[seed_labels[i] for i in idx],
                    textposition="top center",
                    marker=dict(
                        size=18,
                        color=color_map[ep],
                        line=dict(width=1, color="black"),
                    ),
                    name=f"End {ep}",
                    hoverinfo="text",
                    hovertext=hovertext,
                    showlegend=True,
                )
            )
    elif color_by == "endpoint_pie":
        # Endpoint distribution as per-node pie charts (uses end_position_counts)
        if node_endpoint_counts is None:
            raise ValueError("color_by='endpoint_pie' requires node_endpoint_counts.")

        # Collect global endpoint labels for consistent colors
        global_labels = set()
        for c in node_endpoint_counts:
            if isinstance(c, dict):
                for k in c.keys():
                    if isinstance(k, str) and k.strip():
                        global_labels.add(k.strip())
        if not global_labels:
            global_labels = {"unknown"}
        uniq = sorted(global_labels)
        palette = list(qualitative.Dark24) + list(qualitative.Set3) + list(qualitative.Plotly)
        color_map = {ep: palette[i % len(palette)] for i, ep in enumerate(uniq)}

        # Build base figure: edges + edge colorbar
        fig = go.Figure(data=edge_traces + [edge_colorbar_trace])

        # Set axis ranges (so we can map data coords -> paper coords for pie domains)
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        ymin = float(np.min(y))
        ymax = float(np.max(y))
        if xmax == xmin:
            xmax = xmin + 1.0
        if ymax == ymin:
            ymax = ymin + 1.0
        pad_x = 0.05 * (xmax - xmin)
        pad_y = 0.05 * (ymax - ymin)
        xmin -= pad_x
        xmax += pad_x
        ymin -= pad_y
        ymax += pad_y

        # Legend (one entry per endpoint label)
        for ep in uniq:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color_map[ep], line=dict(width=1, color="black")),
                    name=f"End {ep}",
                    showlegend=True,
                    hoverinfo="none",
                )
            )

        # Add a small pie chart at each node using normalized paper coordinates
        pie_size = 0.06  # in [0,1] paper coords
        seed_labels = [f"S{seed}" for seed in seeds]
        for i in range(len(seeds)):
            counts = node_endpoint_counts[i] if i < len(node_endpoint_counts) else None
            if not isinstance(counts, dict) or not counts:
                counts = {"unknown": 1}

            labels = []
            values = []
            for k, v in counts.items():
                if v is None:
                    continue
                try:
                    vv = int(v)
                except Exception:
                    continue
                if vv <= 0:
                    continue
                kk = str(k).strip() if k is not None else "unknown"
                if not kk:
                    kk = "unknown"
                labels.append(kk)
                values.append(vv)
            if not labels:
                labels = ["unknown"]
                values = [1]

            px = (float(x[i]) - xmin) / (xmax - xmin)
            py = (float(y[i]) - ymin) / (ymax - ymin)
            x0 = max(0.0, px - pie_size / 2.0)
            x1 = min(1.0, px + pie_size / 2.0)
            y0 = max(0.0, py - pie_size / 2.0)
            y1 = min(1.0, py + pie_size / 2.0)

            ret = float(node_returns[i]) if node_returns is not None else float("nan")
            sr = None
            if node_success_rates is not None and i < len(node_success_rates):
                sr = float(node_success_rates[i])

            # One customdata row per slice
            custom = []
            for _ in labels:
                custom.append([seed_labels[i], ret, sr])

            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    domain=dict(x=[x0, x1], y=[y0, y1]),
                    sort=False,
                    direction="clockwise",
                    textinfo="none",
                    showlegend=False,
                    marker=dict(colors=[color_map.get(l, "#999999") for l in labels], line=dict(width=1, color="black")),
                    customdata=custom,
                    hovertemplate=(
                        "Seed: %{customdata[0]}<br>"
                        "Return: %{customdata[1]:.2f}<br>"
                        "End: %{label}<br>"
                        "Count: %{value}<br>"
                        "Success: %{customdata[2]:.3f}<extra></extra>"
                    ),
                )
            )

        # Seed labels on top (data coords)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="text",
                text=seed_labels,
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            title=title,
            xaxis=dict(visible=False, range=[xmin, xmax]),
            yaxis=dict(visible=False, range=[ymin, ymax]),
            width=900,
            height=820,
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(
                x=1.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.6)",
            ),
        )

        if output_path.suffix.lower() == ".html":
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path), scale=float(image_scale))
        return
    else:
        raise ValueError(
            f"Unknown color_by={color_by!r} (expected 'return' or 'endpoint' or 'endpoint_pie')"
        )

    fig = go.Figure(data=edge_traces + [edge_colorbar_trace] + node_traces)
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=850,
        height=780,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    if output_path.suffix.lower() == ".html":
        fig.write_html(str(output_path))
    else:
        # Requires kaleido: `pip install kaleido`
        fig.write_image(str(output_path), scale=float(image_scale))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot distance graphs from *_{l1,l2}_dist.json (HTML/PNG/SVG/PDF/JPG)."
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
        "--only-dim",
        type=int,
        default=None,
        help="Only plot distance files with this embedding dimension (e.g., 9).",
    )
    parser.add_argument(
        "--only-norm",
        type=str,
        default=None,
        choices=[None, "l1", "l2"],
        help="Only plot distance files with this distance norm (l1 or l2).",
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
    parser.add_argument(
        "--output-format",
        type=str,
        default="html",
        choices=["html", "png", "svg", "pdf", "jpg", "jpeg"],
        help="Output format. Non-HTML formats require kaleido.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=2.0,
        help="Scale factor for raster image export (e.g., png/jpg).",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="return",
        choices=["return", "endpoint", "endpoint_pie"],
        help=(
            "Node color rule: "
            "'return' (mean return), "
            "'endpoint' (most common end position), "
            "'endpoint_pie' (pie chart of end_position_counts)."
        ),
    )
    parser.add_argument(
        "--only-method",
        type=str,
        default=None,
        choices=[None, "pca", "tsne"],
        help="Only plot distance files with this embedding method (pca or tsne).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If provided, save all outputs to this directory (instead of alongside the input JSON).",
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    eval_path = Path(args.eval_path)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Missing directory: {embeddings_dir}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {eval_path}")

    eval_data = load_eval_data(eval_path)

    dist_files = sorted(embeddings_dir.glob("*_l*_dist.json"))
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
        if args.only_dim is not None and int(dim) != int(args.only_dim):
            continue
        if args.only_method is not None and str(method).lower() != str(args.only_method).lower():
            continue

        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError(f"Invalid distance matrix in {dist_path}")

        if case_id not in eval_data:
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

        # Determine distance norm from payload / filename and filter if requested
        norm = payload.get("distance_norm")
        if norm is None:
            if "_l1_dist" in dist_path.stem:
                norm = "l1"
            elif "_l2_dist" in dist_path.stem:
                norm = "l2"
        norm_str = str(norm).lower() if norm is not None else "l2"
        if args.only_norm is not None and norm_str != str(args.only_norm).lower():
            continue

        node_returns = []
        node_endpoints = []
        node_success_rates = []
        node_endpoint_counts = []
        for seed in seeds:
            if seed not in eval_data[case_id]:
                raise ValueError(f"Missing seed {seed} eval data for case {case_id}")
            sd = eval_data[case_id][seed]
            node_returns.append(float(sd.get("mean_return", 0.0)))
            node_success_rates.append(float(sd.get("success_rate", 0.0)))
            endp = sd.get("most_common_end_position", None)
            if isinstance(endp, (list, tuple)) and len(endp) == 2:
                node_endpoints.append(f"{int(endp[0])},{int(endp[1])}")
            elif isinstance(endp, str) and endp.strip():
                node_endpoints.append(endp.strip())
            else:
                node_endpoints.append("unknown")
            epc = sd.get("end_position_counts", None)
            if isinstance(epc, dict) and epc:
                node_endpoint_counts.append(epc)
            else:
                node_endpoint_counts.append({"unknown": 1})

        # Align PCA-2D positions to the seeds order of this distance matrix.
        node_xy = []
        for seed in seeds:
            if int(seed) not in pos_map:
                raise ValueError(
                    f"Missing seed {seed} in PCA dim=2 embeddings for case {case_id}"
                )
            node_xy.append(pos_map[int(seed)])
        node_xy = np.asarray(node_xy, dtype=np.float64)

        distance_label = "L1" if norm_str == "l1" else "L2"

        title = (
            f"Distance Graph ({distance_label}, {args.threshold_method} threshold; node pos=PCA2; node color={args.color_by}) - "
            f"{str(method).upper()} dim={dim} case={case_id}"
        )
        ext = str(args.output_format).lower()
        if ext == "jpeg":
            ext = "jpg"
        suffix = f".{ext}"

        out_base = f"{dist_path.stem}_{args.threshold_method}_graph_{args.color_by}{suffix}"
        if output_dir is None:
            output_path = dist_path.with_name(out_base)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / out_base

        plot_graph(
            distances,
            seeds,
            node_returns,
            node_endpoints if args.color_by == "endpoint" else None,
            node_success_rates if args.color_by == "endpoint" else None,
            node_endpoint_counts if args.color_by == "endpoint_pie" else None,
            node_xy,
            title,
            output_path,
            threshold_method=args.threshold_method,
            image_scale=args.image_scale,
            distance_label=distance_label,
            color_by=args.color_by,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

