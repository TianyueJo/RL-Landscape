import json
from pathlib import Path


def plot_distance_matrix_plotly(distance_matrix, title, output_path: Path) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(
        data=go.Heatmap(
            z=distance_matrix,
            colorscale="Viridis",
            colorbar=dict(title="L2 Distance"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Policy Index",
        yaxis_title="Policy Index",
        width=720,
        height=640,
    )
    fig.write_html(str(output_path))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot L2 distance matrices as heatmaps (HTML)")
    parser.add_argument(
        "--case-id",
        type=int,
        default=None,
        help="Only plot distance matrices for the specified case_id (omit to plot all).",
    )
    args = parser.parse_args()

    embeddings_dir = Path("behavior_space_embeddings")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Missing directory: {embeddings_dir}")

    dist_files = sorted(embeddings_dir.glob("*_l2_dist.json"))
    if not dist_files:
        raise FileNotFoundError(f"No distance matrix files in {embeddings_dir}")

    for dist_path in dist_files:
        with dist_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        distance_matrix = payload.get("distance_matrix", [])
        case_id = payload.get("case_id")
        method = payload.get("method")
        dim = payload.get("dim")

        if args.case_id is not None and int(case_id) != int(args.case_id):
            continue

        title = f"L2 Distance Matrix ({method.upper()}, dim={dim}, case={case_id})"
        output_path = dist_path.with_suffix("").with_suffix(".html")
        plot_distance_matrix_plotly(distance_matrix, title, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()







