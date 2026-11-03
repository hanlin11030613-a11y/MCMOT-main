"""
ReID embedding latent-space visualizer.

Loads per-frame, per-tracklet smooth_feat vectors from the JSONL log written
by GlobalRegistry and projects them to 2D with UMAP (preferred), t-SNE, or
PCA. Renders an interactive Plotly scatter plot — zoom, pan, hover, and legend
toggle all work out of the box. Saves a self-contained HTML file.

Usage
-----
    python visualize_reid_latent.py                             # defaults
    python visualize_reid_latent.py --input reid_norm_log.jsonl
    python visualize_reid_latent.py --method tsne
    python visualize_reid_latent.py --animate                   # frame slider
    python visualize_reid_latent.py --method pca --no-show      # save only

Dependencies
------------
    pip install plotly pandas scikit-learn
    pip install umap-learn          # optional but strongly recommended
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dimensionality reduction — UMAP > t-SNE > PCA, whichever is installed
# ---------------------------------------------------------------------------

def _reduce(vectors: np.ndarray, method: str) -> np.ndarray:
    """Run dimensionality reduction and return (N, 2) embedding."""
    n = len(vectors)

    if method == "umap":
        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(n_components=2, random_state=42,
                                     n_neighbors=min(15, n - 1),
                                     min_dist=0.1, metric="cosine")
            return reducer.fit_transform(vectors)
        except Exception as e:
            print(f"[warn] UMAP unavailable ({e}) — falling back to t-SNE.")
            method = "tsne"

    if method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        # PCA pre-reduction to 50 dims (required for t-SNE to be tractable on
        # large datasets; cosine distances are well-preserved at 50 dims for
        # unit-normalised ReID vectors).
        n_pca = min(50, vectors.shape[1], n - 1)
        print(f"  PCA pre-reduction: {vectors.shape[1]}-d → {n_pca}-d …")
        pca_vecs = PCA(n_components=n_pca, random_state=42).fit_transform(vectors)
        perplexity = min(30, max(5, n // 10))
        print(f"  t-SNE: perplexity={perplexity}, n={n} …")
        return TSNE(n_components=2, random_state=42, perplexity=perplexity,
                    init="pca", learning_rate="auto").fit_transform(pca_vecs)

    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(vectors)

    raise ValueError(f"Unknown method: {method}")



def load_jsonl(path: str) -> pd.DataFrame:
    """Parse reid_norm_log.jsonl into a flat DataFrame.

    Expected line format (written by GlobalRegistry._write_buffer):
        {"frame": <int>, "gid": [[<gid_int>, [f0, f1, ..., f255]], ...]}
    """
    rows = []
    path = Path(path)
    if not path.exists():
        sys.exit(f"[error] file not found: {path}")

    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[warn] skipping malformed line {lineno}: {e}")
                continue
            frame_id = int(d["frame"])
            for gid, vec in d["gid"]:
                rows.append({
                    "frame":     frame_id,
                    "global_id": int(gid),
                    "vector":    np.array(vec, dtype=np.float32),
                })

    if not rows:
        sys.exit("[error] no data found in the log file.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

def visualize(
    input_path: str = "reid_norm_log.jsonl",
    output_path: str = "reid_latent_space.html",
    method: str = "umap",
    animate: bool = False,
    show: bool = True,
    point_size: int = 7,
    opacity: float = 0.75,
):
    import plotly.express as px
    import plotly.graph_objects as go

    print(f"Loading {input_path} …")
    df = load_jsonl(input_path)
    n = len(df)
    n_gids = df["global_id"].nunique()
    n_frames = df["frame"].nunique()
    print(f"  {n} observations | {n_gids} global IDs | {n_frames} frames")

    # ---- Dimensionality reduction ----------------------------------------
    vectors = np.stack(df["vector"].values)
    print(f"Reducing {n} × {vectors.shape[1]}-d vectors with {method.upper()} …")
    embedding = _reduce(vectors, method)

    df["x"] = embedding[:, 0].astype(float)
    df["y"] = embedding[:, 1].astype(float)
    df["global_id_str"] = df["global_id"].astype(str)   # Plotly color key

    axis_labels = {
        "x": f"{method.upper()} dim 1",
        "y": f"{method.upper()} dim 2",
    }

    # ---- Build figure -------------------------------------------------------
    if animate:
        # Frame-by-frame slider: every frame's points animated in sequence.
        # Fit the reducer on ALL frames first (done above), then animate.
        df_sorted = df.sort_values("frame")
        fig = px.scatter(
            df_sorted,
            x="x", y="y",
            color="global_id_str",
            animation_frame="frame",
            hover_data={"frame": True, "global_id_str": True, "x": ":.3f", "y": ":.3f"},
            title=f"ReID Latent Space — {method.upper()} (animated by frame)",
            labels={**axis_labels, "global_id_str": "Global ID"},
            opacity=opacity,
            category_orders={"global_id_str": sorted(df["global_id_str"].unique(),
                                                      key=lambda x: int(x))},
        )
        fig.update_traces(marker=dict(size=point_size))
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 80
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
    else:
        fig = px.scatter(
            df,
            x="x", y="y",
            color="global_id_str",
            hover_data={"frame": True, "global_id_str": True, "x": ":.3f", "y": ":.3f"},
            title=f"ReID Embedding Latent Space — {method.upper()}",
            labels={**axis_labels, "global_id_str": "Global ID"},
            opacity=opacity,
            category_orders={"global_id_str": sorted(df["global_id_str"].unique(),
                                                      key=lambda x: int(x))},
        )
        fig.update_traces(marker=dict(size=point_size, line=dict(width=0.4, color="white")))

        # Draw per-gid convex hulls so cluster boundaries are obvious
        from scipy.spatial import ConvexHull
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
        gid_list = sorted(df["global_id"].unique())
        for i, gid in enumerate(gid_list):
            pts = df[df["global_id"] == gid][["x", "y"]].values
            if len(pts) < 3:
                continue
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])  # close the polygon
                color = palette[i % len(palette)]
                fig.add_trace(go.Scatter(
                    x=pts[hull_pts, 0], y=pts[hull_pts, 1],
                    mode="lines",
                    line=dict(color=color, width=1.2, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"hull_{gid}",
                ))
            except Exception:
                pass  # collinear points — skip hull

    # ---- Layout polish ------------------------------------------------------
    fig.update_layout(
        legend_title_text="Global ID",
        legend=dict(
            itemsizing="constant",
            font=dict(size=12),
            borderwidth=1,
        ),
        width=1300,
        height=820,
        plot_bgcolor="#f8f8f8",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=13),
        title_font_size=16,
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis=dict(gridcolor="#e0e0e0", zeroline=False),
        yaxis=dict(gridcolor="#e0e0e0", zeroline=False),
        margin=dict(l=60, r=40, t=70, b=60),
    )

    # ---- Export + show ------------------------------------------------------
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Saved → {output_path}")
    if show:
        fig.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize ReID embedding vectors in 2D latent space."
    )
    parser.add_argument(
        "--input", default="reid_norm_log.jsonl",
        help="Path to the JSONL log from GlobalRegistry (default: reid_norm_log.jsonl)",
    )
    parser.add_argument(
        "--output", default="reid_latent_space.html",
        help="Output HTML path (default: reid_latent_space.html)",
    )
    parser.add_argument(
        "--method", choices=["umap", "tsne", "pca"], default="umap",
        help="Dimensionality reduction method (default: umap)",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Add a frame-by-frame slider to animate embedding evolution",
    )
    parser.add_argument(
        "--no-show", dest="show", action="store_false",
        help="Save to file only; don't open the browser",
    )
    parser.add_argument(
        "--size", type=int, default=7,
        help="Scatter point size (default: 7)",
    )
    parser.add_argument(
        "--opacity", type=float, default=0.75,
        help="Point opacity 0–1 (default: 0.75)",
    )
    args = parser.parse_args()

    visualize(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        animate=args.animate,
        show=args.show,
        point_size=args.size,
        opacity=args.opacity,
    )
