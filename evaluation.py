#!/usr/bin/env python3
"""
Evaluation script: PCA + UMAP visualization of learned species anchor embeddings.

Reads:
- Training checkpoint (.pt)
- species_taxonomy.tsv for order/family labels

Writes:
- pca_order.png  (top orders highlighted; other orders in light gray)
- pca_families_in_<ORDER>.png for each top order (ALL points; non-target orders in gray)
- umap_order.png
- umap_families_in_<ORDER>.png for each top order
- summary.txt

Quantitative outputs (cosine-based):
- knn_order_purity.tsv
- order_knn_intra_cosine_distance.tsv
- order_pair_separation.tsv   (centroid cosine dist + dispersion-normalized separation)

Dependencies:
  pip install torch numpy scikit-learn matplotlib umap-learn
"""

import argparse
from pathlib import Path
from collections import Counter
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap


# ----------------------------
# IO: taxonomy + checkpoint
# ----------------------------

def read_species_taxonomy_tsv(path: Path):
    """
    Returns:
      species_sorted (list)
      family_labels (list)
      order_labels (list)
    """
    species = []
    family = []
    order = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        required = ["species", "family", "order"]
        for r in required:
            if r not in idx:
                raise RuntimeError(f"{path} missing required column: {r}")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            sp = parts[idx["species"]]
            fam = parts[idx["family"]] if parts[idx["family"]] else "Unknown"
            ord_ = parts[idx["order"]] if parts[idx["order"]] else "Unknown"
            species.append(sp)
            family.append(fam)
            order.append(ord_)

    # Stable sorting (matches training script's sorted species list)
    order_by_sp = dict(zip(species, order))
    family_by_sp = dict(zip(species, family))
    species_sorted = sorted(species)
    family_labels = [family_by_sp[sp] for sp in species_sorted]
    order_labels = [order_by_sp[sp] for sp in species_sorted]
    return species_sorted, family_labels, order_labels


def load_species_embeddings_from_ckpt(ckpt_path: Path, num_species_expected: int):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt["model"]

    key = None
    for k in sd.keys():
        if k.endswith("species_emb.weight"):
            key = k
            break
    if key is None:
        raise RuntimeError("Could not find species_emb.weight in checkpoint state_dict keys.")

    W = sd[key].detach().cpu().numpy()
    if W.shape[0] != num_species_expected:
        raise RuntimeError(
            f"Checkpoint has {W.shape[0]} species embeddings but labels file has {num_species_expected}. "
            f"Are you using the same filtered dataset as training?"
        )
    return W


# ----------------------------
# Plot helpers
# ----------------------------

def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s[:80] if len(s) > 80 else s


def scatter_plot_2d(
    X2,
    labels,
    title,
    outpath,
    max_classes=25,
    xlabel="dim1",
    ylabel="dim2",
    colors=None,
    legend_exclude=None,
):
    """
    Scatter plot with optional explicit per-point colors.

    - labels: list[str], used to build legend after top-K collapsing
    - colors: optional list/array of matplotlib colors (len N)
    - legend_exclude: set of label names to omit from legend (e.g., {"Background"})
    """
    if legend_exclude is None:
        legend_exclude = set()

    counts = Counter(labels)
    top = set([k for k, _ in counts.most_common(max_classes)])
    labels2 = [lab if lab in top else "Other" for lab in labels]
    uniq = [u for u in dict.fromkeys(labels2) if u not in legend_exclude]

    plt.figure(figsize=(10, 8))
    labels2_arr = np.array(labels2)

    if colors is None:
        for u in uniq:
            idx = np.where(labels2_arr == u)[0]
            plt.scatter(X2[idx, 0], X2[idx, 1], s=8, alpha=0.7, label=f"{u} (n={len(idx)})")
    else:
        plt.scatter(X2[:, 0], X2[:, 1], s=8, alpha=0.7, c=colors)
        for u in uniq:
            idx = np.where(labels2_arr == u)[0]
            if len(idx) == 0:
                continue
            col = colors[idx[0]]
            plt.scatter([], [], s=30, color=col, alpha=0.9, label=f"{u} (n={len(idx)})")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(markerscale=1.5, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def top_orders(order_labels, max_orders: int):
    c = Counter(order_labels)
    return [o for o, _ in c.most_common(max_orders)]


def highlight_top_orders_plot(
    X2: np.ndarray,
    order_labels: list,
    outpath: Path,
    top_order_list: list,
    title: str,
    max_orders: int,
    xlabel="dim1",
    ylabel="dim2",
    background_color=(0.82, 0.82, 0.82, 0.35),
):
    """
    Show ALL points:
      - top orders get distinct colors
      - everything else is "Background" light gray (not in legend)
    """
    order_arr = np.array(order_labels)
    top_set = set(top_order_list)

    labels = [o if o in top_set else "Background" for o in order_arr]

    cmap = plt.get_cmap("tab10")
    order_to_color = {o: cmap(i % 10) for i, o in enumerate(top_order_list)}
    bg = background_color
    colors = [bg if lab == "Background" else order_to_color.get(lab, (0.2, 0.2, 0.2, 0.9)) for lab in labels]

    scatter_plot_2d(
        X2, labels,
        title=title,
        outpath=outpath,
        max_classes=max_orders + 1,
        xlabel=xlabel, ylabel=ylabel,
        colors=colors,
        legend_exclude={"Background"},
    )


def per_order_family_plots(
    X2: np.ndarray,
    order_labels: list,
    family_labels: list,
    out_dir: Path,
    prefix: str,
    top_order_list: list,
    max_families: int,
    xlabel: str = "dim1",
    ylabel: str = "dim2",
    background_color=(0.82, 0.82, 0.82, 0.35),
):
    """
    For each top order o:
      - plot ALL points (global context)
      - points in order o are colored by family (top families within that order)
      - points outside order o are background gray
      - legend excludes background points
    """
    order_arr = np.array(order_labels)
    family_arr = np.array(family_labels)

    cmap = plt.get_cmap("tab10")

    for o in top_order_list:
        in_order = (order_arr == o)

        fams_in = family_arr[in_order].tolist()
        fam_counts = Counter(fams_in)
        top_fams = [f for f, _ in fam_counts.most_common(max_families)]
        top_fams_set = set(top_fams)

        labels = []
        for i in range(len(order_arr)):
            if in_order[i]:
                f = family_arr[i]
                labels.append(f if f in top_fams_set else "Other")
            else:
                labels.append("Background")

        fam_to_color = {fam: cmap(j % 10) for j, fam in enumerate(top_fams)}
        fam_to_color["Other"] = (0.35, 0.35, 0.35, 0.9)
        bg = background_color

        colors = []
        for lab in labels:
            if lab == "Background":
                colors.append(bg)
            else:
                colors.append(fam_to_color.get(lab, fam_to_color["Other"]))

        outpath = out_dir / f"{prefix}_families_in_{slugify(o)}.png"
        scatter_plot_2d(
            X2, labels,
            title=f"{prefix.upper()} — Families highlighted within order: {o}",
            outpath=outpath,
            max_classes=max_families + 2,
            xlabel=xlabel,
            ylabel=ylabel,
            colors=colors,
            legend_exclude={"Background"},
        )


# ----------------------------
# Cosine-based metrics
# ----------------------------

def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n


def build_knn_indices_cosine(Xn: np.ndarray, k: int):
    """
    Returns neighbor indices of shape (N, k) using cosine distance on L2-normalized vectors.
    Excludes self neighbor automatically.
    """
    N = Xn.shape[0]
    if k >= N:
        raise ValueError(f"k must be < N (got k={k}, N={N})")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(Xn)
    neigh_idx = nn.kneighbors(return_distance=False)
    # Remove self (first col)
    neigh_idx = neigh_idx[:, 1:]
    return neigh_idx


def knn_order_purity_from_indices(order_labels: list, neigh_idx: np.ndarray, top_orders_list: list):
    """
    kNN purity (cosine distance), using precomputed neighbor indices.

    purity(i) = fraction of k neighbors that share i's order label.

    Reports:
      - overall purity (all points)
      - per-top-order purity (mean over points in that order)
      - "Other" purity (mean over non-top orders)
    """
    labels = np.array(order_labels)
    top_set = set(top_orders_list)
    neigh_labels = labels[neigh_idx]  # (N, k)
    same = (neigh_labels == labels[:, None])
    purity_per_point = same.mean(axis=1)

    overall = float(purity_per_point.mean())

    per_order = {}
    counts = {}
    for o in top_orders_list:
        mask = (labels == o)
        cnt = int(mask.sum())
        counts[o] = cnt
        per_order[o] = float(purity_per_point[mask].mean()) if cnt > 0 else float("nan")

    mask_other = ~np.isin(labels, list(top_set))
    cnt_other = int(mask_other.sum())
    counts["Other"] = cnt_other
    per_order["Other"] = float(purity_per_point[mask_other].mean()) if cnt_other > 0 else float("nan")

    return {
        "k": int(neigh_idx.shape[1]),
        "overall": overall,
        "per_order": per_order,
        "counts": counts,
    }


def knn_restricted_intra_order_cosine_distance(
    Xn: np.ndarray,
    order_labels: list,
    neigh_idx: np.ndarray,
    top_orders_list: list,
):
    """
    kNN-restricted intra-order cosine distance:

    For each point i:
      - consider its k nearest neighbors
      - keep only neighbors with SAME order label
      - compute mean cosine distance to those (1 - dot)
      - if no same-order neighbors, mark as NaN for that point

    Then for each top order:
      - average over points in that order that have >=1 same-order neighbor

    Returns:
      dict order -> {
        "n_points": int,
        "n_points_with_inorder_neighbors": int,
        "avg_knn_intra_cosine_distance": float (NaN if none)
      }
    """
    labels = np.array(order_labels)
    top_set = set(top_orders_list)

    # dot products with neighbors (since Xn rows are unit vectors)
    # For each i, neighbor dots = Xn[i] · Xn[j]
    # cosine distance = 1 - dot
    Xi = Xn[:, None, :]                        # (N, 1, D)
    Xj = Xn[neigh_idx]                         # (N, k, D)
    dots = np.sum(Xi * Xj, axis=2)             # (N, k)
    dists = 1.0 - dots                         # (N, k)

    neigh_labels = labels[neigh_idx]           # (N, k)
    same_mask = (neigh_labels == labels[:, None])

    # mean distance to same-order neighbors per point
    out_per_point = np.full((Xn.shape[0],), np.nan, dtype=np.float64)
    for i in range(Xn.shape[0]):
        m = same_mask[i]
        if np.any(m):
            out_per_point[i] = float(dists[i, m].mean())

    results = {}
    for o in top_orders_list:
        mask_o = (labels == o)
        n_points = int(mask_o.sum())
        vals = out_per_point[mask_o]
        valid = ~np.isnan(vals)
        n_valid = int(valid.sum())
        avg = float(vals[valid].mean()) if n_valid > 0 else float("nan")
        results[o] = {
            "n_points": n_points,
            "n_points_with_inorder_neighbors": n_valid,
            "avg_knn_intra_cosine_distance": avg,
        }

    # Other (optional but useful)
    mask_other = ~np.isin(labels, list(top_set))
    n_points = int(mask_other.sum())
    vals = out_per_point[mask_other]
    valid = ~np.isnan(vals)
    n_valid = int(valid.sum())
    avg = float(vals[valid].mean()) if n_valid > 0 else float("nan")
    results["Other"] = {
        "n_points": n_points,
        "n_points_with_inorder_neighbors": n_valid,
        "avg_knn_intra_cosine_distance": avg,
    }

    return results


def order_centroids_and_dispersion(Xn: np.ndarray, order_labels: list, orders: list, eps: float = 1e-12):
    """
    For each order:
      - centroid direction c (unit vector) from sum of unit vectors
      - dispersion sigma2 = mean( (cosine_distance(x, c))^2 ) = mean( (1 - x·c)^2 )

    Returns:
      centroids: dict order -> unit centroid (D,)
      sigma2:    dict order -> float
      counts:    dict order -> int
    """
    labels = np.array(order_labels)
    centroids = {}
    sigma2 = {}
    counts = {}

    for o in orders:
        idx = np.where(labels == o)[0]
        n = int(len(idx))
        counts[o] = n
        if n == 0:
            centroids[o] = None
            sigma2[o] = float("nan")
            continue

        S = Xn[idx].sum(axis=0)
        normS = float(np.linalg.norm(S))
        if normS < eps:
            # extremely unlikely unless vectors cancel perfectly
            c = S * 0.0
        else:
            c = S / normS

        centroids[o] = c

        dots = Xn[idx] @ c  # (n,)
        d = 1.0 - dots      # cosine distance to centroid
        sigma2[o] = float(np.mean(d * d))

    return centroids, sigma2, counts


def inter_order_separation_effect_size(
    Xn: np.ndarray,
    order_labels: list,
    top_orders_list: list,
):
    """
    For each pair (A,B) among top orders:
      - centroid cosine distance: d = 1 - cA·cB
      - dispersion-normalized separation:
          sep = d / sqrt(sigma2_A + sigma2_B)

    where sigma2 is mean squared cosine distance to centroid within each order.

    Returns list of tuples:
      (order_a, order_b, nA, nB, centroid_cosine_dist, sigma2_A, sigma2_B, sep_effect)
    """
    centroids, sigma2, counts = order_centroids_and_dispersion(Xn, order_labels, top_orders_list)

    rows = []
    for i, a in enumerate(top_orders_list):
        for b in top_orders_list[i + 1:]:
            nA, nB = counts[a], counts[b]
            cA, cB = centroids[a], centroids[b]
            if cA is None or cB is None or nA == 0 or nB == 0:
                rows.append((a, b, nA, nB, float("nan"), sigma2[a], sigma2[b], float("nan")))
                continue
            dist = 1.0 - float(np.dot(cA, cB))
            denom = float(np.sqrt(max(1e-12, sigma2[a] + sigma2[b])))
            sep = dist / denom
            rows.append((a, b, nA, nB, dist, sigma2[a], sigma2[b], sep))
    return rows


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--species_taxonomy_tsv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)

    ap.add_argument("--pca_components", type=int, default=80)

    ap.add_argument("--max_orders", type=int, default=12,
                    help="Top-N orders to highlight (also generates per-order family plots)")
    ap.add_argument("--max_families", type=int, default=15,
                    help="Top-N families to highlight within each selected order")

    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

    # Metrics args
    ap.add_argument("--knn_k", type=int, default=50, help="k for kNN metrics (cosine distance)")
    ap.add_argument("--metrics_space", choices=["raw", "pca"], default="raw",
                    help="Compute cosine metrics in raw embedding space or PCA-reduced space (same PCA dims as --pca_components)")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    species_sorted, family_labels, order_labels = read_species_taxonomy_tsv(args.species_taxonomy_tsv)
    W = load_species_embeddings_from_ckpt(args.ckpt, num_species_expected=len(species_sorted))

    top_order_list = top_orders(order_labels, args.max_orders)

    # --------------------
    # PCA 2D plots
    # --------------------
    pca2 = PCA(n_components=2, random_state=args.seed)
    X_pca2 = pca2.fit_transform(W)
    var = pca2.explained_variance_ratio_
    xlabel = f"PC1 ({var[0]*100:.1f}% var)"
    ylabel = f"PC2 ({var[1]*100:.1f}% var)"

    highlight_top_orders_plot(
        X_pca2,
        order_labels,
        outpath=args.out_dir / "pca_order.png",
        top_order_list=top_order_list,
        title="PCA (2D) — Top orders highlighted (others in gray)",
        max_orders=args.max_orders,
        xlabel=xlabel, ylabel=ylabel
    )

    per_order_family_plots(
        X_pca2,
        order_labels=order_labels,
        family_labels=family_labels,
        out_dir=args.out_dir,
        prefix="pca",
        top_order_list=top_order_list,
        max_families=args.max_families,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    # --------------------
    # UMAP plots (cosine)
    # --------------------
    pca = PCA(n_components=min(args.pca_components, W.shape[1]), random_state=args.seed)
    Wp = pca.fit_transform(W)

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric="cosine",
        random_state=args.seed
    )
    X_umap = reducer.fit_transform(Wp)

    highlight_top_orders_plot(
        X_umap,
        order_labels,
        outpath=args.out_dir / "umap_order.png",
        top_order_list=top_order_list,
        title="UMAP — Top orders highlighted (others in gray)",
        max_orders=args.max_orders,
    )

    per_order_family_plots(
        X_umap,
        order_labels=order_labels,
        family_labels=family_labels,
        out_dir=args.out_dir,
        prefix="umap",
        top_order_list=top_order_list,
        max_families=args.max_families,
    )

    # --------------------
    # Quantitative metrics (cosine)
    # --------------------
    if args.metrics_space == "raw":
        X_metrics = W
        metrics_space_desc = "raw"
    else:
        X_metrics = Wp
        metrics_space_desc = f"pca{Wp.shape[1]}"

    Xn = l2_normalize_rows(X_metrics)

    # Shared kNN indices for all kNN-based metrics
    neigh_idx = build_knn_indices_cosine(Xn, k=args.knn_k)

    # 1) kNN purity
    knn = knn_order_purity_from_indices(order_labels, neigh_idx, top_order_list)

    knn_path = args.out_dir / "knn_order_purity.tsv"
    with knn_path.open("w", encoding="utf-8") as f:
        f.write("order\tcount\tknn_purity_cosine\n")
        for o in top_order_list + ["Other"]:
            cnt = knn["counts"].get(o, 0)
            pur = knn["per_order"].get(o, float("nan"))
            if np.isnan(pur):
                f.write(f"{o}\t{cnt}\tNA\n")
            else:
                f.write(f"{o}\t{cnt}\t{pur:.6f}\n")
        f.write(f"OVERALL\t{len(order_labels)}\t{knn['overall']:.6f}\n")

    # 2) kNN-restricted intra-order cosine distance
    intra_knn = knn_restricted_intra_order_cosine_distance(
        Xn, order_labels, neigh_idx, top_order_list
    )

    intra_path = args.out_dir / "order_knn_intra_cosine_distance.tsv"
    with intra_path.open("w", encoding="utf-8") as f:
        f.write("order\tn_points\tn_points_with_inorder_neighbors\tavg_knn_intra_cosine_distance\n")
        for o in top_order_list + ["Other"]:
            rec = intra_knn[o]
            avg = rec["avg_knn_intra_cosine_distance"]
            if np.isnan(avg):
                f.write(f"{o}\t{rec['n_points']}\t{rec['n_points_with_inorder_neighbors']}\tNA\n")
            else:
                f.write(f"{o}\t{rec['n_points']}\t{rec['n_points_with_inorder_neighbors']}\t{avg:.6f}\n")

    # 3) Inter-order separation effect size (centroid dist normalized by dispersion)
    pairs = inter_order_separation_effect_size(Xn, order_labels, top_order_list)

    pair_path = args.out_dir / "order_pair_separation.tsv"
    with pair_path.open("w", encoding="utf-8") as f:
        f.write("order_a\torder_b\tcount_a\tcount_b\tcentroid_cosine_dist\tsigma2_a\tsigma2_b\tseparation_effect\n")
        for a, b, nA, nB, dist, s2a, s2b, sep in pairs:
            if np.isnan(dist) or np.isnan(sep):
                f.write(f"{a}\t{b}\t{nA}\t{nB}\tNA\t{s2a:.6g}\t{s2b:.6g}\tNA\n")
            else:
                f.write(f"{a}\t{b}\t{nA}\t{nB}\t{dist:.6f}\t{s2a:.6g}\t{s2b:.6g}\t{sep:.6f}\n")

    # --------------------
    # summary.txt
    # --------------------
    def _fmt(x):
        return "NA" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"

    lines = []
    lines += [
        f"Species plotted: {len(species_sorted)}",
        f"Embedding dim: {W.shape[1]}",
        f"max_orders: {args.max_orders}",
        f"max_families (per order): {args.max_families}",
        f"Top orders: {', '.join(top_order_list)}",
        f"PCA pre-reduction dims: {min(args.pca_components, W.shape[1])}",
        f"UMAP n_neighbors: {args.umap_neighbors}",
        f"UMAP min_dist: {args.umap_min_dist}",
        f"seed: {args.seed}",
        "",
        f"Metrics space: {metrics_space_desc}",
        f"kNN k: {args.knn_k}",
        "",
        "kNN order purity (cosine):",
        f"  overall={knn['overall']:.4f}",
    ]
    # list per-order purity sorted by count desc
    items = sorted([(o, knn["counts"][o], knn["per_order"][o]) for o in top_order_list],
                   key=lambda t: t[1], reverse=True)
    for o, cnt, pur in items:
        lines.append(f"  {o}: n={cnt}, purity={_fmt(pur)}")
    lines.append(f"  Other: n={knn['counts'].get('Other', 0)}, purity={_fmt(knn['per_order'].get('Other', np.nan))}")

    lines += ["", "kNN-restricted intra-order cosine distance (lower = tighter):"]
    items2 = sorted([(o, intra_knn[o]["n_points"], intra_knn[o]["avg_knn_intra_cosine_distance"]) for o in top_order_list],
                    key=lambda t: t[1], reverse=True)
    for o, cnt, d in items2:
        lines.append(f"  {o}: n={cnt}, avg_knn_intra_dist={_fmt(d)}")
    lines.append(
        f"  Other: n={intra_knn['Other']['n_points']}, avg_knn_intra_dist={_fmt(intra_knn['Other']['avg_knn_intra_cosine_distance'])}")

    lines += ["", "Inter-order separation effect size (higher = more separated vs spread):"]
    # show top 10 pairs by effect size
    pairs_sorted = sorted([p for p in pairs if not np.isnan(p[-1])], key=lambda t: t[-1], reverse=True)
    for a, b, nA, nB, dist, s2a, s2b, sep in pairs_sorted[:10]:
        lines.append(f"  {a} vs {b}: centroid_dist={dist:.4f}, sep_effect={sep:.4f}")

    lines += [
        "",
        "Files written:",
        f"  {knn_path.name}",
        f"  {intra_path.name}",
        f"  {pair_path.name}",
    ]

    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Wrote plots + metrics to:", args.out_dir)
    print(f"- {knn_path.name}")
    print(f"- {intra_path.name}")
    print(f"- {pair_path.name}")


if __name__ == "__main__":
    main()
