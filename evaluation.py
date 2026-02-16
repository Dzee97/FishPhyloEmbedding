#!/usr/bin/env python3
"""
Evaluation script: PCA + UMAP visualization + cosine metrics for learned species embeddings.

Works with BOTH:
  - Baseline CLIP checkpoint (species embedding table)
  - GNN phylogeny checkpoint (exports species_emb_weight)

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
1) knn_order_purity.tsv
2) order_knn_intra_cosine_distance.tsv
3) order_pair_separation.tsv
4) order_pair_knn_margin.tsv

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

    # Stable sorting
    order_by_sp = dict(zip(species, order))
    family_by_sp = dict(zip(species, family))
    species_sorted = sorted(species)
    family_labels = [family_by_sp[sp] for sp in species_sorted]
    order_labels = [order_by_sp[sp] for sp in species_sorted]
    return species_sorted, family_labels, order_labels


def _torch_load_compat(path: Path):
    """
    PyTorch 2.6+ defaults weights_only=True, which can fail if checkpoint contains
    non-allowlisted objects (e.g., pathlib.PosixPath inside args).
    We:
      1) try weights_only=True
      2) fallback to weights_only=False if needed (safe if you trust the checkpoint).
    Also compatible with older torch versions that don't support weights_only.
    """
    try:
        # Try safe load first (torch>=2.6)
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        # torch<2.6 doesn't know weights_only
        return torch.load(str(path), map_location="cpu")
    except Exception as e:
        # Likely weights_only failure; retry full unpickle
        print(
            "WARNING: torch.load(weights_only=True) failed.\n"
            "Retrying with weights_only=False (this unpickles arbitrary objects).\n"
            "Only do this for checkpoints you trust.\n"
            f"Original error: {repr(e)}"
        )
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            # torch<2.6 path
            return torch.load(str(path), map_location="cpu")


def load_species_embeddings_from_ckpt(ckpt_path: Path, species_sorted_expected: list):
    """
    Priority:
      1) ckpt["species_emb_weight"] (preferred; exported by both scripts)
      2) state_dict key that ends with "species_emb.weight" (legacy baseline)
    """
    ckpt = _torch_load_compat(ckpt_path)

    # Preferred unified export
    if isinstance(ckpt, dict) and "species_emb_weight" in ckpt:
        W = ckpt["species_emb_weight"]
        if isinstance(W, torch.Tensor):
            W = W.detach().cpu().numpy()
        else:
            W = np.asarray(W)

        # Optional sanity check if the ckpt stores ordering
        if "species_sorted" in ckpt:
            sp_ckpt = ckpt["species_sorted"]
            if list(sp_ckpt) != list(species_sorted_expected):
                raise RuntimeError(
                    "Checkpoint species_sorted does not match species_taxonomy.tsv ordering.\n"
                    "Make sure you evaluate on the same filtered dataset used for training."
                )

        if W.shape[0] != len(species_sorted_expected):
            raise RuntimeError(
                f"Checkpoint species_emb_weight has {W.shape[0]} rows but species_taxonomy.tsv has {len(species_sorted_expected)} species."
            )
        return W

    # Legacy fallback: baseline species_emb.weight inside state_dict
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError("Checkpoint missing 'model' state_dict and missing 'species_emb_weight' export.")
    sd = ckpt["model"]

    key = None
    for k in sd.keys():
        if k.endswith("species_emb.weight"):
            key = k
            break
    if key is None:
        raise RuntimeError("Could not find species_emb.weight in checkpoint, and no species_emb_weight export found.")

    W = sd[key].detach().cpu().numpy()
    if W.shape[0] != len(species_sorted_expected):
        raise RuntimeError(
            f"Checkpoint has {W.shape[0]} species embeddings but labels file has {len(species_sorted_expected)}."
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
    N = Xn.shape[0]
    if k >= N:
        raise ValueError(f"k must be < N (got k={k}, N={N})")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(Xn)
    neigh_idx = nn.kneighbors(return_distance=False)
    neigh_idx = neigh_idx[:, 1:]  # drop self
    return neigh_idx


def knn_order_purity_from_indices(order_labels: list, neigh_idx: np.ndarray, top_orders_list: list):
    labels = np.array(order_labels)
    top_set = set(top_orders_list)
    neigh_labels = labels[neigh_idx]
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
    labels = np.array(order_labels)
    top_set = set(top_orders_list)

    Xi = Xn[:, None, :]
    Xj = Xn[neigh_idx]
    dots = np.sum(Xi * Xj, axis=2)
    dists = 1.0 - dots

    neigh_labels = labels[neigh_idx]
    same_mask = (neigh_labels == labels[:, None])

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
        c = (S / normS) if normS >= eps else (S * 0.0)
        centroids[o] = c

        dots = Xn[idx] @ c
        d = 1.0 - dots
        sigma2[o] = float(np.mean(d * d))

    return centroids, sigma2, counts


def inter_order_separation_effect_size(Xn: np.ndarray, order_labels: list, top_orders_list: list):
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


def knn_inter_order_margin(
    Xn: np.ndarray,
    order_labels: list,
    neigh_idx: np.ndarray,
    top_orders_list: list,
):
    labels = np.array(order_labels)

    Xi = Xn[:, None, :]
    Xj = Xn[neigh_idx]
    dots = np.sum(Xi * Xj, axis=2)
    dists = 1.0 - dots

    neigh_labels = labels[neigh_idx]

    def directed_margin(a: str, b: str):
        idx_a = np.where(labels == a)[0]
        if idx_a.size == 0:
            return float("nan"), 0

        deltas = []
        for i in idx_a:
            lab_nei = neigh_labels[i]
            dist_nei = dists[i]
            m_A = (lab_nei == a)
            m_B = (lab_nei == b)
            if (not np.any(m_A)) or (not np.any(m_B)):
                continue
            d_aa = float(dist_nei[m_A].mean())
            d_ab = float(dist_nei[m_B].mean())
            deltas.append(d_ab - d_aa)

        if len(deltas) == 0:
            return float("nan"), 0
        return float(np.mean(deltas)), int(len(deltas))

    rows = []
    for i, a in enumerate(top_orders_list):
        for b in top_orders_list[i + 1:]:
            m_ab, n_ab = directed_margin(a, b)
            m_ba, n_ba = directed_margin(b, a)
            sym = float("nan") if (np.isnan(m_ab) or np.isnan(m_ba)) else 0.5 * (m_ab + m_ba)
            rows.append((a, b, m_ab, n_ab, m_ba, n_ba, sym))
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
    ap.add_argument("--max_orders", type=int, default=12)
    ap.add_argument("--max_families", type=int, default=15)

    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

    # Metrics
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--metrics_space", choices=["raw", "pca"], default="raw",
                    help="Compute cosine metrics in raw embedding space or PCA-reduced space (dims = --pca_components)")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    species_sorted, family_labels, order_labels = read_species_taxonomy_tsv(args.species_taxonomy_tsv)
    W = load_species_embeddings_from_ckpt(args.ckpt, species_sorted_expected=species_sorted)

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
    # UMAP plots
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
    # Metrics
    # --------------------
    if args.metrics_space == "raw":
        X_metrics = W
        metrics_space_desc = "raw"
    else:
        X_metrics = Wp
        metrics_space_desc = f"pca{Wp.shape[1]}"

    Xn = l2_normalize_rows(X_metrics)
    neigh_idx = build_knn_indices_cosine(Xn, k=args.knn_k)

    # 1) kNN purity
    knn = knn_order_purity_from_indices(order_labels, neigh_idx, top_order_list)
    with (args.out_dir / "knn_order_purity.tsv").open("w", encoding="utf-8") as f:
        f.write("order\tcount\tknn_purity_cosine\n")
        for o in top_order_list + ["Other"]:
            cnt = knn["counts"].get(o, 0)
            pur = knn["per_order"].get(o, float("nan"))
            f.write(f"{o}\t{cnt}\t{'NA' if np.isnan(pur) else f'{pur:.6f}'}\n")
        f.write(f"OVERALL\t{len(order_labels)}\t{knn['overall']:.6f}\n")

    # 2) kNN intra-order distance
    intra = knn_restricted_intra_order_cosine_distance(Xn, order_labels, neigh_idx, top_order_list)
    with (args.out_dir / "order_knn_intra_cosine_distance.tsv").open("w", encoding="utf-8") as f:
        f.write("order\tn_points\tn_points_with_inorder_neighbors\tavg_knn_intra_cosine_distance\n")
        for o in top_order_list + ["Other"]:
            rec = intra[o]
            avg = rec["avg_knn_intra_cosine_distance"]
            f.write(
                f"{o}\t{rec['n_points']}\t{rec['n_points_with_inorder_neighbors']}\t{'NA' if np.isnan(avg) else f'{avg:.6f}'}\n")

    # 3) centroid separation effect size
    pairs = inter_order_separation_effect_size(Xn, order_labels, top_order_list)
    with (args.out_dir / "order_pair_separation.tsv").open("w", encoding="utf-8") as f:
        f.write("order_a\torder_b\tcount_a\tcount_b\tcentroid_cosine_dist\tsigma2_a\tsigma2_b\tseparation_effect\n")
        for a, b, nA, nB, dist, s2a, s2b, sep in pairs:
            if np.isnan(dist) or np.isnan(sep):
                f.write(f"{a}\t{b}\t{nA}\t{nB}\tNA\t{s2a:.6g}\t{s2b:.6g}\tNA\n")
            else:
                f.write(f"{a}\t{b}\t{nA}\t{nB}\t{dist:.6f}\t{s2a:.6g}\t{s2b:.6g}\t{sep:.6f}\n")

    # 4) kNN inter-order margin
    margins = knn_inter_order_margin(Xn, order_labels, neigh_idx, top_order_list)
    with (args.out_dir / "order_pair_knn_margin.tsv").open("w", encoding="utf-8") as f:
        f.write("order_a\torder_b\tmargin_a_to_b\tcount_a_used\tmargin_b_to_a\tcount_b_used\tsymmetric_margin\n")
        for a, b, m_ab, n_ab, m_ba, n_ba, sym in margins:
            def fmt(x):
                return "NA" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.6f}"
            f.write(f"{a}\t{b}\t{fmt(m_ab)}\t{n_ab}\t{fmt(m_ba)}\t{n_ba}\t{fmt(sym)}\n")

    # Summary
    (args.out_dir / "summary.txt").write_text(
        "\n".join([
            f"Species plotted: {len(species_sorted)}",
            f"Embedding dim: {W.shape[1]}",
            f"max_orders: {args.max_orders}",
            f"max_families: {args.max_families}",
            f"Top orders: {', '.join(top_order_list)}",
            f"PCA pre-reduction dims for UMAP: {min(args.pca_components, W.shape[1])}",
            f"UMAP n_neighbors: {args.umap_neighbors}",
            f"UMAP min_dist: {args.umap_min_dist}",
            f"Metrics: cosine (space={metrics_space_desc}, knn_k={args.knn_k})",
        ]) + "\n",
        encoding="utf-8"
    )

    print("Wrote plots + metrics to:", args.out_dir)


if __name__ == "__main__":
    main()
