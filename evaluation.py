#!/usr/bin/env python3
"""
Evaluation script: PCA + UMAP visualization of learned species embeddings.

Reads:
- Training checkpoint (.pt) (baseline or GNN)
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

    order_by_sp = dict(zip(species, order))
    family_by_sp = dict(zip(species, family))
    species_sorted = sorted(species)
    family_labels = [family_by_sp[sp] for sp in species_sorted]
    order_labels = [order_by_sp[sp] for sp in species_sorted]
    return species_sorted, family_labels, order_labels


def load_species_embeddings_from_ckpt(ckpt_path: Path, num_species_expected: int) -> np.ndarray:
    """
    Checkpoint-agnostic loader.

    Supports:
    - Baseline CLIP checkpoint: ckpt["model"] contains ... species_emb.weight
    - GNN checkpoint: ckpt has ckpt["species_embeddings"] (preferred)
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # 1) Preferred: explicit tensor
    if "species_embeddings" in ckpt:
        W = ckpt["species_embeddings"]
        if isinstance(W, np.ndarray):
            W = torch.from_numpy(W)
        W = W.detach().cpu().numpy()
        if W.shape[0] != num_species_expected:
            raise RuntimeError(
                f"Checkpoint has species_embeddings with {W.shape[0]} rows but labels file has {num_species_expected}. "
                f"Are you using the same filtered dataset?"
            )
        return W

    # 2) Backward-compatible: look in state_dict
    if "model" not in ckpt:
        raise RuntimeError("Checkpoint has no 'species_embeddings' and no 'model' state_dict.")

    sd = ckpt["model"]
    key = None
    for k in sd.keys():
        if k.endswith("species_emb.weight"):
            key = k
            break
    if key is None:
        raise RuntimeError(
            "Could not find species embeddings in checkpoint. "
            "Expected ckpt['species_embeddings'] or a state_dict key ending with 'species_emb.weight'."
        )

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


def top_orders(order_labels, max_orders: int):
    c = Counter(order_labels)
    return [o for o, _ in c.most_common(max_orders)]


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


def highlight_top_orders_plot(
    X2: np.ndarray,
    order_labels: list,
    outpath: Path,
    top_order_list: list,
    title: str,
    max_orders: int,
    xlabel: str = "dim1",
    ylabel: str = "dim2",
    background_color=(0.82, 0.82, 0.82, 0.35),
):
    order_arr = np.array(order_labels)
    top_set = set(top_order_list)

    labels = []
    for o in order_arr:
        labels.append(o if o in top_set else "Background")

    cmap = plt.get_cmap("tab10")
    order_to_color = {o: cmap(i % 10) for i, o in enumerate(top_order_list)}
    bg = background_color

    colors = []
    for lab in labels:
        if lab == "Background":
            colors.append(bg)
        else:
            colors.append(order_to_color.get(lab, (0.2, 0.2, 0.2, 0.9)))

    scatter_plot_2d(
        X2, labels,
        title=title,
        outpath=outpath,
        max_classes=max_orders + 2,
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

        fam_to_color = {}
        for j, fam in enumerate(top_fams):
            fam_to_color[fam] = cmap(j % 10)
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
# Cosine kNN + metrics
# ----------------------------

def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)


def build_knn_indices_cosine(Xn: np.ndarray, k: int) -> np.ndarray:
    """
    Returns neighbor indices excluding self.
    Uses cosine distance (equivalent to 1 - dot for normalized vectors).
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(Xn)
    _, idx = nn.kneighbors(Xn, return_distance=True)
    return idx[:, 1:]  # drop self


def knn_order_purity_from_indices(order_labels: list, neigh_idx: np.ndarray, top_order_list: list) -> dict:
    order_arr = np.array(order_labels)
    top_set = set(top_order_list)

    mapped = np.array([o if o in top_set else "Other" for o in order_arr], dtype=object)

    per_order_hits = Counter()
    per_order_total = Counter()

    for i in range(len(mapped)):
        o = mapped[i]
        nbrs = mapped[neigh_idx[i]]
        hits = int((nbrs == o).sum())
        per_order_hits[o] += hits
        per_order_total[o] += len(nbrs)

    per_order = {}
    for o in per_order_total:
        per_order[o] = per_order_hits[o] / max(1, per_order_total[o])

    overall = sum(per_order_hits.values()) / max(1, sum(per_order_total.values()))
    counts = Counter(mapped.tolist())

    return {"overall": overall, "per_order": per_order, "counts": counts}


def knn_restricted_intra_order_cosine_distance(Xn: np.ndarray, order_labels: list, neigh_idx: np.ndarray, top_order_list: list) -> dict:
    order_arr = np.array(order_labels)
    top_set = set(top_order_list)
    mapped = np.array([o if o in top_set else "Other" for o in order_arr], dtype=object)

    out = {}
    for o in list(top_order_list) + ["Other"]:
        mask = (mapped == o)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            out[o] = {"n_points": 0, "avg_knn_intra_cosine_distance": float("nan")}
            continue

        dists = []
        for i in idxs:
            nbrs = neigh_idx[i]
            nbrs_same = [j for j in nbrs if mapped[j] == o]
            if not nbrs_same:
                continue
            # cosine dist = 1 - dot, since Xn rows are normalized
            dots = (Xn[i] * Xn[nbrs_same]).sum(axis=1)
            dists.append(float(np.mean(1.0 - dots)))
        out[o] = {
            "n_points": int(len(idxs)),
            "avg_knn_intra_cosine_distance": float(np.mean(dists)) if dists else float("nan"),
        }
    return out


def order_pair_separation_effect(Xn: np.ndarray, order_labels: list, top_order_list: list) -> List[Tuple[str, str, float, float, float, float]]:
    """
    Pair separation based on centroid distance normalized by within-order dispersion:
      centroid_cosine_dist = 1 - dot(cA, cB)
      sigma2_A = mean( (1 - dot(x, cA))^2 ) over x in A
      separation_effect = centroid_cosine_dist / sqrt(sigma2_A + sigma2_B)
    """
    order_arr = np.array(order_labels)
    top_set = set(top_order_list)
    mapped = np.array([o if o in top_set else "Other" for o in order_arr], dtype=object)

    # compute centroids
    centroids = {}
    sig2 = {}
    for o in list(top_order_list) + ["Other"]:
        idxs = np.where(mapped == o)[0]
        if len(idxs) == 0:
            centroids[o] = None
            sig2[o] = float("nan")
            continue
        c = Xn[idxs].mean(axis=0)
        c = c / max(1e-12, np.linalg.norm(c))
        centroids[o] = c

        dots = (Xn[idxs] * c).sum(axis=1)
        d = 1.0 - dots
        sig2[o] = float(np.mean(d ** 2))

    orders = list(top_order_list) + ["Other"]
    pairs = []
    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            a, b = orders[i], orders[j]
            if centroids[a] is None or centroids[b] is None:
                continue
            cA, cB = centroids[a], centroids[b]
            centroid_cos_dist = float(1.0 - float(np.dot(cA, cB)))
            sA, sB = sig2[a], sig2[b]
            denom = math.sqrt(max(1e-12, sA + sB)) if (not np.isnan(sA) and not np.isnan(sB)) else float("nan")
            eff = float(centroid_cos_dist / denom) if denom == denom else float("nan")
            pairs.append((a, b, centroid_cos_dist, float(sA), float(sB), eff))
    return pairs


def knn_inter_order_margin(Xn: np.ndarray, order_labels: list, neigh_idx: np.ndarray, top_order_list: list):
    """
    Local overlap / mixing between order pairs based on kNN neighborhoods.
    For each pair (A,B):
      margin A->B = mean_over_i_in_A [ mean_cos_dist(i, N_B(i)) - mean_cos_dist(i, N_A(i)) ]
    where N_B(i) are neighbors of i that belong to B (if any), N_A(i) those in A.
    """
    order_arr = np.array(order_labels)
    top_set = set(top_order_list)
    mapped = np.array([o if o in top_set else "Other" for o in order_arr], dtype=object)

    orders = list(top_order_list) + ["Other"]
    out = []

    # precompute dot products for neighbor retrieval quickly
    # cosine dist = 1 - dot for normalized rows
    for ia in range(len(orders)):
        for ib in range(ia + 1, len(orders)):
            A = orders[ia]
            B = orders[ib]
            idxA = np.where(mapped == A)[0]
            idxB = np.where(mapped == B)[0]
            if len(idxA) == 0 or len(idxB) == 0:
                continue

            def margin_one_way(src_order, tgt_order):
                idxS = np.where(mapped == src_order)[0]
                margins = []
                used = 0
                for i in idxS:
                    nbrs = neigh_idx[i]
                    n_same = [j for j in nbrs if mapped[j] == src_order]
                    n_tgt = [j for j in nbrs if mapped[j] == tgt_order]
                    if not n_same or not n_tgt:
                        continue
                    used += 1
                    dot_same = (Xn[i] * Xn[n_same]).sum(axis=1)
                    dot_tgt = (Xn[i] * Xn[n_tgt]).sum(axis=1)
                    d_same = 1.0 - dot_same
                    d_tgt = 1.0 - dot_tgt
                    margins.append(float(np.mean(d_tgt) - float(np.mean(d_same))))
                return (float(np.mean(margins)) if margins else float("nan"), used)

            m_ab, n_ab = margin_one_way(A, B)
            m_ba, n_ba = margin_one_way(B, A)
            sym = float(np.nanmean([m_ab, m_ba]))
            out.append((A, B, m_ab, n_ab, m_ba, n_ba, sym))

    return out


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--species_taxonomy_tsv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pca_components", type=int, default=50)

    ap.add_argument("--max_orders", type=int, default=12)
    ap.add_argument("--max_families", type=int, default=15)

    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.2)

    ap.add_argument("--knn_k", type=int, default=10, help="k for kNN metrics (cosine distance)")
    ap.add_argument("--metrics_space", choices=["raw", "pca"], default="raw",
                    help="Compute cosine metrics in raw embedding space or PCA-reduced space (same dims as --pca_components)")

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

    neigh_idx = build_knn_indices_cosine(Xn, k=args.knn_k)

    # 1) kNN purity
    knn = knn_order_purity_from_indices(order_labels, neigh_idx, top_order_list)
    knn_path = args.out_dir / "knn_order_purity.tsv"
    with knn_path.open("w", encoding="utf-8") as f:
        f.write("order\tn_species\tknn_purity\n")
        for o in top_order_list:
            f.write(f"{o}\t{knn['counts'][o]}\t{knn['per_order'].get(o, float('nan')):.6f}\n")
        f.write(f"Other\t{knn['counts'].get('Other', 0)}\t{knn['per_order'].get('Other', float('nan')):.6f}\n")
        f.write(f"OVERALL\t{len(order_labels)}\t{knn['overall']:.6f}\n")

    # 2) kNN intra-order distance
    intra_knn = knn_restricted_intra_order_cosine_distance(Xn, order_labels, neigh_idx, top_order_list)
    intra_path = args.out_dir / "order_knn_intra_cosine_distance.tsv"
    with intra_path.open("w", encoding="utf-8") as f:
        f.write("order\tn_points\tavg_knn_intra_cosine_distance\n")
        for o in top_order_list:
            f.write(f"{o}\t{intra_knn[o]['n_points']}\t{intra_knn[o]['avg_knn_intra_cosine_distance']:.6f}\n")
        f.write(f"Other\t{intra_knn['Other']['n_points']}\t{intra_knn['Other']['avg_knn_intra_cosine_distance']:.6f}\n")

    # 3) centroid+dispersion separation
    pairs = order_pair_separation_effect(Xn, order_labels, top_order_list)
    pair_path = args.out_dir / "order_pair_separation.tsv"
    with pair_path.open("w", encoding="utf-8") as f:
        f.write("order_a\torder_b\tcentroid_cosine_dist\tsigma2_a\tsigma2_b\tseparation_effect\n")
        for a, b, dc, s2a, s2b, eff in pairs:
            f.write(f"{a}\t{b}\t{dc:.6f}\t{s2a:.6f}\t{s2b:.6f}\t{eff:.6f}\n")

    # 4) kNN inter-order margin
    knn_pairs = knn_inter_order_margin(Xn, order_labels, neigh_idx, top_order_list)
    knn_pair_path = args.out_dir / "order_pair_knn_margin.tsv"
    with knn_pair_path.open("w", encoding="utf-8") as f:
        f.write("order_a\torder_b\tmargin_a_to_b\tcount_a_used\tmargin_b_to_a\tcount_b_used\tsymmetric_margin\n")
        for a, b, m_ab, n_ab, m_ba, n_ba, sym in knn_pairs:
            def fmt(x):
                return "NA" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.6f}"
            f.write(f"{a}\t{b}\t{fmt(m_ab)}\t{n_ab}\t{fmt(m_ba)}\t{n_ba}\t{fmt(sym)}\n")

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
        "1) kNN order purity (cosine):",
        f"  overall={knn['overall']:.4f}",
    ]

    items = sorted([(o, knn["counts"][o], knn["per_order"][o]) for o in top_order_list],
                   key=lambda t: t[1], reverse=True)
    for o, cnt, pur in items:
        lines.append(f"  {o}: n={cnt}, purity={_fmt(pur)}")
    lines.append(f"  Other: n={knn['counts'].get('Other', 0)}, purity={_fmt(knn['per_order'].get('Other', np.nan))}")

    lines += ["", "2) kNN-restricted intra-order cosine distance (lower = tighter):"]
    items2 = sorted([(o, intra_knn[o]["n_points"], intra_knn[o]["avg_knn_intra_cosine_distance"]) for o in top_order_list],
                    key=lambda t: t[1], reverse=True)
    for o, cnt, d in items2:
        lines.append(f"  {o}: n={cnt}, avg_knn_intra_dist={_fmt(d)}")
    lines.append(
        f"  Other: n={intra_knn['Other']['n_points']}, avg_knn_intra_dist={_fmt(intra_knn['Other']['avg_knn_intra_cosine_distance'])}"
    )

    lines += ["", "3) Inter-order separation effect (centroid dist / dispersion):",
              "   (larger = more separated after accounting for within-order spread)"]
    # show top 10 by effect size
    pairs_sorted = sorted(pairs, key=lambda x: x[-1], reverse=True)
    for a, b, dc, s2a, s2b, eff in pairs_sorted[:10]:
        lines.append(f"  {a} vs {b}: effect={_fmt(eff)}, centroid_dist={_fmt(dc)}, sigma2=({_fmt(s2a)}, {_fmt(s2b)})")

    lines += ["", "4) kNN inter-order margin (local mixing):",
              "   (larger = farther from the other order than from own order, locally)"]
    knn_pairs_sorted = sorted(knn_pairs, key=lambda x: (x[-1] if x[-1] == x[-1] else -1e9), reverse=True)
    for a, b, m_ab, n_ab, m_ba, n_ba, sym in knn_pairs_sorted[:10]:
        lines.append(f"  {a} vs {b}: sym_margin={_fmt(sym)} (A->B={_fmt(m_ab)}, B->A={_fmt(m_ba)})")

    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Wrote plots + metrics to:", args.out_dir)


if __name__ == "__main__":
    main()
