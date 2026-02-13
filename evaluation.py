#!/usr/bin/env python3
"""
Evaluation script: PCA + UMAP + t-SNE visualization of learned species anchor embeddings.

Reads:
- Training checkpoint (.pt) from baseline script
- species_taxonomy.tsv for family/order labels

Writes:
- pca_order.png
- pca_families_in_<ORDER>.png for each top order (shows ALL points; non-target orders in gray)
- umap_order.png
- umap_families_in_<ORDER>.png for each top order (shows ALL points; non-target orders in gray)
- tsne_order.png
- tsne_families_in_<ORDER>.png for each top order (shows ALL points; non-target orders in gray)
- summary.txt

Key features:
- Separate caps:
    --max_orders (for order plot + choosing which per-order plots to make)
    --max_families (top families within each selected order)
- Per-order family plots show all points, but highlight families only within that order.
- Legend excludes background points.
- Uses PCA->(UMAP/t-SNE) for speed and stability.

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
from sklearn.manifold import TSNE
import umap


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
            fam = parts[idx["family"]]
            ord_ = parts[idx["order"]]
            species.append(sp)
            family.append(fam if fam else "Unknown")
            order.append(ord_ if ord_ else "Unknown")

    # Ensure stable sorting (matches training script's sorted species IDs)
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
    - colors: optional list/array of matplotlib colors (len N). If set, points are plotted in one call.
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
            plt.scatter(
                X2[idx, 0], X2[idx, 1],
                s=8, alpha=0.7,
                label=f"{u} (n={len(idx)})"
            )
    else:
        plt.scatter(X2[:, 0], X2[:, 1], s=8, alpha=0.7, c=colors, linewidths=0)
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
      - legend excludes "Background"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--species_taxonomy_tsv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)

    # PCA pre-reduction dims used for UMAP and t-SNE
    ap.add_argument("--pca_components", type=int, default=80)

    # Separate caps
    ap.add_argument("--max_orders", type=int, default=8,
                    help="Max orders to show (and to generate per-order plots for)")
    ap.add_argument("--max_families", type=int, default=10,
                    help="Max families to show within each selected order")

    # UMAP params
    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.2)

    # t-SNE params
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_learning_rate", type=float, default=200.0)
    ap.add_argument("--tsne_iterations", type=int, default=1500)
    ap.add_argument("--tsne_init", type=str, default="pca", choices=["pca", "random"])
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    species_sorted, family_labels, order_labels = read_species_taxonomy_tsv(args.species_taxonomy_tsv)
    W = load_species_embeddings_from_ckpt(args.ckpt, num_species_expected=len(species_sorted))

    top_order_list = top_orders(order_labels, args.max_orders)

    # --------------------
    # PCA 2D (direct)
    # --------------------
    pca2 = PCA(n_components=2, random_state=args.seed)
    X_pca2 = pca2.fit_transform(W)
    var = pca2.explained_variance_ratio_
    xlabel = f"PC1 ({var[0]*100:.1f}% var)"
    ylabel = f"PC2 ({var[1]*100:.1f}% var)"

    scatter_plot_2d(
        X_pca2, order_labels,
        title="PCA (2D) of species anchor embeddings — colored by Order",
        outpath=args.out_dir / "pca_order.png",
        max_classes=args.max_orders,
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
    # PCA pre-reduction for UMAP / t-SNE
    # --------------------
    pca = PCA(n_components=min(args.pca_components, W.shape[1]), random_state=args.seed)
    Wp = pca.fit_transform(W)

    # --------------------
    # UMAP
    # --------------------
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric="cosine",
        random_state=args.seed
    )
    X_umap = reducer.fit_transform(Wp)

    scatter_plot_2d(
        X_umap, order_labels,
        title="UMAP of species anchor embeddings — colored by Order",
        outpath=args.out_dir / "umap_order.png",
        max_classes=args.max_orders,
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
    # t-SNE
    # --------------------
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        learning_rate=args.tsne_learning_rate,
        n_iter=args.tsne_iterations,
        init=args.tsne_init,
        random_state=args.seed,
        metric="euclidean",  # we already PCA-reduced; euclidean is standard for t-SNE here
        verbose=1,
    )
    X_tsne = tsne.fit_transform(Wp)

    scatter_plot_2d(
        X_tsne, order_labels,
        title="t-SNE of species anchor embeddings — colored by Order",
        outpath=args.out_dir / "tsne_order.png",
        max_classes=args.max_orders,
    )
    per_order_family_plots(
        X_tsne,
        order_labels=order_labels,
        family_labels=family_labels,
        out_dir=args.out_dir,
        prefix="tsne",
        top_order_list=top_order_list,
        max_families=args.max_families,
    )

    # Summary
    (args.out_dir / "summary.txt").write_text(
        "\n".join([
            f"Species plotted: {len(species_sorted)}",
            f"Embedding dim: {W.shape[1]}",
            f"max_orders: {args.max_orders}",
            f"max_families (per top order): {args.max_families}",
            f"Top orders: {', '.join(top_order_list)}",
            f"PCA pre-reduction dims: {min(args.pca_components, W.shape[1])}",
            f"UMAP n_neighbors: {args.umap_neighbors}",
            f"UMAP min_dist: {args.umap_min_dist}",
            f"t-SNE perplexity: {args.tsne_perplexity}",
            f"t-SNE learning_rate: {args.tsne_learning_rate}",
            f"t-SNE iterations: {args.tsne_iterations}",
            f"t-SNE init: {args.tsne_init}",
            f"seed: {args.seed}",
        ]) + "\n",
        encoding="utf-8"
    )

    print("Wrote plots to:", args.out_dir)


if __name__ == "__main__":
    main()
