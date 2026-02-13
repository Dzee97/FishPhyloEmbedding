#!/usr/bin/env python3
"""
Evaluation script: PCA + UMAP visualization of learned species anchor embeddings
with hierarchical taxonomy plots: Class -> Orders-within-class -> Families-within-(class, order).

Reads:
- Training checkpoint (.pt)
- species_taxonomy.tsv with columns:
    species, class, order, family  (plus any extra columns)

Writes:
- pca_class.png
- pca_orders_in_<CLASS>.png
- pca_families_in_<CLASS>__<ORDER>.png
- umap_class.png
- umap_orders_in_<CLASS>.png
- umap_families_in_<CLASS>__<ORDER>.png
- summary.txt

Dependencies:
  pip install torch numpy scikit-learn matplotlib umap-learn
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap


def read_species_taxonomy_tsv(path: Path):
    """
    Returns:
      species_sorted (list)
      class_labels (list)
      order_labels (list)
      family_labels (list)
    """
    species = []
    cls = []
    order = []
    family = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        required = ["species", "class", "order", "family"]
        for r in required:
            if r not in idx:
                raise RuntimeError(f"{path} missing required column: {r}")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            sp = parts[idx["species"]]
            c = parts[idx["class"]] or "Unknown"
            o = parts[idx["order"]] or "Unknown"
            fam = parts[idx["family"]] or "Unknown"
            species.append(sp)
            cls.append(c)
            order.append(o)
            family.append(fam)

    cls_by_sp = dict(zip(species, cls))
    order_by_sp = dict(zip(species, order))
    fam_by_sp = dict(zip(species, family))

    species_sorted = sorted(species)
    class_labels = [cls_by_sp[sp] for sp in species_sorted]
    order_labels = [order_by_sp[sp] for sp in species_sorted]
    family_labels = [fam_by_sp[sp] for sp in species_sorted]
    return species_sorted, class_labels, order_labels, family_labels


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


def top_k(labels, k):
    return [x for x, _ in Counter(labels).most_common(k)]


def highlight_categories_plot(
    X2: np.ndarray,
    labels: list,
    outpath: Path,
    top_list: list,
    title: str,
    max_classes: int,
    xlabel="dim1",
    ylabel="dim2",
    background_color=(0.82, 0.82, 0.82, 0.35),
):
    lab_arr = np.array(labels)
    top_set = set(top_list)

    plot_labels = ["Background" if l not in top_set else l for l in lab_arr]

    cmap = plt.get_cmap("tab10")
    cat_to_color = {cat: cmap(i % 10) for i, cat in enumerate(top_list)}
    bg = background_color

    colors = [bg if l == "Background" else cat_to_color.get(l, (0.2, 0.2, 0.2, 0.9)) for l in plot_labels]

    scatter_plot_2d(
        X2, plot_labels,
        title=title,
        outpath=outpath,
        max_classes=max_classes + 1,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors,
        legend_exclude={"Background"},
    )


def plot_orders_within_class(
    X2: np.ndarray,
    class_labels: list,
    order_labels: list,
    out_dir: Path,
    prefix: str,
    top_classes: list,
    max_orders_per_class: int,
    xlabel="dim1",
    ylabel="dim2",
):
    cls_arr = np.array(class_labels)
    ord_arr = np.array(order_labels)

    for c in top_classes:
        in_class = (cls_arr == c)
        orders_in_c = ord_arr[in_class].tolist()
        top_orders = top_k(orders_in_c, max_orders_per_class)

        outpath = out_dir / f"{prefix}_orders_in_{slugify(c)}.png"
        # Show all points: highlight top orders inside this class; everything else background
        labels = []
        for i in range(len(cls_arr)):
            if cls_arr[i] == c and ord_arr[i] in set(top_orders):
                labels.append(ord_arr[i])
            else:
                labels.append("Background")

        highlight_categories_plot(
            X2, labels,
            outpath=outpath,
            top_list=top_orders,
            title=f"{prefix.upper()} — Orders highlighted within class: {c}",
            max_classes=max_orders_per_class,
            xlabel=xlabel, ylabel=ylabel,
        )


def plot_families_within_class_order(
    X2: np.ndarray,
    class_labels: list,
    order_labels: list,
    family_labels: list,
    out_dir: Path,
    prefix: str,
    top_classes: list,
    max_orders_per_class: int,
    max_families_per_order: int,
    xlabel="dim1",
    ylabel="dim2",
):
    cls_arr = np.array(class_labels)
    ord_arr = np.array(order_labels)
    fam_arr = np.array(family_labels)

    for c in top_classes:
        in_class = (cls_arr == c)
        orders_in_c = ord_arr[in_class].tolist()
        top_orders = top_k(orders_in_c, max_orders_per_class)

        for o in top_orders:
            in_group = (cls_arr == c) & (ord_arr == o)
            fams_in_group = fam_arr[in_group].tolist()
            top_fams = top_k(fams_in_group, max_families_per_order)
            top_fams_set = set(top_fams)

            labels = []
            for i in range(len(cls_arr)):
                if cls_arr[i] == c and ord_arr[i] == o:
                    f = fam_arr[i]
                    labels.append(f if f in top_fams_set else "Other")
                else:
                    labels.append("Background")

            outpath = out_dir / f"{prefix}_families_in_{slugify(c)}__{slugify(o)}.png"

            # Color top families + Other; background excluded from legend
            cmap = plt.get_cmap("tab10")
            fam_to_color = {fam: cmap(j % 10) for j, fam in enumerate(top_fams)}
            fam_to_color["Other"] = (0.35, 0.35, 0.35, 0.9)
            bg = (0.82, 0.82, 0.82, 0.35)

            colors = []
            for lab in labels:
                if lab == "Background":
                    colors.append(bg)
                else:
                    colors.append(fam_to_color.get(lab, fam_to_color["Other"]))

            scatter_plot_2d(
                X2, labels,
                title=f"{prefix.upper()} — Families within (class={c}, order={o})",
                outpath=outpath,
                max_classes=max_families_per_order + 2,
                xlabel=xlabel, ylabel=ylabel,
                colors=colors,
                legend_exclude={"Background"},
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--species_taxonomy_tsv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)

    ap.add_argument("--pca_components", type=int, default=80)

    ap.add_argument("--max_classes", type=int, default=6)
    ap.add_argument("--max_orders_per_class", type=int, default=8)
    ap.add_argument("--max_families_per_order", type=int, default=10)

    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    species_sorted, class_labels, order_labels, family_labels = read_species_taxonomy_tsv(args.species_taxonomy_tsv)
    W = load_species_embeddings_from_ckpt(args.ckpt, num_species_expected=len(species_sorted))

    top_classes = top_k(class_labels, args.max_classes)

    # --------------------
    # PCA 2D
    # --------------------
    pca2 = PCA(n_components=2, random_state=args.seed)
    X_pca2 = pca2.fit_transform(W)
    var = pca2.explained_variance_ratio_
    xlabel = f"PC1 ({var[0]*100:.1f}% var)"
    ylabel = f"PC2 ({var[1]*100:.1f}% var)"

    highlight_categories_plot(
        X_pca2,
        class_labels,
        outpath=args.out_dir / "pca_class.png",
        top_list=top_classes,
        title="PCA (2D) — Classes highlighted",
        max_classes=args.max_classes,
        xlabel=xlabel, ylabel=ylabel,
    )
    plot_orders_within_class(
        X_pca2, class_labels, order_labels,
        out_dir=args.out_dir, prefix="pca",
        top_classes=top_classes,
        max_orders_per_class=args.max_orders_per_class,
        xlabel=xlabel, ylabel=ylabel,
    )
    plot_families_within_class_order(
        X_pca2, class_labels, order_labels, family_labels,
        out_dir=args.out_dir, prefix="pca",
        top_classes=top_classes,
        max_orders_per_class=args.max_orders_per_class,
        max_families_per_order=args.max_families_per_order,
        xlabel=xlabel, ylabel=ylabel,
    )

    # --------------------
    # PCA pre-reduction for UMAP
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

    highlight_categories_plot(
        X_umap,
        class_labels,
        outpath=args.out_dir / "umap_class.png",
        top_list=top_classes,
        title="UMAP — Classes highlighted",
        max_classes=args.max_classes,
    )
    plot_orders_within_class(
        X_umap, class_labels, order_labels,
        out_dir=args.out_dir, prefix="umap",
        top_classes=top_classes,
        max_orders_per_class=args.max_orders_per_class,
    )
    plot_families_within_class_order(
        X_umap, class_labels, order_labels, family_labels,
        out_dir=args.out_dir, prefix="umap",
        top_classes=top_classes,
        max_orders_per_class=args.max_orders_per_class,
        max_families_per_order=args.max_families_per_order,
    )

    (args.out_dir / "summary.txt").write_text(
        "\n".join([
            f"Species plotted: {len(species_sorted)}",
            f"Embedding dim: {W.shape[1]}",
            f"max_classes: {args.max_classes}",
            f"max_orders_per_class: {args.max_orders_per_class}",
            f"max_families_per_order: {args.max_families_per_order}",
            f"Top classes: {', '.join(top_classes)}",
            f"PCA pre-reduction dims: {min(args.pca_components, W.shape[1])}",
            f"UMAP n_neighbors: {args.umap_neighbors}",
            f"UMAP min_dist: {args.umap_min_dist}",
            f"seed: {args.seed}",
        ]) + "\n",
        encoding="utf-8"
    )

    print("Wrote plots to:", args.out_dir)


if __name__ == "__main__":
    main()
