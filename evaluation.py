#!/usr/bin/env python3
"""
Evaluation script: PCA + UMAP visualization of learned species anchor embeddings.

Reads:
- Training checkpoint (.pt) from baseline script
- species_taxonomy.tsv (preferred) OR sequences.jsonl (fallback) for family/order labels

Writes:
- pca_family.png, pca_order.png
- umap_family.png, umap_order.png (if umap-learn installed)
- summary.txt

Dependencies:
  pip install torch numpy scikit-learn matplotlib
  pip install umap-learn   # optional
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict
import colorsys

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def _lighten_rgb(rgb, amount):
    """
    amount in [0,1]; 0 = original, 1 = white
    """
    r, g, b = rgb
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def make_hierarchical_labels(
    species_sorted,
    order_labels,
    family_labels,
    max_orders=12,
    max_families_per_order=12,
):
    """
    Returns:
      order_top: list[str]  (order, with rare -> OtherOrder)
      family_top: list[str] (family, with rare-within-order -> OtherFamily)
    """
    # Top orders globally
    order_counts = Counter(order_labels)
    top_orders = [o for o, _ in order_counts.most_common(max_orders)]
    top_orders_set = set(top_orders)

    order_top = [o if o in top_orders_set else "OtherOrder" for o in order_labels]

    # Top families within each kept order
    fam_counts_by_order = defaultdict(Counter)
    for o, f in zip(order_top, family_labels):
        fam_counts_by_order[o][f if f else "Unknown"] += 1

    top_fams_by_order = {}
    for o, c in fam_counts_by_order.items():
        # For OtherOrder we typically won't subdivide too much
        if o == "OtherOrder":
            top_fams_by_order[o] = set()
            continue
        top_fams = [fam for fam, _ in c.most_common(max_families_per_order)]
        top_fams_by_order[o] = set(top_fams)

    family_top = []
    for o, f in zip(order_top, family_labels):
        f = f if f else "Unknown"
        if o == "OtherOrder":
            family_top.append("OtherFamily")
        else:
            family_top.append(f if f in top_fams_by_order[o] else "OtherFamily")

    return order_top, family_top


def assign_order_family_colors(
    order_top,
    family_top,
    max_orders=12,
):
    """
    Returns:
      colors: list[(r,g,b)] for each point
      order_base_color: dict order -> rgb
      family_color: dict (order, family) -> rgb
    """
    # Base colors for top orders (distinct hues)
    base_cmap = plt.get_cmap("tab20")
    # tab20 has 20 distinct-ish colors; we’ll take first max_orders
    unique_orders = [o for o, _ in Counter(order_top).most_common()]

    # Ensure "OtherOrder" is last (gray)
    unique_orders = [o for o in unique_orders if o != "OtherOrder"] + \
        (["OtherOrder"] if "OtherOrder" in unique_orders else [])

    order_base_color = {}
    color_idx = 0
    for o in unique_orders:
        if o == "OtherOrder":
            order_base_color[o] = (0.6, 0.6, 0.6)
        else:
            order_base_color[o] = base_cmap(color_idx % 20)[:3]
            color_idx += 1

    # Shades within order for families: vary lightness towards white
    family_color = {}
    for o in unique_orders:
        base = order_base_color[o]
        # collect families within this order (excluding OtherFamily last)
        fams = [f for f, _ in Counter([f for oo, f in zip(order_top, family_top) if oo == o]).most_common()]
        if "OtherFamily" in fams:
            fams = [f for f in fams if f != "OtherFamily"] + ["OtherFamily"]

        # Choose shade amounts: evenly spaced lightening
        n = max(1, len(fams))
        # Darkest = near base, lightest = quite light
        # Keep range moderate so families are distinguishable but still "same hue"
        min_amt, max_amt = (0.10, 0.75) if o != "OtherOrder" else (0.0, 0.0)

        for i, f in enumerate(fams):
            if o == "OtherOrder":
                family_color[(o, f)] = base
            else:
                amt = min_amt + (max_amt - min_amt) * (i / max(1, n - 1))
                # OtherFamily: make it lightest so it reads as “misc”
                if f == "OtherFamily":
                    amt = 0.88
                family_color[(o, f)] = _lighten_rgb(base, amt)

    colors = [family_color[(o, f)] for o, f in zip(order_top, family_top)]
    return colors, order_base_color, family_color


def plot_umap_order_family_shaded(
    X2,
    species_sorted,
    order_labels,
    family_labels,
    outpath,
    title="UMAP — Orders as colors, Families as shades",
    max_orders=12,
    max_families_per_order=12,
    point_size=8,
    alpha=0.8,
    legend_orders=True,
    legend_families_per_order=4,
):
    """
    Single plot:
      - top orders get distinct colors
      - families within an order get shades of that order's color

    Legend strategy:
      - Always show order legend (recommended)
      - Optionally show up to N families per order (can get huge otherwise)
    """
    order_top, family_top = make_hierarchical_labels(
        species_sorted, order_labels, family_labels,
        max_orders=max_orders, max_families_per_order=max_families_per_order
    )
    colors, order_base_color, family_color = assign_order_family_colors(order_top, family_top, max_orders=max_orders)

    plt.figure(figsize=(11, 9))
    plt.scatter(X2[:, 0], X2[:, 1], s=point_size, alpha=alpha, c=colors, linewidths=0)
    plt.title(title)
    plt.xlabel("dim1")
    plt.ylabel("dim2")

    handles = []
    labels = []

    # Order legend (base colors)
    if legend_orders:
        for o, _ in Counter(order_top).most_common():
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=order_base_color[o], markersize=8))
            labels.append(f"Order: {o} (n={Counter(order_top)[o]})")

    # Optional: a small family legend subset per order (top N families)
    if legend_families_per_order and legend_families_per_order > 0:
        fam_counts_by_order = defaultdict(Counter)
        for o, f in zip(order_top, family_top):
            fam_counts_by_order[o][f] += 1

        for o, _ in Counter(order_top).most_common():
            if o == "OtherOrder":
                continue
            fams = [f for f, _ in fam_counts_by_order[o].most_common(legend_families_per_order)]
            for f in fams:
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=family_color[(o, f)], markersize=7))
                labels.append(f"  {o} → {f} (n={fam_counts_by_order[o][f]})")

    if handles:
        plt.legend(handles, labels, fontsize=8, loc="best", frameon=True, markerscale=1)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--species_taxonomy_tsv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--pca_components", type=int, default=50)
    ap.add_argument("--max_classes", type=int, default=25)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    species_sorted, family_labels, order_labels = read_species_taxonomy_tsv(args.species_taxonomy_tsv)
    W = load_species_embeddings_from_ckpt(args.ckpt, num_species_expected=len(species_sorted))

    pca = PCA(n_components=min(args.pca_components, W.shape[1]), random_state=0)
    Wp = pca.fit_transform(W)
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.2,
        n_components=2,
        metric="cosine",
        random_state=0
    )
    X_umap = reducer.fit_transform(Wp)

    plot_umap_order_family_shaded(
        X_umap,
        species_sorted=species_sorted,
        order_labels=order_labels,
        family_labels=family_labels,
        outpath=args.out_dir / "umap_order_family_shaded.png",
        title="UMAP of species anchors — Order hue, Family shade",
        max_orders=3,
        max_families_per_order=3,
        legend_orders=True,
        legend_families_per_order=3,   # keep legend manageable
    )

    print("Wrote plots to:", args.out_dir)


if __name__ == "__main__":
    main()
