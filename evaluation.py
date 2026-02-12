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
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


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


def scatter_plot_2d(X2, labels, title, outpath, max_classes=25, xlabel="dim1", ylabel="dim2"):
    counts = Counter(labels)
    top = set([k for k, _ in counts.most_common(max_classes)])
    labels2 = [lab if lab in top else "Other" for lab in labels]

    uniq = list(dict.fromkeys(labels2))
    plt.figure(figsize=(10, 8))
    for u in uniq:
        idx = np.where(np.array(labels2) == u)[0]
        plt.scatter(X2[idx, 0], X2[idx, 1], s=8, alpha=0.7, label=f"{u} (n={len(idx)})")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(markerscale=2, fontsize=8, loc="best")
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

    # PCA 2D
    pca2 = PCA(n_components=2, random_state=0)
    X_pca2 = pca2.fit_transform(W)
    var = pca2.explained_variance_ratio_

    xlabel = f"PC1 ({var[0]*100:.1f}% var)"
    ylabel = f"PC2 ({var[1]*100:.1f}% var)"

    scatter_plot_2d(
        X_pca2, family_labels,
        title="PCA (2D) of species anchor embeddings — colored by Family",
        outpath=args.out_dir / "pca_family.png",
        max_classes=args.max_classes,
        xlabel=xlabel, ylabel=ylabel
    )
    scatter_plot_2d(
        X_pca2, order_labels,
        title="PCA (2D) of species anchor embeddings — colored by Order",
        outpath=args.out_dir / "pca_order.png",
        max_classes=args.max_classes,
        xlabel=xlabel, ylabel=ylabel
    )

    # UMAP (optional)
    if HAS_UMAP:
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

        scatter_plot_2d(
            X_umap, family_labels,
            title="UMAP of species anchor embeddings — colored by Family",
            outpath=args.out_dir / "umap_family.png",
            max_classes=args.max_classes,
        )
        scatter_plot_2d(
            X_umap, order_labels,
            title="UMAP of species anchor embeddings — colored by Order",
            outpath=args.out_dir / "umap_order.png",
            max_classes=args.max_classes,
        )
    else:
        print("umap-learn not installed; skipping UMAP. Install with: pip install umap-learn")

    (args.out_dir / "summary.txt").write_text(
        f"Species plotted: {len(species_sorted)}\n"
        f"Embedding dim: {W.shape[1]}\n"
        f"UMAP available: {HAS_UMAP}\n",
        encoding="utf-8"
    )

    print("Wrote plots to:", args.out_dir)


if __name__ == "__main__":
    main()
