#!/usr/bin/env python3
"""
Stage 1 (phylogeny-only): Learn species embeddings from the pruned phylogeny graph using a GNN.

Inputs (from preprocessing out_dir):
  - edge_index.npz     (directed edges, shape [2, E])
  - edge_attr.npz      (optional; branch lengths, shape [E, 1])  [not used by default]
  - nodes.tsv          (node_id, is_tip, name) where name is normalized binomial for tips (may be empty for internals)
  - leaf_map.tsv       (species -> node_id), species is normalized binomial Genus_species
  - species_taxonomy.tsv (species list; used to define *ordering* of exported species embeddings)

Outputs:
  - runs/<name>/checkpoints/*.pt  (includes "species_emb_weight" + "species_sorted")

Objective:
  - Triplet loss in cosine space:
      d(u,v)=1-cos(u,v)
      enforce d(a,p) + margin < d(a,n)

Sampling (NEW):
  You can choose how to sample positives on the tree:
    1) pos_sampling=within
       - pick positive uniformly among all leaf nodes within <= pos_max_hops hops from anchor
    2) pos_sampling=uniform_hop
       - first sample hop distance h ~ Uniform{1..pos_max_hops}
       - then sample positive uniformly among leaves at exactly hop distance h
       - if none exist for that h, resample h a few times, then fall back to "within"

  Negatives:
    - after picking a positive at hop distance h_pos, sample a negative uniformly among
      leaf nodes that are at least (h_pos + neg_min_hops) hops away from anchor.
    - implemented by excluding all nodes within <= (h_pos + neg_min_hops - 1) hops.

Dependencies:
  pip install torch numpy

Run example:
  python stage1_gnn_phylo.py \
    --data_dir output \
    --out_dir runs/gnn_phylo1 \
    --emb_dim 256 \
    --n_layers 3 \
    --steps_per_epoch 2000 \
    --epochs 20 \
    --batch_anchors 512 \
    --pos_max_hops 30 \
    --pos_sampling uniform_hop \
    --neg_min_hops 1 \
    --margin 0.1 \
    --lr 3e-3 \
    --amp
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Reproducibility
# -----------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# IO helpers
# -----------------------------

def read_species_taxonomy_tsv(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        if "species" not in idx:
            raise RuntimeError(f"{path} missing required column: species")
        species = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= idx["species"]:
                continue
            species.append(parts[idx["species"]])
    return sorted(set(species))


def read_leaf_map_tsv(path: Path) -> Dict[str, int]:
    m = {}
    with path.open("r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            sp, nid = line.rstrip("\n").split("\t")
            m[sp] = int(nid)
    return m


def read_nodes_tsv(path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      is_tip: np.int8 array [N]
      tip_name_by_node: list[str] length N ("" if not tip or missing)
    """
    rows = []
    max_id = -1
    with path.open("r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            nid_s, is_tip_s, name = line.rstrip("\n").split("\t")
            nid = int(nid_s)
            is_tip = int(is_tip_s)
            rows.append((nid, is_tip, name))
            max_id = max(max_id, nid)

    N = max_id + 1
    is_tip_arr = np.zeros((N,), dtype=np.int8)
    tip_name_by_node = [""] * N
    for nid, is_tip, name in rows:
        is_tip_arr[nid] = is_tip
        tip_name_by_node[nid] = name
    return is_tip_arr, tip_name_by_node


def load_edge_index(path: Path) -> np.ndarray:
    z = np.load(path)
    if "edge_index" not in z:
        raise RuntimeError(f"{path} missing edge_index")
    ei = z["edge_index"].astype(np.int64)
    if ei.shape[0] != 2:
        raise RuntimeError(f"edge_index must have shape [2, E], got {ei.shape}")
    return ei


# -----------------------------
# Graph utilities
# -----------------------------

def build_adj_lists(edge_index: np.ndarray, num_nodes: int) -> List[List[int]]:
    """
    edge_index is directed; for neighborhood sampling we treat it as (effectively) undirected
    because preprocessing saved both directions. We just use adjacency as given.
    """
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0]
    dst = edge_index[1]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)
    return adj


def bfs_leaf_buckets_and_cumulative(
    adj: List[List[int]],
    start: int,
    max_hops: int,
    leaf_set: Set[int],
) -> Tuple[Dict[int, List[int]], List[Set[int]]]:
    """
    BFS out to max_hops.

    Returns:
      leaf_by_hop: dict hop -> list of leaf node ids at exactly that hop (excluding start)
      cumulative_nodes: list of sets, cumulative_nodes[h] = nodes within <=h hops (includes start)
                        length = max_hops+1
    """
    leaf_by_hop: Dict[int, List[int]] = defaultdict(list)

    seen = {start}
    frontier = {start}
    cumulative_nodes: List[Set[int]] = [set([start])]

    for hop in range(1, max_hops + 1):
        nxt = set()
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)

        # collect leaves at exactly this hop
        for v in nxt:
            if v in leaf_set:
                leaf_by_hop[hop].append(v)

        # cumulative set up to this hop
        cumulative_nodes.append(cumulative_nodes[-1] | nxt)

        frontier = nxt
        if not frontier:
            # pad remaining cumulative sets (so indexing works)
            for _ in range(hop + 1, max_hops + 1):
                cumulative_nodes.append(set(cumulative_nodes[-1]))
            break

    return leaf_by_hop, cumulative_nodes


def sample_positive(
    leaf_by_hop: Dict[int, List[int]],
    pos_max_hops: int,
    rng: random.Random,
    strategy: str,
    max_h_resamples: int,
) -> Tuple[Optional[int], int]:
    """
    Returns (pos_node, pos_hop). pos_node=None if no candidates found.
    """
    # Quick total candidates in <=pos_max_hops
    counts = {h: len(leaf_by_hop.get(h, [])) for h in range(1, pos_max_hops + 1)}
    total = sum(counts.values())
    if total == 0:
        return None, 0

    if strategy == "within":
        # uniform among all leaves within <= pos_max_hops
        r = rng.randrange(total)
        acc = 0
        for h in range(1, pos_max_hops + 1):
            c = counts[h]
            if c == 0:
                continue
            if r < acc + c:
                pos = rng.choice(leaf_by_hop[h])
                return pos, h
            acc += c
        # should not happen
        h = max((h for h in range(1, pos_max_hops + 1) if counts[h] > 0), default=1)
        return rng.choice(leaf_by_hop[h]), h

    elif strategy == "uniform_hop":
        # sample hop distance first, then a leaf at that exact distance
        for _ in range(max_h_resamples):
            h = rng.randint(1, pos_max_hops)
            bucket = leaf_by_hop.get(h, [])
            if bucket:
                return rng.choice(bucket), h
        # fallback to within
        r = rng.randrange(total)
        acc = 0
        for h in range(1, pos_max_hops + 1):
            c = counts[h]
            if c == 0:
                continue
            if r < acc + c:
                return rng.choice(leaf_by_hop[h]), h
            acc += c
        h = max((h for h in range(1, pos_max_hops + 1) if counts[h] > 0), default=1)
        return rng.choice(leaf_by_hop[h]), h

    else:
        raise ValueError(f"Unknown pos sampling strategy: {strategy}")


# -----------------------------
# Simple GCN
# -----------------------------

class GCNLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, deg_inv: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D]
        edge_index: [2, E] (directed)
        deg_inv: [N] = 1/deg for mean aggregation

        mean aggregation:
          m[v] = mean_{u in N(v)} x[u]
        """
        src = edge_index[0]
        dst = edge_index[1]

        m = torch.zeros_like(x)
        m.index_add_(0, dst, x[src])
        m = m * deg_inv.unsqueeze(1)

        h = self.lin(m)
        h = F.gelu(h)
        h = self.dropout(h)
        return h


class PhyloGCN(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [GCNLayer(emb_dim, emb_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, edge_index: torch.Tensor, deg_inv: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight  # [N, D]
        for layer in self.layers:
            x = x + layer(x, edge_index, deg_inv)
        x = self.norm(x)
        return x


# -----------------------------
# Loss
# -----------------------------

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (a * b).sum(dim=-1)


def triplet_margin_cosine(z_a: torch.Tensor, z_p: torch.Tensor, z_n: torch.Tensor, margin: float) -> torch.Tensor:
    za = F.normalize(z_a, dim=-1)
    zp = F.normalize(z_p, dim=-1)
    zn = F.normalize(z_n, dim=-1)

    d_ap = cosine_distance(za, zp)
    d_an = cosine_distance(za, zn)
    return F.relu(d_ap - d_an + margin).mean()


# -----------------------------
# Export + checkpoint
# -----------------------------

@torch.no_grad()
def export_species_embeddings(
    model: PhyloGCN,
    edge_index_t: torch.Tensor,
    deg_inv_t: torch.Tensor,
    species_sorted: List[str],
    leaf_map: Dict[str, int],
) -> torch.Tensor:
    model.eval()
    X = model(edge_index_t, deg_inv_t)  # [N, D]
    rows = []
    D = X.shape[1]
    for sp in species_sorted:
        nid = leaf_map.get(sp, None)
        if nid is None:
            rows.append(torch.zeros((D,), device=X.device))
        else:
            rows.append(X[nid])
    W = torch.stack(rows, dim=0).detach().cpu()
    model.train()
    return W


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    args: argparse.Namespace,
    species_sorted: List[str],
    species_emb_weight: torch.Tensor,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "args": vars(args),
            # unified export for evaluation.py
            "species_sorted": species_sorted,
            "species_emb_weight": species_emb_weight,
        },
        str(path),
    )


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path, help="Preprocessing output dir")
    ap.add_argument("--out_dir", required=True, type=Path)

    # Model
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Sampling + loss
    ap.add_argument("--batch_anchors", type=int, default=256, help="anchors per step (leaf nodes)")
    ap.add_argument("--pos_max_hops", type=int, default=30, help="positives must be within <= this hop distance")
    ap.add_argument(
        "--pos_sampling",
        choices=["within", "uniform_hop"],
        default="within",
        help="Positive sampling strategy: within=uniform among all leaves within <=pos_max_hops; "
             "uniform_hop=sample hop first then leaf at that hop (fallbacks to within).",
    )
    ap.add_argument("--pos_max_hop_resamples", type=int, default=20, help="Resamples for uniform_hop before fallback")
    ap.add_argument(
        "--neg_min_hops",
        type=int,
        default=1,
        help="Negatives must be at least (pos_hop + neg_min_hops) away from anchor. "
             "neg_min_hops=1 means strictly farther than the chosen positive hop.",
    )
    ap.add_argument("--margin", type=float, default=0.1)

    # Training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=500)

    # Logging
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()
    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load graph + mappings
    edge_index = load_edge_index(args.data_dir / "edge_index.npz")
    is_tip, _ = read_nodes_tsv(args.data_dir / "nodes.tsv")
    leaf_map = read_leaf_map_tsv(args.data_dir / "leaf_map.tsv")
    species_sorted = read_species_taxonomy_tsv(args.data_dir / "species_taxonomy.tsv")

    num_nodes = int(is_tip.shape[0])
    leaf_nodes = sorted(set(leaf_map.values()))
    leaf_set = set(leaf_nodes)
    print(f"Graph nodes: {num_nodes}")
    print(f"Leaf nodes (species mapped): {len(leaf_nodes)}")
    print(f"Species in taxonomy table: {len(species_sorted)}")

    # Build adjacency for sampling
    adj = build_adj_lists(edge_index, num_nodes=num_nodes)

    # Torch tensors for message passing
    edge_index_t = torch.from_numpy(edge_index).long().to(device)

    # Degree for mean aggregation (aggregate into dst)
    dst = edge_index_t[1]
    deg = torch.zeros((num_nodes,), device=device, dtype=torch.float32)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg = torch.clamp(deg, min=1.0)
    deg_inv = 1.0 / deg

    model = PhyloGCN(num_nodes=num_nodes, emb_dim=args.emb_dim, n_layers=args.n_layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    metrics_path = args.out_dir / "metrics.jsonl"
    rng = random.Random(args.seed + 999)

    step = 0
    t0 = time.time()

    # For diagnostics of selected positive hops
    pos_hop_counter = Counter()

    print("Starting GNN training...")
    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):
            # sample anchors among leaf nodes
            anchors = [rng.choice(leaf_nodes) for _ in range(args.batch_anchors)]

            pos = []
            neg = []
            pos_hops = []

            for a in anchors:
                leaf_by_hop, cumulative_nodes = bfs_leaf_buckets_and_cumulative(
                    adj, start=a, max_hops=args.pos_max_hops, leaf_set=leaf_set
                )

                p, h_pos = sample_positive(
                    leaf_by_hop=leaf_by_hop,
                    pos_max_hops=args.pos_max_hops,
                    rng=rng,
                    strategy=args.pos_sampling,
                    max_h_resamples=args.pos_max_hop_resamples,
                )

                if p is None:
                    # no nearby positive found (rare unless pos_max_hops tiny / disconnected)
                    p = a
                    h_pos = 0

                # negative must be farther than positive by at least neg_min_hops
                # exclude nodes within <= (h_pos + neg_min_hops - 1)
                excl_h = max(0, min(args.pos_max_hops, h_pos + args.neg_min_hops - 1))
                excluded = cumulative_nodes[excl_h] if excl_h < len(cumulative_nodes) else cumulative_nodes[-1]

                tries = 0
                while True:
                    cand = rng.choice(leaf_nodes)
                    tries += 1
                    if cand not in excluded:
                        n = cand
                        break
                    if tries > 200:
                        # fallback: allow any random leaf (should be very rare)
                        n = cand
                        break

                pos.append(p)
                neg.append(n)
                pos_hops.append(h_pos)
                if h_pos > 0:
                    pos_hop_counter[h_pos] += 1

            anchors_t = torch.tensor(anchors, device=device, dtype=torch.long)
            pos_t = torch.tensor(pos, device=device, dtype=torch.long)
            neg_t = torch.tensor(neg, device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                X = model(edge_index_t, deg_inv)  # [N, D]
                z_a = X[anchors_t]
                z_p = X[pos_t]
                z_n = X[neg_t]
                loss = triplet_margin_cosine(z_a, z_p, z_n, margin=args.margin)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            step += 1

            if step % args.log_every == 0:
                it_s = step / max(1e-9, (time.time() - t0))
                top5 = pos_hop_counter.most_common(5)
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.6f} it/s={it_s:.2f} "
                    f"pos_hop_hist_top5={top5}"
                )
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type": "train",
                        "epoch": epoch,
                        "step": step,
                        "loss": float(loss.item()),
                        "pos_hop_hist_top5": top5,
                        "pos_max_hops": args.pos_max_hops,
                        "pos_sampling": args.pos_sampling,
                        "neg_min_hops": args.neg_min_hops,
                        "margin": args.margin,
                    }) + "\n")

            # periodic export + checkpoint
            if (step % args.eval_every == 0) or (step % args.save_every == 0):
                with torch.no_grad():
                    W_sp = export_species_embeddings(model, edge_index_t, deg_inv, species_sorted, leaf_map)
                ckpt_path = args.out_dir / "checkpoints" / f"ckpt_step{step:07d}.pt"
                save_checkpoint(
                    ckpt_path, model, opt, step, epoch, args,
                    species_sorted=species_sorted,
                    species_emb_weight=W_sp,
                )
                print("Saved:", ckpt_path)

    # final export
    with torch.no_grad():
        W_sp = export_species_embeddings(model, edge_index_t, deg_inv, species_sorted, leaf_map)
    ckpt_path = args.out_dir / "checkpoints" / f"ckpt_final_step{step:07d}.pt"
    save_checkpoint(
        ckpt_path, model, opt, step, args.epochs, args,
        species_sorted=species_sorted,
        species_emb_weight=W_sp,
    )
    print("Done. Final checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
