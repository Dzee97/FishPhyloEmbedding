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
  - Positives sampled from local tree neighborhood (random walks)
  - Negatives sampled outside that neighborhood

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
    --pos_walk_len 2 \
    --neg_exclusion_hops 3 \
    --margin 0.2 \
    --lr 3e-3 \
    --amp
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

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
    edge_index is directed; for neighborhood sampling we treat it as undirected by adding both directions.
    """
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0]
    dst = edge_index[1]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)
    return adj


def bfs_hops(adj: List[List[int]], start: int, max_hops: int) -> Set[int]:
    """
    Return set of nodes within <= max_hops hops (including start).
    """
    seen = {start}
    frontier = {start}
    for _ in range(max_hops):
        nxt = set()
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    return seen


def random_walk_leaf(adj: List[List[int]], start: int, walk_len: int, leaf_set: Set[int], rng: random.Random) -> int:
    """
    Random walk from start for walk_len steps, return the final node if it's a leaf,
    otherwise keep stepping a bit more until hitting a leaf (bounded tries).
    """
    u = start
    for _ in range(walk_len):
        neigh = adj[u]
        if not neigh:
            break
        u = rng.choice(neigh)

    if u in leaf_set:
        return u

    # try a few extra steps to land on a leaf
    for _ in range(10):
        neigh = adj[u]
        if not neigh:
            break
        u = rng.choice(neigh)
        if u in leaf_set:
            return u

    # fallback: return start (worst case)
    return start


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

        # aggregate neighbors: sum then multiply by 1/deg
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
            x = x + layer(x, edge_index, deg_inv)  # residual
        x = self.norm(x)
        return x


# -----------------------------
# Loss
# -----------------------------

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: [..., D] assumed normalized
    return 1.0 - (a * b).sum(dim=-1)


def triplet_margin_cosine(z_a: torch.Tensor, z_p: torch.Tensor, z_n: torch.Tensor, margin: float) -> torch.Tensor:
    """
    z_*: [B, D] (not necessarily normalized)
    """
    za = F.normalize(z_a, dim=-1)
    zp = F.normalize(z_p, dim=-1)
    zn = F.normalize(z_n, dim=-1)

    d_ap = cosine_distance(za, zp)
    d_an = cosine_distance(za, zn)
    loss = F.relu(d_ap - d_an + margin).mean()
    return loss


# -----------------------------
# Training loop
# -----------------------------

@torch.no_grad()
def export_species_embeddings(
    model: PhyloGCN,
    edge_index_t: torch.Tensor,
    deg_inv_t: torch.Tensor,
    species_sorted: List[str],
    leaf_map: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    X = model(edge_index_t, deg_inv_t)  # [N, D]
    # species order is species_sorted
    rows = []
    for sp in species_sorted:
        nid = leaf_map.get(sp, None)
        if nid is None:
            # should not happen if consistent preprocessing/training
            rows.append(torch.zeros((X.shape[1],), device=X.device))
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
            "species_emb_weight": species_emb_weight,  # [num_species, dim]
        },
        str(path),
    )


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
    ap.add_argument("--pos_walk_len", type=int, default=2, help="random walk length to sample positives")
    ap.add_argument("--neg_exclusion_hops", type=int, default=3, help="negatives must be outside this hop radius")
    ap.add_argument("--margin", type=float, default=0.2)

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
    ap.add_argument("--eval_every", type=int, default=500)  # exports embeddings periodically

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

    # Build adjacency for sampling (treat as undirected-ish using directed edges)
    adj = build_adj_lists(edge_index, num_nodes=num_nodes)

    # Torch tensors for message passing
    edge_index_t = torch.from_numpy(edge_index).long().to(device)

    # Degree for mean aggregation: deg_inv[v] = 1/deg_in(v)
    # (since we aggregate into dst)
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

    print("Starting GNN training...")
    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):
            # sample anchors among leaf nodes
            anchors = [rng.choice(leaf_nodes) for _ in range(args.batch_anchors)]

            # sample positives by local random walk
            pos = [random_walk_leaf(adj, a, args.pos_walk_len, leaf_set, rng) for a in anchors]

            # negatives: sample leaf outside hop neighborhood
            neg = []
            for a in anchors:
                excluded = bfs_hops(adj, a, max_hops=args.neg_exclusion_hops)
                # ensure we exclude non-leaves too by checking leaf set
                tries = 0
                while True:
                    cand = rng.choice(leaf_nodes)
                    tries += 1
                    if cand not in excluded:
                        neg.append(cand)
                        break
                    if tries > 50:
                        # fallback: just pick random leaf
                        neg.append(cand)
                        break

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

            if step % 50 == 0:
                it_s = step / max(1e-9, (time.time() - t0))
                print(f"epoch={epoch} step={step} loss={loss.item():.6f} it/s={it_s:.2f}")

            if step % 50 == 0:
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"type": "train", "epoch": epoch,
                            "step": step, "loss": float(loss.item())}) + "\n")

            # periodic export + checkpoint
            if (step % args.eval_every == 0) or (step % args.save_every == 0):
                with torch.no_grad():
                    W_sp = export_species_embeddings(
                        model, edge_index_t, deg_inv, species_sorted, leaf_map, device
                    )
                ckpt_path = args.out_dir / "checkpoints" / f"ckpt_step{step:07d}.pt"
                save_checkpoint(
                    ckpt_path, model, opt, step, epoch, args,
                    species_sorted=species_sorted,
                    species_emb_weight=W_sp,
                )
                print("Saved:", ckpt_path)

    # final export
    with torch.no_grad():
        W_sp = export_species_embeddings(model, edge_index_t, deg_inv, species_sorted, leaf_map, device)
    ckpt_path = args.out_dir / "checkpoints" / f"ckpt_final_step{step:07d}.pt"
    save_checkpoint(
        ckpt_path, model, opt, step, args.epochs, args,
        species_sorted=species_sorted,
        species_emb_weight=W_sp,
    )
    print("Done. Final checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
