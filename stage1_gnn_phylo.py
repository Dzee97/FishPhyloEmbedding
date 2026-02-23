#!/usr/bin/env python3
"""
Stage 1 (phylogeny-only): Learn species embeddings from the pruned phylogeny graph using a GNN.

Inputs (from preprocessing out_dir):
  - edge_index.npz
  - nodes.tsv
  - leaf_map.tsv
  - species_taxonomy.tsv
  - NEW: leaf_hops.npz  (leaf_nodes [L], hop_dist [L,L])

Outputs:
  - runs/<name>/checkpoints/*.pt  (includes "species_emb_weight" + "species_sorted")

Triplet loss in cosine space:
  d(u,v)=1-cos(u,v)
  enforce d(a,p) + margin < d(a,n)

Sampling now uses the precomputed leaf hop matrix (no BFS per step):

Positives:
  --pos_sampling within:
     pick positive uniformly among all leaves with 1..pos_max_hops hops from anchor
  --pos_sampling uniform_hop:
     sample hop h ~ Uniform{1..pos_max_hops}, then pick uniformly among leaves at hop h
     (resamples; fallback to within)

Negatives:
  --neg_sampling outside:
     pick uniformly among leaves with hop >= (h_pos + neg_min_hops)
  --neg_sampling uniform_hop:
     sample hop h_neg ~ Uniform{min_neg .. max_neg} then pick uniformly among leaves at hop h_neg
     where min_neg = h_pos + neg_min_hops
     max_neg = --neg_max_hops if set else max finite hop for that anchor
     (resamples; fallback to outside)

GNN types:
  --gnn_type gcn   : mean neighbor aggregation (like your current)
  --gnn_type sage  : GraphSAGE mean aggregator with self+neighbor concatenation

Dependencies:
  pip install torch numpy
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

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


def load_leaf_hops(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    if "leaf_nodes" not in z or "hop_dist" not in z:
        raise RuntimeError(f"{path} must contain leaf_nodes and hop_dist arrays")
    leaf_nodes = z["leaf_nodes"].astype(np.int32)
    hop_dist = z["hop_dist"].astype(np.uint16)
    if hop_dist.shape[0] != hop_dist.shape[1]:
        raise RuntimeError(f"hop_dist must be square [L,L], got {hop_dist.shape}")
    if hop_dist.shape[0] != leaf_nodes.shape[0]:
        raise RuntimeError("leaf_nodes length must match hop_dist size")
    return leaf_nodes, hop_dist


# -----------------------------
# Sampling utilities (hop matrix)
# -----------------------------

INF_HOP = np.uint16(65535)


class HopSampler:
    """
    Uses a precomputed hop distance matrix between leaves.
    Provides positive/negative sampling according to your strategies, without BFS per step.

    To avoid scanning the full row (L) repeatedly, we build a per-anchor cache
    the first time a leaf is used as anchor.
    """

    def __init__(
        self,
        hop_dist: np.ndarray,      # [L,L] uint16
        rng: random.Random,
        pos_max_hops: int,
        pos_sampling: str,
        pos_max_hop_resamples: int,
        neg_min_hops: int,
        neg_sampling: str,
        neg_max_hops: Optional[int],
        neg_max_hop_resamples: int,
    ):
        self.hop = hop_dist
        self.L = int(hop_dist.shape[0])
        self.rng = rng

        self.pos_max_hops = int(pos_max_hops)
        self.pos_sampling = pos_sampling
        self.pos_max_hop_resamples = int(pos_max_hop_resamples)

        self.neg_min_hops = int(neg_min_hops)
        self.neg_sampling = neg_sampling
        self.neg_max_hops = None if neg_max_hops is None else int(neg_max_hops)
        self.neg_max_hop_resamples = int(neg_max_hop_resamples)

        # cache[ai] = dict with:
        #   "by_hop": {h: np.ndarray leaf indices at exactly hop h}
        #   "within": np.ndarray leaf indices at hop 1..pos_max_hops
        #   "max_finite": int
        self.cache: Dict[int, dict] = {}

    def _ensure_cache(self, ai: int):
        if ai in self.cache:
            return

        row = self.hop[ai]  # [L]
        # finite distances exclude INF and exclude self (0)
        finite = row[(row != INF_HOP)]
        max_finite = int(finite.max()) if finite.size > 0 else 0

        by_hop = {}
        # build buckets only up to max(pos_max_hops, neg_max_hops if provided)
        H = self.pos_max_hops
        if self.neg_max_hops is not None:
            H = max(H, self.neg_max_hops)
        H = min(H, max_finite) if max_finite > 0 else H

        # One scan per hop would be slow; instead do one pass over row values:
        # gather indices for hop 1..H
        # (still O(L), but only once per anchor leaf)
        for h in range(1, H + 1):
            idx = np.where(row == np.uint16(h))[0]
            if idx.size > 0:
                by_hop[h] = idx

        within_list = []
        for h in range(1, min(self.pos_max_hops, H) + 1):
            if h in by_hop:
                within_list.append(by_hop[h])
        within = np.concatenate(within_list) if within_list else np.empty((0,), dtype=np.int64)

        self.cache[ai] = {
            "by_hop": by_hop,
            "within": within,
            "max_finite": max_finite,
        }

    def sample_positive(self, ai: int) -> Tuple[int, int]:
        """
        Returns (pos_leaf_index, pos_hop).
        If no candidates exist, returns (ai, 0).
        """
        self._ensure_cache(ai)
        c = self.cache[ai]
        by_hop = c["by_hop"]
        within = c["within"]

        if within.size == 0:
            return ai, 0

        if self.pos_sampling == "within":
            pj = int(within[self.rng.randrange(int(within.size))])
            h = int(self.hop[ai, pj])
            return pj, h

        if self.pos_sampling == "uniform_hop":
            # try sampling hop first
            for _ in range(self.pos_max_hop_resamples):
                h = self.rng.randint(1, self.pos_max_hops)
                bucket = by_hop.get(h, None)
                if bucket is not None and bucket.size > 0:
                    pj = int(bucket[self.rng.randrange(int(bucket.size))])
                    return pj, h
            # fallback to within
            pj = int(within[self.rng.randrange(int(within.size))])
            h = int(self.hop[ai, pj])
            return pj, h

        raise ValueError(f"Unknown pos_sampling: {self.pos_sampling}")

    def sample_negative(self, ai: int, h_pos: int) -> int:
        """
        Returns neg_leaf_index.
        Enforces hop(ai, neg) >= (h_pos + neg_min_hops), unless fallback happens.
        """
        self._ensure_cache(ai)
        row = self.hop[ai]
        min_neg = int(h_pos + self.neg_min_hops)

        # Determine max hop to consider (for uniform_hop)
        max_finite = self.cache[ai]["max_finite"]
        max_neg = self.neg_max_hops if self.neg_max_hops is not None else max_finite
        max_neg = max(min_neg, int(max_neg))

        if self.neg_sampling == "outside":
            # uniform among all leaves farther than threshold
            cand = np.where((row != INF_HOP) & (row >= np.uint16(min_neg)))[0]
            if cand.size == 0:
                # fallback: any random leaf
                return self.rng.randrange(self.L)
            return int(cand[self.rng.randrange(int(cand.size))])

        if self.neg_sampling == "uniform_hop":
            # sample hop first, then choose leaf at exactly that hop
            # resample a few times, then fallback to outside
            for _ in range(self.neg_max_hop_resamples):
                if min_neg > max_neg:
                    break
                h = self.rng.randint(min_neg, max_neg)
                bucket = np.where(row == np.uint16(h))[0]
                if bucket.size > 0:
                    return int(bucket[self.rng.randrange(int(bucket.size))])

            cand = np.where((row != INF_HOP) & (row >= np.uint16(min_neg)))[0]
            if cand.size == 0:
                return self.rng.randrange(self.L)
            return int(cand[self.rng.randrange(int(cand.size))])

        raise ValueError(f"Unknown neg_sampling: {self.neg_sampling}")


# -----------------------------
# GNNs
# -----------------------------

class GCNLayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, deg_inv: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        m = torch.zeros_like(x)
        m.index_add_(0, dst, x[src])
        m = m * deg_inv.unsqueeze(1)
        h = self.lin(m)
        h = F.gelu(h)
        h = self.dropout(h)
        return h


class SAGELayer(nn.Module):
    """
    GraphSAGE (mean) layer:
      m[v] = mean_{u in N(v)} x[u]
      h[v] = W [x[v] || m[v]]
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, deg_inv: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        m = torch.zeros_like(x)
        m.index_add_(0, dst, x[src])
        m = m * deg_inv.unsqueeze(1)

        h = torch.cat([x, m], dim=1)
        h = self.lin(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return h


class PhyloGNN(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, n_layers: int, dropout: float, gnn_type: str):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)

        if gnn_type == "gcn":
            layer_cls = GCNLayer
        elif gnn_type == "sage":
            layer_cls = SAGELayer
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.layers = nn.ModuleList([layer_cls(emb_dim, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, edge_index: torch.Tensor, deg_inv: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight
        for layer in self.layers:
            x = x + layer(x, edge_index, deg_inv)  # residual
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
    model: PhyloGNN,
    edge_index_t: torch.Tensor,
    deg_inv_t: torch.Tensor,
    species_sorted: List[str],
    leaf_map: Dict[str, int],
) -> torch.Tensor:
    model.eval()
    X = model(edge_index_t, deg_inv_t)  # [N, D]
    rows = []
    D = int(X.shape[1])
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
    ap.add_argument("--gnn_type", choices=["gcn", "sage"], default="gcn")
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Sampling + loss
    ap.add_argument("--batch_anchors", type=int, default=256, help="anchors per step (leaf indices)")
    ap.add_argument("--pos_max_hops", type=int, default=30)
    ap.add_argument("--pos_sampling", choices=["within", "uniform_hop"], default="within")
    ap.add_argument("--pos_max_hop_resamples", type=int, default=30)

    ap.add_argument("--neg_min_hops", type=int, default=1)
    ap.add_argument("--neg_sampling", choices=["outside", "uniform_hop"], default="outside")
    ap.add_argument("--neg_max_hops", type=int, default=None,
                    help="Max hop for negative uniform_hop sampling. If not set, uses max finite hop for anchor.")
    ap.add_argument("--neg_max_hop_resamples", type=int, default=50)

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

    # Load leaf hop matrix
    leaf_nodes, hop_dist = load_leaf_hops(args.data_dir / "leaf_hops.npz")
    L = int(len(leaf_nodes))
    print(f"Leaf hop matrix: L={L}, dtype={hop_dist.dtype}, shape={hop_dist.shape}")

    num_nodes = int(is_tip.shape[0])
    print(f"Graph nodes: {num_nodes}")
    print(f"Species in taxonomy table: {len(species_sorted)}")

    # Torch tensors for message passing
    edge_index_t = torch.from_numpy(edge_index).long().to(device)

    # Degree for mean aggregation (aggregate into dst)
    dst = edge_index_t[1]
    deg = torch.zeros((num_nodes,), device=device, dtype=torch.float32)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg = torch.clamp(deg, min=1.0)
    deg_inv = 1.0 / deg

    model = PhyloGNN(
        num_nodes=num_nodes,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    metrics_path = args.out_dir / "metrics.jsonl"
    rng = random.Random(args.seed + 999)

    sampler = HopSampler(
        hop_dist=hop_dist,
        rng=rng,
        pos_max_hops=args.pos_max_hops,
        pos_sampling=args.pos_sampling,
        pos_max_hop_resamples=args.pos_max_hop_resamples,
        neg_min_hops=args.neg_min_hops,
        neg_sampling=args.neg_sampling,
        neg_max_hops=args.neg_max_hops,
        neg_max_hop_resamples=args.neg_max_hop_resamples,
    )

    step = 0
    t0 = time.time()

    pos_hop_counter = Counter()

    print("Starting GNN training...")
    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):

            # sample anchor leaf indices (0..L-1)
            anchors_li = [rng.randrange(L) for _ in range(args.batch_anchors)]

            pos_li = []
            neg_li = []
            pos_hops = []

            for ai in anchors_li:
                pj, h_pos = sampler.sample_positive(ai)
                nj = sampler.sample_negative(ai, h_pos=h_pos)

                pos_li.append(pj)
                neg_li.append(nj)
                pos_hops.append(h_pos)
                if h_pos > 0:
                    pos_hop_counter[h_pos] += 1

            # Map leaf indices -> graph node ids
            anchors_nodes = leaf_nodes[np.array(anchors_li, dtype=np.int64)]
            pos_nodes = leaf_nodes[np.array(pos_li, dtype=np.int64)]
            neg_nodes = leaf_nodes[np.array(neg_li, dtype=np.int64)]

            anchors_t = torch.from_numpy(anchors_nodes).to(device=device, dtype=torch.long)
            pos_t = torch.from_numpy(pos_nodes).to(device=device, dtype=torch.long)
            neg_t = torch.from_numpy(neg_nodes).to(device=device, dtype=torch.long)

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
                        "neg_sampling": args.neg_sampling,
                        "neg_min_hops": args.neg_min_hops,
                        "neg_max_hops": args.neg_max_hops,
                        "margin": args.margin,
                        "gnn_type": args.gnn_type,
                        "n_layers": args.n_layers,
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
