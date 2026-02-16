#!/usr/bin/env python3
"""
Stage 1 (phylogeny-only): Learn species embeddings from the pruned phylogeny graph using a GNN.

Inputs (from preprocessing out_dir):
  - edge_index.npz     (directed edges, shape [2, E])  [preprocessing stores both directions]
  - nodes.tsv          (node_id, is_tip, name)
  - leaf_map.tsv       (species -> node_id)
  - species_taxonomy.tsv (species list; defines ordering of exported species embeddings)

Outputs:
  - runs/<name>/checkpoints/*.pt  (includes "species_emb_weight" + "species_sorted")

Objective:
  - Triplet margin loss in cosine space:
      d(u,v)=1-cos(u,v)
      enforce d(a,p) + margin < d(a,n)

Triplet sampling (NEW; hop-based):
  1) Sample anchor leaf a uniformly from leaf nodes.
  2) BFS from a up to --pos_max_hops.
  3) Sample positive p among leaves within <= pos_max_hops hops (excluding a).
  4) Let d_pos = dist(a,p).
  5) Sample negative n among leaves with dist(a,n) >= d_pos+1
     implemented by rejection: pick random leaf not in "leaves within d_pos hops".

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
    --pos_max_hops 5 \
    --margin 0.2 \
    --lr 3e-3 \
    --amp
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict, deque
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
    edge_index is directed. In your preprocessing, edges were written in both directions,
    so this adjacency list is effectively undirected for BFS/hop distances.
    """
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0]
    dst = edge_index[1]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)
    return adj


# -----------------------------
# NEW: Hop-based triplet sampler
# -----------------------------

class HopTripletSampler:
    """
    Triplet sampling on a tree/graph using hop distances.

    anchor: random leaf
    positive: random leaf within <= max_pos_hops hops
    negative: random leaf with hop distance >= (d_pos + 1)
              enforced by: negative not in visited_leaves_within_d_pos
    """

    def __init__(
        self,
        adj: List[List[int]],
        leaf_nodes: List[int],
        max_pos_hops: int = 5,
        rng: Optional[random.Random] = None,
        max_anchor_tries: int = 100,
        max_neg_tries: int = 2000,
    ):
        self.adj = adj
        self.leaf_nodes = list(leaf_nodes)
        self.leaf_set = set(self.leaf_nodes)
        self.max_pos_hops = int(max_pos_hops)
        self.rng = rng or random.Random()
        self.max_anchor_tries = int(max_anchor_tries)
        self.max_neg_tries = int(max_neg_tries)

        if self.max_pos_hops < 1:
            raise ValueError("max_pos_hops must be >= 1")
        if len(self.leaf_nodes) < 3:
            raise ValueError("Need at least 3 leaves to sample triplets.")

    def _bfs_leaves_by_dist(self, anchor: int, max_hops: int) -> Tuple[Dict[int, List[int]], Dict[int, Set[int]]]:
        """
        BFS from anchor up to max_hops.
        Returns:
          leaves_by_dist[d] = list of leaf node ids at exact hop distance d (d>=1)
          leaves_within[d] = set of leaf node ids with distance <= d
        """
        q = deque([anchor])
        dist = {anchor: 0}

        leaves_by_dist = defaultdict(list)

        while q:
            u = q.popleft()
            du = dist[u]
            if du == max_hops:
                continue
            for v in self.adj[u]:
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
                    dv = du + 1
                    if dv >= 1 and v in self.leaf_set:
                        leaves_by_dist[dv].append(v)

        leaves_within: Dict[int, Set[int]] = {}
        running: Set[int] = set()
        for d in range(1, max_hops + 1):
            for v in leaves_by_dist.get(d, []):
                running.add(v)
            leaves_within[d] = set(running)

        return dict(leaves_by_dist), leaves_within

    def sample_triplet(self) -> Tuple[int, int, int, int]:
        """
        Returns: (anchor_leaf, pos_leaf, neg_leaf, d_pos)
        """
        for _ in range(self.max_anchor_tries):
            a = self.rng.choice(self.leaf_nodes)

            leaves_by_dist, leaves_within = self._bfs_leaves_by_dist(a, self.max_pos_hops)

            candidates = []
            for d in range(1, self.max_pos_hops + 1):
                candidates.extend(leaves_by_dist.get(d, []))
            if not candidates:
                continue

            p = self.rng.choice(candidates)

            d_pos = None
            for d in range(1, self.max_pos_hops + 1):
                if p in leaves_by_dist.get(d, []):
                    d_pos = d
                    break
            if d_pos is None:
                continue

            forbidden = leaves_within.get(d_pos, set())
            forbidden.add(a)

            n = None
            for _t in range(self.max_neg_tries):
                cand = self.rng.choice(self.leaf_nodes)
                if cand not in forbidden:
                    n = cand
                    break

            if n is None:
                allowed = [x for x in self.leaf_nodes if x not in forbidden]
                if not allowed:
                    continue
                n = self.rng.choice(allowed)

            return a, p, n, d_pos

        raise RuntimeError("Failed to sample a valid triplet after many anchor retries.")

    def sample_batch(self, batch_size: int) -> Tuple[List[int], List[int], List[int], List[int]]:
        A, P, N, D = [], [], [], []
        for _ in range(batch_size):
            a, p, n, d = self.sample_triplet()
            A.append(a)
            P.append(p)
            N.append(n)
            D.append(d)
        return A, P, N, D


# -----------------------------
# Simple GCN
# -----------------------------

class GCNLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out)
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


class PhyloGCN(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([GCNLayer(emb_dim, emb_dim, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, edge_index: torch.Tensor, deg_inv: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight
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
    for sp in species_sorted:
        nid = leaf_map.get(sp, None)
        if nid is None:
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
            "species_sorted": species_sorted,
            "species_emb_weight": species_emb_weight,  # [num_species, dim]
        },
        str(path),
    )


# -----------------------------
# Training loop
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path, help="Preprocessing output dir")
    ap.add_argument("--out_dir", required=True, type=Path)

    # Model
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Sampling + loss (NEW)
    ap.add_argument("--batch_anchors", type=int, default=256, help="triplets per step")
    ap.add_argument("--pos_max_hops", type=int, default=5, help="positive must be within this hop radius")
    ap.add_argument("--max_anchor_tries", type=int, default=100)
    ap.add_argument("--max_neg_tries", type=int, default=2000)
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
    ap.add_argument("--eval_every", type=int, default=500)

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
    print(f"Graph nodes: {num_nodes}")
    print(f"Leaf nodes (species mapped): {len(leaf_nodes)}")
    print(f"Species in taxonomy table: {len(species_sorted)}")

    # Adjacency for BFS sampling
    adj = build_adj_lists(edge_index, num_nodes=num_nodes)

    # Torch tensors for message passing
    edge_index_t = torch.from_numpy(edge_index).long().to(device)

    # Degree for mean aggregation into dst
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

    sampler = HopTripletSampler(
        adj=adj,
        leaf_nodes=leaf_nodes,
        max_pos_hops=args.pos_max_hops,
        rng=rng,
        max_anchor_tries=args.max_anchor_tries,
        max_neg_tries=args.max_neg_tries,
    )

    step = 0
    t0 = time.time()
    hop_hist = Counter()

    print("Starting GNN training (hop-based triplets)...")
    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):
            anchors, pos, neg, dpos = sampler.sample_batch(args.batch_anchors)
            hop_hist.update(dpos)

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
                # show a tiny hop summary
                top_hops = ", ".join([f"{d}:{c}" for d, c in hop_hist.most_common(5)])
                print(f"epoch={epoch} step={step} loss={loss.item():.6f} it/s={it_s:.2f} d_pos(top)={top_hops}")

                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type": "train",
                        "epoch": epoch,
                        "step": step,
                        "loss": float(loss.item()),
                        "pos_hop_hist_top5": hop_hist.most_common(5),
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
