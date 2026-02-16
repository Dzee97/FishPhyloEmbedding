#!/usr/bin/env python3
"""
Stage-1 (phylogeny-only) training: learn species embeddings from the phylogeny graph using a GNN.

Goal:
- Use the pruned phylogeny graph to learn *leaf/species* embeddings that respect tree topology.
- No sequences involved yet. This is "Option 4 pretraining" groundwork: a phylogeny-only latent space.

Inputs (from preprocessing):
- edge_index.npz        (directed edges, shape [2, E])
- edge_attr.npz         (optional; branch_length per directed edge, shape [E, 1])  [not used by default]
- leaf_map.tsv          (species -> node_id in the tree graph)
- species_taxonomy.tsv  (species list; we use sorted species to match evaluation ordering)

Outputs:
- runs/<name>/checkpoints/*.pt
- runs/<name>/metrics.jsonl

Checkpoint format:
- Saves BOTH:
   (a) ckpt["species_embeddings"] as a float tensor [num_species, d]
   (b) ckpt["model"] state_dict containing a key "species_emb.weight" (for compatibility)

Dependencies:
  pip install torch numpy
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

def read_leaf_map_tsv(path: Path) -> Dict[str, int]:
    m = {}
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            sp, nid = line.rstrip("\n").split("\t")
            m[sp] = int(nid)
    return m


def read_species_list_from_species_taxonomy(path: Path) -> List[str]:
    # species_taxonomy.tsv header: species \t seq_count \t order \t family \t genus_raw
    species = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        if "species" not in idx:
            raise RuntimeError(f"{path} missing 'species' column")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            species.append(parts[idx["species"]])
    return sorted(species)


def load_edge_index_npz(path: Path) -> np.ndarray:
    arr = np.load(path)
    # expects key edge_index
    if "edge_index" not in arr:
        raise RuntimeError(f"{path} does not contain 'edge_index'")
    edge_index = arr["edge_index"]
    if edge_index.shape[0] != 2:
        raise RuntimeError(f"edge_index should have shape [2, E], got {edge_index.shape}")
    return edge_index.astype(np.int64)


# -----------------------------
# Graph utilities (topology sampling)
# -----------------------------

def build_adj_lists(num_nodes: int, edge_index: np.ndarray) -> List[List[int]]:
    """
    Build undirected adjacency lists from a directed edge_index.
    Your preprocessing stored both directions, but we still deduplicate.
    """
    adj = [set() for _ in range(num_nodes)]
    src = edge_index[0]
    dst = edge_index[1]
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return [list(s) for s in adj]


def bfs_collect_within(adj: List[List[int]], start: int, max_hops: int) -> Dict[int, int]:
    """
    Return dict node -> hop_distance for nodes within max_hops.
    """
    dist = {start: 0}
    q = [start]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        du = dist[u]
        if du >= max_hops:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
    return dist


def sample_triplets(
    leaf_node_ids: List[int],
    leaf_set: set,
    adj: List[List[int]],
    batch_size: int,
    pos_hops: int,
    neg_min_hops: int,
    max_tries: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample (anchor, positive, negative) triplets on *leaf nodes* using hop distances.

    - positive: a different leaf within <= pos_hops hops (if possible)
    - negative: a leaf with hop distance >= neg_min_hops (we enforce by BFS up to neg_min_hops-1 and rejection)

    If an anchor has no positive within pos_hops, we resample anchor.
    """
    A, P, N = [], [], []
    leaf_node_ids_arr = np.array(leaf_node_ids, dtype=np.int64)

    while len(A) < batch_size:
        a = int(np.random.choice(leaf_node_ids_arr))
        # get nodes within pos_hops, then filter to leaves
        near = bfs_collect_within(adj, a, max_hops=pos_hops)
        pos_cands = [n for n, d in near.items() if (n != a and n in leaf_set and d <= pos_hops)]
        if not pos_cands:
            continue
        p = int(np.random.choice(pos_cands))

        # negatives: reject if within neg_min_hops-1
        close = bfs_collect_within(adj, a, max_hops=max(0, neg_min_hops - 1))
        close_set = set(close.keys())

        n = None
        for _ in range(max_tries):
            cand = int(np.random.choice(leaf_node_ids_arr))
            if cand == a:
                continue
            if cand not in close_set:
                n = cand
                break
        if n is None:
            # fallback: random different leaf
            n = int(np.random.choice(leaf_node_ids_arr[leaf_node_ids_arr != a]))

        A.append(a)
        P.append(p)
        N.append(n)

    return np.array(A, dtype=np.int64), np.array(P, dtype=np.int64), np.array(N, dtype=np.int64)


# -----------------------------
# Simple GCN (no torch_geometric)
# -----------------------------

class GCNLayer(nn.Module):
    """
    Mean-aggregating GCN-like layer:
      h' = W_self h + W_nei mean_{j in N(i)} h_j
    """

    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.w_self = nn.Linear(d_in, d_out, bias=True)
        self.w_nei = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        # edge_index: [2, E] directed (we'll assume contains both directions, but it's fine either way)
        src = edge_index[0]
        dst = edge_index[1]

        # sum neighbor messages into dst
        msg = h[src]  # [E, d]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)  # sum over incoming

        # mean
        agg = agg / deg.clamp(min=1.0).unsqueeze(-1)

        out = self.w_self(h) + self.w_nei(agg)
        out = self.act(out)
        out = self.dropout(out)
        return out


class PhyloGNN(nn.Module):
    def __init__(self, num_nodes: int, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, d_model)
        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            GCNLayer(d_model, d_model, dropout=dropout) for _ in range(n_layers)
        ])

    def forward(self, edge_index: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        h = self.node_emb.weight
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, deg=deg)
        return h  # [num_nodes, d_model]


def triplet_margin_loss_cosine(z_a, z_p, z_n, margin: float) -> torch.Tensor:
    """
    Triplet loss using cosine distance:
      d(u,v) = 1 - cosine(u,v)
      loss = max(0, d(a,p) - d(a,n) + margin)
    """
    z_a = F.normalize(z_a, dim=-1)
    z_p = F.normalize(z_p, dim=-1)
    z_n = F.normalize(z_n, dim=-1)

    d_ap = 1.0 - (z_a * z_p).sum(dim=-1)
    d_an = 1.0 - (z_a * z_n).sum(dim=-1)
    return F.relu(d_ap - d_an + margin).mean()


@torch.no_grad()
def compute_species_embeddings(
    model: PhyloGNN,
    edge_index_t: torch.Tensor,
    deg_t: torch.Tensor,
    species_sorted: List[str],
    leaf_map: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    h = model(edge_index_t, deg_t)  # [num_nodes, d]
    # gather leaves in species_sorted order
    ids = torch.tensor([leaf_map[sp] for sp in species_sorted], dtype=torch.long, device=device)
    Z = h[ids].detach().cpu()
    return Z


def save_checkpoint(
    path: Path,
    model: PhyloGNN,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    args: argparse.Namespace,
    species_embeddings: torch.Tensor,
):
    """
    Save in a way evaluation doesn't care how embeddings were trained.
    We store:
      - ckpt["species_embeddings"] : [num_species, d]
      - ckpt["model"]["species_emb.weight"] : same tensor (compat with older loader)
      - plus model state for potential resume
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # fake state_dict entry for evaluation compatibility
    eval_sd = {"species_emb.weight": species_embeddings.clone()}

    torch.save(
        {
            "model": eval_sd,
            "gnn_state": model.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "args": vars(args),
            "species_embeddings": species_embeddings.clone(),
        },
        str(path),
    )


# -----------------------------
# Main training
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--edge_index_npz", required=True, type=Path)
    ap.add_argument("--leaf_map_tsv", required=True, type=Path)
    ap.add_argument("--species_taxonomy_tsv", required=True, type=Path)

    ap.add_argument("--out_dir", required=True, type=Path)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Triplet sampling topology params
    ap.add_argument("--pos_hops", type=int, default=2, help="positive leaf must be within this many hops")
    ap.add_argument("--neg_min_hops", type=int, default=6, help="negative leaf must be at least this many hops away")

    ap.add_argument("--batch_size", type=int, default=256, help="triplets per step")
    ap.add_argument("--steps_per_epoch", type=int, default=200, help="number of triplet batches per epoch")
    ap.add_argument("--epochs", type=int, default=50)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.2)

    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=500)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Load species + leaf map
    leaf_map = read_leaf_map_tsv(args.leaf_map_tsv)
    species_sorted = read_species_list_from_species_taxonomy(args.species_taxonomy_tsv)

    # Ensure all species in species_sorted exist in leaf_map
    missing = [sp for sp in species_sorted if sp not in leaf_map]
    if missing:
        raise RuntimeError(
            f"{len(missing)} species from species_taxonomy.tsv not found in leaf_map.tsv. "
            f"Example: {missing[:10]}"
        )

    # Load edge index
    edge_index = load_edge_index_npz(args.edge_index_npz)
    num_nodes = int(edge_index.max()) + 1

    # Build adj lists for sampling
    adj = build_adj_lists(num_nodes, edge_index)

    # Leaf node IDs in species-sorted order (and as a set)
    leaf_node_ids = [leaf_map[sp] for sp in species_sorted]
    leaf_set = set(leaf_node_ids)

    # Torch tensors for GNN
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)

    # Degree for mean aggregation: deg[dst] = number of incoming neighbors
    dst = edge_index_t[1]
    deg = torch.zeros((num_nodes,), dtype=torch.float32, device=device)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg_t = deg

    model = PhyloGNN(num_nodes=num_nodes, d_model=args.d_model, n_layers=args.n_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_path = args.out_dir / "metrics.jsonl"
    step = 0
    t0 = time.time()

    def log(obj: dict):
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    @torch.no_grad()
    def quick_eval(num_batches: int = 5) -> dict:
        model.eval()
        losses = []
        for _ in range(num_batches):
            A, P, N = sample_triplets(
                leaf_node_ids=leaf_node_ids,
                leaf_set=leaf_set,
                adj=adj,
                batch_size=min(512, args.batch_size),
                pos_hops=args.pos_hops,
                neg_min_hops=args.neg_min_hops,
            )
            h = model(edge_index_t, deg_t)
            z_a = h[torch.tensor(A, device=device)]
            z_p = h[torch.tensor(P, device=device)]
            z_n = h[torch.tensor(N, device=device)]
            loss = triplet_margin_loss_cosine(z_a, z_p, z_n, margin=args.margin)
            losses.append(float(loss.item()))
        model.train()
        return {"eval_triplet_loss": float(np.mean(losses))}

    # Initial eval
    ev0 = quick_eval(num_batches=10)
    print("Initial eval:", ev0)
    log({"step": step, "epoch": 0, "type": "eval", **ev0})

    print("Starting GNN training...")
    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):
            step += 1

            # Full-batch forward once (tree is small enough)
            h = model(edge_index_t, deg_t)  # [num_nodes, d]

            # Sample leaf triplets by topology
            A, P, N = sample_triplets(
                leaf_node_ids=leaf_node_ids,
                leaf_set=leaf_set,
                adj=adj,
                batch_size=args.batch_size,
                pos_hops=args.pos_hops,
                neg_min_hops=args.neg_min_hops,
            )
            A_t = torch.tensor(A, dtype=torch.long, device=device)
            P_t = torch.tensor(P, dtype=torch.long, device=device)
            N_t = torch.tensor(N, dtype=torch.long, device=device)

            z_a = h[A_t]
            z_p = h[P_t]
            z_n = h[N_t]

            loss = triplet_margin_loss_cosine(z_a, z_p, z_n, margin=args.margin)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                it_s = step / max(1e-9, (time.time() - t0))
                print(f"epoch={epoch} step={step} triplet_loss={loss.item():.4f} it/s={it_s:.2f}")
                log({"step": step, "epoch": epoch, "type": "train", "triplet_loss": float(loss.item())})

            if args.eval_every > 0 and step % args.eval_every == 0:
                ev = quick_eval(num_batches=10)
                print("Eval:", ev)
                log({"step": step, "epoch": epoch, "type": "eval", **ev})

            if args.save_every > 0 and step % args.save_every == 0:
                Z = compute_species_embeddings(
                    model=model,
                    edge_index_t=edge_index_t,
                    deg_t=deg_t,
                    species_sorted=species_sorted,
                    leaf_map=leaf_map,
                    device=device,
                )
                ckpt_path = args.out_dir / "checkpoints" / f"ckpt_step{step}.pt"
                save_checkpoint(
                    ckpt_path, model, optimizer,
                    step=step, epoch=epoch, args=args,
                    species_embeddings=Z
                )
                print("Saved:", ckpt_path)

    # Final save
    Z = compute_species_embeddings(
        model=model,
        edge_index_t=edge_index_t,
        deg_t=deg_t,
        species_sorted=species_sorted,
        leaf_map=leaf_map,
        device=device,
    )
    ckpt_path = args.out_dir / "checkpoints" / f"ckpt_final.pt"
    save_checkpoint(
        ckpt_path, model, optimizer,
        step=step, epoch=args.epochs, args=args,
        species_embeddings=Z
    )
    print("Saved final:", ckpt_path)
    print("Done.")


if __name__ == "__main__":
    main()
