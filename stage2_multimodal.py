#!/usr/bin/env python3
"""
Stage 2 (multimodal): Combine the sequence Transformer and the phylogeny GNN to learn a shared species embedding space.

This script is designed to KEEP your existing stage1 scripts unchanged, and instead re-use their building blocks
via imports.

It supports three training modes (descriptive names, not A/B/C1):

1) freeze_phylo_align
   - Load a pretrained phylogeny GNN checkpoint.
   - Freeze it (or even avoid running it) and align the sequence encoder to the fixed phylogeny species embeddings
     using a CLIP/InfoNCE loss.
   - Safest / most stable.

2) joint_clip
   - Train sequence encoder and phylogeny GNN end-to-end using ONLY the CLIP/InfoNCE alignment loss.
   - Lets both modalities co-adapt, but can drift away from phylogenetic structure.

3) joint_clip_triplet
   - Train end-to-end with CLIP/InfoNCE alignment loss PLUS a phylogeny triplet regularizer on leaves.
   - Loss = clip_loss + lambda_triplet * triplet_loss
   - Often the best “end-to-end” compromise: alignment + keeps phylo geometry sane.

Inputs:
- sequences.jsonl (from preprocessing)
- preprocessing out_dir (for phylogeny graph + leaf_hops.npz + leaf_map.tsv + species_taxonomy.tsv)
- pretrained phylogeny GNN checkpoint (.pt) (for freeze_phylo_align and as init for joint modes)

Outputs:
- checkpoints/*.pt  (exports "species_sorted" + "species_emb_weight" for evaluation.py)
- metrics.jsonl

Dependencies:
  pip install torch numpy
  (relies on your existing stage1_baseline.py and stage1_gnn_phylo.py being importable)
"""

import argparse
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- Reuse your existing building blocks (no changes to stage1 scripts needed) ----
import stage1_baseline as s1_seq
import stage1_gnn_phylo as s1_phy


# -----------------------------
# Reproducibility
# -----------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# CLIP/InfoNCE loss (same as stage1_baseline)
# -----------------------------

def clip_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Symmetric InfoNCE (CLIP-style).
      - z_a: [B, D]
      - z_b: [B, D]
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)

    logits = (z_a @ z_b.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.T, targets)
    loss = 0.5 * (loss_a + loss_b)

    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc1 = (pred == targets).float().mean().item()
        k = min(5, logits.size(1))
        topk = logits.topk(k=k, dim=1).indices
        acc5 = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    return loss, {"acc@1": acc1, "acc@5": acc5}


# -----------------------------
# Models
# -----------------------------

class SeqOnlyModel(nn.Module):
    """
    Sequence encoder only (Transformer + projection), reusing stage1_baseline.SequenceTransformer.
    """

    def __init__(self, d_model: int, emb_dim: int, n_layers: int, n_heads: int,
                 ff_mult: int, dropout: float, max_len: int):
        super().__init__()
        self.seq_encoder = s1_seq.SequenceTransformer(
            vocab_size=len(s1_seq.VOCAB),
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            max_len=max_len,
            out_dim=emb_dim,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.seq_encoder(tokens)


# -----------------------------
# IO: torch.load compat (PyTorch 2.6 weights_only default)
# -----------------------------

def torch_load_compat(path: Path):
    """
    PyTorch 2.6+ defaults weights_only=True, which can fail if checkpoint contains
    non-allowlisted objects (e.g., pathlib.PosixPath inside args).

    We:
      1) try weights_only=True
      2) fallback to weights_only=False if needed (ONLY if you trust the checkpoint).
    Also compatible with older torch versions that don't support weights_only.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")
    except Exception as e:
        print(
            "WARNING: torch.load(weights_only=True) failed.\n"
            "Retrying with weights_only=False (this unpickles arbitrary objects).\n"
            "Only do this for checkpoints you trust.\n"
            f"Original error: {repr(e)}"
        )
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")


# -----------------------------
# Utilities: mapping species_id -> leaf node id
# -----------------------------

def build_species_to_leaf_node_ids(
    species_sorted: List[str],
    leaf_map: Dict[str, int],
) -> np.ndarray:
    """
    Returns:
      species_to_leaf_node: int64 array [num_species], maps species_id to graph node id
      Missing species should not happen, but is set to -1 defensively.
    """
    out = np.full((len(species_sorted),), -1, dtype=np.int64)
    for i, sp in enumerate(species_sorted):
        out[i] = int(leaf_map.get(sp, -1))
    if np.any(out < 0):
        missing = [species_sorted[i] for i in np.where(out < 0)[0][:20].tolist()]
        raise RuntimeError(
            "Some species from sequences.jsonl / taxonomy are missing in leaf_map.tsv.\n"
            f"Examples: {missing}"
        )
    return out


# -----------------------------
# Checkpoint saving
# -----------------------------

@torch.no_grad()
def export_species_emb_weight_from_gnn(
    gnn: nn.Module,
    edge_index_t: torch.Tensor,
    deg_inv_t: torch.Tensor,
    species_sorted: List[str],
    leaf_map: Dict[str, int],
) -> torch.Tensor:
    # Use the helper from stage1_gnn_phylo, which extracts leaf embeddings in species_sorted order.
    return s1_phy.export_species_embeddings(
        model=gnn,
        edge_index_t=edge_index_t,
        deg_inv_t=deg_inv_t,
        species_sorted=species_sorted,
        leaf_map=leaf_map,
    )


def save_checkpoint(
    path: Path,
    seq_model: nn.Module,
    gnn_model: nn.Module,
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
            "seq_model": seq_model.state_dict(),
            "gnn_model": gnn_model.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "args": vars(args),
            # unified export for evaluation.py
            "species_sorted": species_sorted,
            "species_emb_weight": species_emb_weight.detach().cpu(),
        },
        str(path),
    )


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data_jsonl", required=True, type=Path, help="sequences.jsonl (from preprocessing)")
    ap.add_argument("--data_dir", required=True, type=Path, help="preprocessing output dir (graph + leaf_hops + maps)")
    ap.add_argument("--out_dir", required=True, type=Path)

    # Mode
    ap.add_argument(
        "--mode",
        choices=["freeze_phylo_align", "joint_clip", "joint_clip_triplet"],
        default="freeze_phylo_align",
        help="Training pipeline choice"
    )

    # Init phylo
    ap.add_argument("--phylo_ckpt", required=True, type=Path, help="pretrained stage1_gnn_phylo checkpoint")
    ap.add_argument("--gnn_type", choices=["gcn", "sage"], default="gcn")
    ap.add_argument("--gnn_layers", type=int, default=3)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--gnn_dropout", type=float, default=0.1)

    # Sequence model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--seq_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--seq_dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=800)

    # CLIP training
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--B", type=int, default=64, help="species per batch")
    ap.add_argument("--K", type=int, default=2, help="seqs per species")
    ap.add_argument("--steps_per_epoch", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--rc_prob", type=float, default=0.5)

    # Triplet regularizer (only used in joint_clip_triplet)
    ap.add_argument("--lambda_triplet", type=float, default=1.0)
    ap.add_argument("--triplet_margin", type=float, default=0.1)
    ap.add_argument("--batch_anchors", type=int, default=256)
    ap.add_argument("--pos_max_hops", type=int, default=30)
    ap.add_argument("--pos_sampling", choices=["within", "uniform_hop"], default="uniform_hop")
    ap.add_argument("--pos_max_hop_resamples", type=int, default=30)
    ap.add_argument("--neg_min_hops", type=int, default=1)
    ap.add_argument("--neg_sampling", choices=["outside", "uniform_hop"], default="uniform_hop")
    ap.add_argument("--neg_max_hops", type=int, default=None)
    ap.add_argument("--neg_max_hop_resamples", type=int, default=50)

    # Optim
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # System
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # Logging / saving
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=500)  # periodic export only (no val set here)

    args = ap.parse_args()
    seed_all(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Mode:", args.mode)

    # ---- Load dataset (sequence side) ----
    dataset = s1_seq.JsonlSeqDataset(args.data_jsonl)
    species_sorted_seq = [dataset.id_to_species[i] for i in range(dataset.num_species)]
    print(f"Loaded sequences: {dataset.num_sequences} sequences, {dataset.num_species} species")

    sampler = s1_seq.SpeciesBalancedBatchSampler(
        indices_by_species=dataset.indices_by_species,
        B=args.B,
        K=args.K,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )

    def _collate(batch):
        return s1_seq.collate_batch(batch, max_len=args.max_len, rc_prob=args.rc_prob)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ---- Load phylogeny side data (graph + leaf map + leaf hops) ----
    edge_index = s1_phy.load_edge_index(args.data_dir / "edge_index.npz")
    leaf_map = s1_phy.read_leaf_map_tsv(args.data_dir / "leaf_map.tsv")
    species_sorted_phy = s1_phy.read_species_taxonomy_tsv(args.data_dir / "species_taxonomy.tsv")

    # Ensure consistent ordering across seq dataset and phy taxonomy
    if list(species_sorted_phy) != list(species_sorted_seq):
        # If mismatch, we can still proceed by remapping, but it is usually a pipeline inconsistency.
        raise RuntimeError(
            "species_sorted mismatch between sequences.jsonl and species_taxonomy.tsv.\n"
            "Re-run preprocessing and ensure both stage1 scripts used the same filtered dataset.\n"
            f"Example seq species: {species_sorted_seq[:5]}\n"
            f"Example phy species: {species_sorted_phy[:5]}"
        )
    species_sorted = species_sorted_seq

    # Graph tensors
    edge_index_t = torch.from_numpy(edge_index).to(device=device, dtype=torch.long)

    # edge_index: torch.LongTensor [2, E] on device
    num_nodes = int(edge_index_t.max().item()) + 1  # or pass num_nodes explicitly if you have it

    dst = edge_index_t[1]
    deg = torch.zeros((num_nodes,), device=edge_index_t.device, dtype=torch.float32)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg = torch.clamp(deg, min=1.0)
    deg_inv = 1.0 / deg

    # Leaf hop matrix for triplet regularizer (only needed for joint_clip_triplet)
    leaf_nodes = None
    hop_dist = None
    hop_sampler = None
    if args.mode == "joint_clip_triplet":
        leaf_nodes, hop_dist = s1_phy.load_leaf_hops(args.data_dir / "leaf_hops.npz")
        rng = random.Random(args.seed + 1234)
        hop_sampler = s1_phy.HopSampler(
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
        leaf_nodes = leaf_nodes.astype(np.int64)

    # Mapping from species_id -> graph node id (leaf)
    species_to_leaf_node = build_species_to_leaf_node_ids(species_sorted, leaf_map)
    species_to_leaf_node_t = torch.from_numpy(species_to_leaf_node).to(device=device, dtype=torch.long)

    # ---- Build models ----
    seq_model = SeqOnlyModel(
        d_model=args.d_model,
        emb_dim=args.emb_dim,
        n_layers=args.seq_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        dropout=args.seq_dropout,
        max_len=args.max_len,
    ).to(device)

    gnn_model = s1_phy.PhyloGNN(
        num_nodes=int(edge_index.max()) + 1 if edge_index.size > 0 else int(max(leaf_map.values())) + 1,
        emb_dim=args.emb_dim,
        n_layers=args.gnn_layers,
        dropout=args.gnn_dropout,
        gnn_type=args.gnn_type,
    ).to(device)

    # ---- Load pretrained phylo checkpoint weights into gnn_model ----
    ckpt_phy = torch_load_compat(args.phylo_ckpt)
    if not isinstance(ckpt_phy, dict) or "model" not in ckpt_phy:
        raise RuntimeError("phylo_ckpt does not look like a stage1_gnn_phylo checkpoint (missing 'model').")
    gnn_model.load_state_dict(ckpt_phy["model"], strict=True)

    # Fixed species embeddings for freeze_phylo_align (avoid running GNN every step)
    fixed_species_emb = None
    if args.mode == "freeze_phylo_align":
        if "species_emb_weight" in ckpt_phy:
            W = ckpt_phy["species_emb_weight"]
            fixed_species_emb = (W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else np.asarray(W))
            if fixed_species_emb.shape[0] != len(species_sorted) or fixed_species_emb.shape[1] != args.emb_dim:
                raise RuntimeError(
                    "phylo_ckpt species_emb_weight has unexpected shape. "
                    f"Got {fixed_species_emb.shape}, expected [{len(species_sorted)},{args.emb_dim}]"
                )
            fixed_species_emb = torch.from_numpy(fixed_species_emb).to(device=device, dtype=torch.float32)
        else:
            # Fallback: export from loaded GNN once
            with torch.no_grad():
                fixed_species_emb = export_species_emb_weight_from_gnn(
                    gnn_model, edge_index_t, deg_inv, species_sorted, leaf_map
                ).to(device=device, dtype=torch.float32)

        # Freeze GNN parameters
        for p in gnn_model.parameters():
            p.requires_grad_(False)
        gnn_model.eval()

    # For joint modes, allow training
    if args.mode in ("joint_clip", "joint_clip_triplet"):
        gnn_model.train()
        for p in gnn_model.parameters():
            p.requires_grad_(True)

    # ---- Optimizer ----
    params = list(seq_model.parameters())
    if args.mode in ("joint_clip", "joint_clip_triplet"):
        params += list(gnn_model.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    metrics_path = args.out_dir / "metrics.jsonl"
    step = 0
    t0 = time.time()

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        for tokens, species_ids in loader:
            tokens = tokens.to(device, non_blocking=True)
            species_ids = species_ids.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                # Sequence embeddings
                z_seq = seq_model(tokens)  # [B*K, D]

                # Phylo species embeddings for the SAME rows (species_ids aligned with tokens)
                if args.mode == "freeze_phylo_align":
                    z_phy = fixed_species_emb[species_ids]
                    clip, clip_m = clip_loss(z_seq, z_phy, temperature=args.temperature)
                    loss = clip
                    trip = torch.tensor(0.0, device=device)
                else:
                    # Run GNN forward (with grad), gather leaf embeddings per species id
                    X = gnn_model(edge_index_t, deg_inv)  # [N, D]
                    leaf_nodes_for_batch = species_to_leaf_node_t[species_ids]  # [B*K]
                    z_phy = X[leaf_nodes_for_batch]  # [B*K, D]

                    clip, clip_m = clip_loss(z_seq, z_phy, temperature=args.temperature)
                    loss = clip
                    trip = torch.tensor(0.0, device=device)

                    if args.mode == "joint_clip_triplet":
                        # Sample triplets on leaves and apply triplet regularizer to X
                        assert hop_sampler is not None and leaf_nodes is not None

                        L = int(hop_dist.shape[0])
                        # anchors in leaf-index space [0..L-1]
                        anchors_li = [hop_sampler.rng.randrange(L) for _ in range(args.batch_anchors)]

                        pos_li = []
                        neg_li = []
                        pos_hops = []

                        for ai in anchors_li:
                            pj, h_pos = hop_sampler.sample_positive(ai)
                            nj = hop_sampler.sample_negative(ai, h_pos=h_pos)
                            pos_li.append(pj)
                            neg_li.append(nj)
                            pos_hops.append(h_pos)

                        anchors_nodes = leaf_nodes[np.array(anchors_li, dtype=np.int64)]
                        pos_nodes = leaf_nodes[np.array(pos_li, dtype=np.int64)]
                        neg_nodes = leaf_nodes[np.array(neg_li, dtype=np.int64)]

                        a_t = torch.from_numpy(anchors_nodes).to(device=device, dtype=torch.long)
                        p_t = torch.from_numpy(pos_nodes).to(device=device, dtype=torch.long)
                        n_t = torch.from_numpy(neg_nodes).to(device=device, dtype=torch.long)

                        trip = s1_phy.triplet_margin_cosine(X[a_t], X[p_t], X[n_t], margin=args.triplet_margin)
                        loss = loss + (args.lambda_triplet * trip)

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, args.grad_clip)

            scaler.step(opt)
            scaler.update()

            step += 1

            if step % args.log_every == 0:
                it_s = step / max(1e-9, (time.time() - t0))
                rec = {
                    "type": "train",
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss.item()),
                    "clip_loss": float(clip.item()) if "clip" in locals() else float("nan"),
                    "triplet_loss": float(trip.item()) if isinstance(trip, torch.Tensor) else float(trip),
                    "acc@1": clip_m.get("acc@1", float("nan")),
                    "acc@5": clip_m.get("acc@5", float("nan")),
                    "it/s": float(it_s),
                    "mode": args.mode,
                    "temperature": args.temperature,
                    "lambda_triplet": args.lambda_triplet,
                    "triplet_margin": args.triplet_margin,
                    "gnn_type": args.gnn_type,
                    "gnn_layers": args.gnn_layers,
                    "seq_layers": args.seq_layers,
                }
                print(
                    f"epoch={epoch} step={step} loss={rec['loss']:.4f} "
                    f"clip={rec['clip_loss']:.4f} trip={rec['triplet_loss']:.4f} "
                    f"acc1={rec['acc@1']:.3f} acc5={rec['acc@5']:.3f} it/s={rec['it/s']:.2f}"
                )
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

            # periodic export + checkpoint (for evaluation.py compatibility)
            if (step % args.eval_every == 0) or (step % args.save_every == 0):
                with torch.no_grad():
                    W_sp = export_species_emb_weight_from_gnn(
                        gnn_model if args.mode != "freeze_phylo_align" else gnn_model,
                        edge_index_t, deg_inv, species_sorted, leaf_map
                    )
                    # If phylo is frozen, W_sp will just be the frozen phylo embedding (still fine).
                ckpt_path = args.out_dir / "checkpoints" / f"ckpt_step{step:07d}.pt"
                save_checkpoint(
                    ckpt_path,
                    seq_model=seq_model,
                    gnn_model=gnn_model,
                    optimizer=opt,
                    step=step,
                    epoch=epoch,
                    args=args,
                    species_sorted=species_sorted,
                    species_emb_weight=W_sp,
                )
                print("Saved:", ckpt_path)

    # final checkpoint
    with torch.no_grad():
        W_sp = export_species_emb_weight_from_gnn(gnn_model, edge_index_t, deg_inv, species_sorted, leaf_map)
    ckpt_path = args.out_dir / "checkpoints" / f"ckpt_final_step{step:07d}.pt"
    save_checkpoint(
        ckpt_path,
        seq_model=seq_model,
        gnn_model=gnn_model,
        optimizer=opt,
        step=step,
        epoch=args.epochs,
        args=args,
        species_sorted=species_sorted,
        species_emb_weight=W_sp,
    )
    print("Done. Final checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
