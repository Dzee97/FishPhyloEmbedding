#!/usr/bin/env python3
"""
Baseline Stage-1 training script (Python-only training; uses Torch).

Goal:
- Learn a sequence encoder (Transformer) that maps raw COI sequences to an embedding z_seq
- Learn per-species "anchor" embeddings t_species (nn.Embedding)
- Train with a CLIP/InfoNCE contrastive loss: z_seq should match the anchor of its species

Inputs (from your preprocessing):
- sequences.jsonl  (each line: {"seq_id","species","node_id","sequence"})
  NOTE: we do NOT use node_id for this baseline; we map species -> contiguous integer ID.

Outputs:
- checkpoints/*.pt
- metrics.jsonl

Recommended run (RTX 4060 8GB):
  python train_baseline_clip.py \
    --data_jsonl output/sequences.jsonl \
    --out_dir runs/baseline1 \
    --epochs 10 \
    --d_model 256 --n_layers 4 --n_heads 4 \
    --max_len 800 \
    --B 64 --K 2 \
    --amp

Dependencies:
  pip install torch numpy
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler


# -----------------------------
# Reproducibility
# -----------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# DNA utilities
# -----------------------------

VOCAB = {
    "PAD": 0,
    "CLS": 1,
    "A": 2,
    "C": 3,
    "G": 4,
    "T": 5,
    "N": 6,
}
ID2TOK = {v: k for k, v in VOCAB.items()}

_RC_MAP = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"})


def reverse_complement(seq: str) -> str:
    return seq.translate(_RC_MAP)[::-1]


def tokenize(seq: str, max_len: int) -> np.ndarray:
    """
    Returns int32 tokens of length max_len:
      [CLS] + seq[:max_len-1], padded with PAD
    """
    seq = seq[: max_len - 1]
    arr = np.full((max_len,), VOCAB["PAD"], dtype=np.int32)
    arr[0] = VOCAB["CLS"]
    for i, ch in enumerate(seq, start=1):
        arr[i] = VOCAB.get(ch, VOCAB["N"])
    return arr


# -----------------------------
# Dataset
# -----------------------------

@dataclass
class Example:
    seq_id: str
    species: str
    species_id: int
    sequence: str


class JsonlSeqDataset(Dataset):
    """
    Loads sequences.jsonl into memory (strings + integer IDs).
    Stores indices per species for balanced sampling.
    """

    def __init__(self, jsonl_path: Path):
        self.jsonl_path = Path(jsonl_path)
        self.examples: List[Example] = []

        # First pass: read all, build species vocab
        species_set = []
        raw_rows = 0
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw_rows += 1
                obj = json.loads(line)
                sp = obj["species"]
                species_set.append(sp)

        # Stable mapping
        species_sorted = sorted(set(species_set))
        self.species_to_id = {sp: i for i, sp in enumerate(species_sorted)}
        self.id_to_species = {i: sp for sp, i in self.species_to_id.items()}

        # Second pass: store examples + per-species index
        self.indices_by_species: List[List[int]] = [[] for _ in range(len(self.species_to_id))]
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                sp = obj["species"]
                sid = self.species_to_id[sp]
                ex = Example(
                    seq_id=obj.get("seq_id", ""),
                    species=sp,
                    species_id=sid,
                    sequence=obj["sequence"].upper(),
                )
                idx = len(self.examples)
                self.examples.append(ex)
                self.indices_by_species[sid].append(idx)

        # Basic stats
        self.num_species = len(self.species_to_id)
        self.num_sequences = len(self.examples)
        self.seqs_per_species = np.array([len(v) for v in self.indices_by_species], dtype=np.int32)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


class SpeciesBalancedBatchSampler(Sampler[List[int]]):
    """
    Each batch: sample B species, and K sequences per species => batch size = B*K
    Returns list of dataset indices.

    Note: This sampler is stochastic and infinite-ish per epoch; we use steps_per_epoch to stop.
    """

    def __init__(
        self,
        indices_by_species: List[List[int]],
        B: int,
        K: int,
        steps_per_epoch: int,
        seed: int = 0,
    ):
        self.indices_by_species = indices_by_species
        self.B = B
        self.K = K
        self.steps_per_epoch = steps_per_epoch
        self.rng = random.Random(seed)

        # Species that can provide at least 1 sequence (they should all, but be safe)
        self.valid_species = [i for i, idxs in enumerate(indices_by_species) if len(idxs) > 0]

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            species_batch = self.rng.sample(self.valid_species, k=min(self.B, len(self.valid_species)))
            batch_indices = []
            for sid in species_batch:
                idxs = self.indices_by_species[sid]
                if len(idxs) >= self.K:
                    chosen = self.rng.sample(idxs, k=self.K)
                else:
                    # sample with replacement if needed
                    chosen = [self.rng.choice(idxs) for _ in range(self.K)]
                batch_indices.extend(chosen)
            yield batch_indices


def collate_batch(
    batch: List[Example],
    max_len: int,
    rc_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      tokens: LongTensor [B, max_len]
      species_ids: LongTensor [B]
    """
    tokens = np.zeros((len(batch), max_len), dtype=np.int32)
    species_ids = np.zeros((len(batch),), dtype=np.int64)

    for i, ex in enumerate(batch):
        seq = ex.sequence
        if rc_prob > 0 and random.random() < rc_prob:
            seq = reverse_complement(seq)
        tokens[i] = tokenize(seq, max_len=max_len)
        species_ids[i] = ex.species_id

    return torch.from_numpy(tokens).long(), torch.from_numpy(species_ids).long()


# -----------------------------
# Model
# -----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class SequenceTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        max_len: int,
        out_dim: int,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=VOCAB["PAD"])
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, L] Long
        returns: [B, out_dim] float
        """
        x = self.token_emb(tokens)          # [B, L, D]
        x = self.pos_enc(x)                 # [B, L, D]
        # Key padding mask: True where PAD
        pad_mask = tokens.eq(VOCAB["PAD"])  # [B, L]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        cls = x[:, 0, :]                    # [B, D]
        z = self.proj(cls)                  # [B, out_dim]
        return z


class BaselineCLIPModel(nn.Module):
    def __init__(
        self,
        num_species: int,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        max_len: int,
        emb_dim: int,
    ):
        super().__init__()
        self.seq_encoder = SequenceTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            max_len=max_len,
            out_dim=emb_dim,
        )
        self.species_emb = nn.Embedding(num_species, emb_dim)
        nn.init.normal_(self.species_emb.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, species_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_seq = self.seq_encoder(tokens)                 # [B, d]
        z_sp = self.species_emb(species_ids)             # [B, d]
        return z_seq, z_sp


# -----------------------------
# Loss + metrics
# -----------------------------

def clip_loss(z_seq: torch.Tensor, z_sp: torch.Tensor, temperature: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Symmetric CLIP loss:
      logits = z_seq @ z_sp^T / T
      CE over diagonal for seq->sp and sp->seq
    """
    z_seq = F.normalize(z_seq, dim=-1)
    z_sp = F.normalize(z_sp, dim=-1)

    logits = (z_seq @ z_sp.T) / temperature  # [B, B]
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.T, targets)
    loss = 0.5 * (loss_a + loss_b)

    with torch.no_grad():
        # retrieval accuracy (top-1)
        pred = logits.argmax(dim=1)
        acc = (pred == targets).float().mean().item()
        # also top-5 if batch is large enough
        k = min(5, logits.size(1))
        topk = logits.topk(k=k, dim=1).indices
        acc5 = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    return loss, {"acc@1": acc, "acc@5": acc5}


@torch.no_grad()
def quick_eval_species_retrieval(
    model: BaselineCLIPModel,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    max_batches: int = 50,
) -> Dict[str, float]:
    """
    Evaluates CLIP retrieval on a few batches (fast sanity check).
    """
    model.eval()
    acc1s, acc5s, losses = [], [], []
    for i, (tokens, species_ids) in enumerate(loader):
        if i >= max_batches:
            break
        tokens = tokens.to(device, non_blocking=True)
        species_ids = species_ids.to(device, non_blocking=True)

        z_seq, z_sp = model(tokens, species_ids)
        loss, m = clip_loss(z_seq, z_sp, temperature=temperature)
        losses.append(loss.item())
        acc1s.append(m["acc@1"])
        acc5s.append(m["acc@5"])

    model.train()
    return {
        "eval_loss": float(np.mean(losses)) if losses else float("nan"),
        "eval_acc@1": float(np.mean(acc1s)) if acc1s else float("nan"),
        "eval_acc@5": float(np.mean(acc5s)) if acc5s else float("nan"),
        "eval_batches": min(max_batches, i + 1),
    }


# -----------------------------
# Training loop
# -----------------------------

def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, epoch: int, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "args": vars(args),
        },
        str(path),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", required=True, type=str, help="Path to sequences.jsonl from preprocessing")
    ap.add_argument("--out_dir", required=True, type=str)

    # Model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=800)

    # Training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--B", type=int, default=64, help="Species per batch")
    ap.add_argument("--K", type=int, default=2, help="Sequences per species per batch")
    ap.add_argument("--steps_per_epoch", type=int, default=1000, help="Batches per epoch (stochastic sampler)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--rc_prob", type=float, default=0.5, help="Reverse-complement augmentation probability")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # AMP / device
    ap.add_argument("--amp", action="store_true", help="Use mixed precision AMP")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
    ap.add_argument("--eval_every", type=int, default=500, help="Run quick eval every N steps")
    args = ap.parse_args()

    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data
    dataset = JsonlSeqDataset(Path(args.data_jsonl))
    print(f"Loaded dataset: {dataset.num_sequences} sequences, {dataset.num_species} species")

    # Sampler (balanced species batches)
    sampler = SpeciesBalancedBatchSampler(
        indices_by_species=dataset.indices_by_species,
        B=args.B,
        K=args.K,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )

    # DataLoader
    def _collate(batch):
        return collate_batch(batch, max_len=args.max_len, rc_prob=args.rc_prob)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Build model
    model = BaselineCLIPModel(
        num_species=dataset.num_species,
        vocab_size=len(VOCAB),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_len=args.max_len,
        emb_dim=args.emb_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    metrics_path = out_dir / "metrics.jsonl"
    step = 0

    # Initial quick eval sanity (on a few batches)
    eval_loader = DataLoader(
        dataset,
        batch_sampler=SpeciesBalancedBatchSampler(
            dataset.indices_by_species, B=args.B, K=args.K, steps_per_epoch=100, seed=args.seed + 123
        ),
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True,
        collate_fn=_collate,
        persistent_workers=False,
    )
    init_eval = quick_eval_species_retrieval(model, eval_loader, device, temperature=args.temperature, max_batches=20)
    print("Initial eval:", init_eval)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"step": step, "epoch": 0, "type": "eval", **init_eval}) + "\n")

    # Train
    print("Starting training...")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for tokens, species_ids in loader:
            tokens = tokens.to(device, non_blocking=True)
            species_ids = species_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                z_seq, z_sp = model(tokens, species_ids)
                loss, m = clip_loss(z_seq, z_sp, temperature=args.temperature)

            scaler.scale(loss).backward()

            # Grad clip (unscale first)
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            step += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                it_s = step / max(1e-9, elapsed)
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} acc1={m['acc@1']:.3f} acc5={m['acc@5']:.3f} it/s={it_s:.2f}"
                )

            # Write train metrics occasionally
            if step % 50 == 0:
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step": step,
                        "epoch": epoch,
                        "type": "train",
                        "loss": float(loss.item()),
                        **m,
                        "B": args.B,
                        "K": args.K,
                        "max_len": args.max_len,
                    }) + "\n")

            # Quick eval
            if args.eval_every > 0 and step % args.eval_every == 0:
                ev = quick_eval_species_retrieval(model, eval_loader, device, temperature=args.temperature, max_batches=30)
                print("Eval:", ev)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "epoch": epoch, "type": "eval", **ev}) + "\n")

            # Save checkpoint
            if args.save_every > 0 and step % args.save_every == 0:
                ckpt_path = out_dir / "checkpoints" / f"ckpt_step{step}.pt"
                save_checkpoint(ckpt_path, model, optimizer, step=step, epoch=epoch, args=args)
                print("Saved:", ckpt_path)

        # End of epoch checkpoint
        ckpt_path = out_dir / "checkpoints" / f"ckpt_epoch{epoch}.pt"
        save_checkpoint(ckpt_path, model, optimizer, step=step, epoch=epoch, args=args)
        print("Saved:", ckpt_path)

    print("Done.")


if __name__ == "__main__":
    main()

