#!/usr/bin/env python3
"""
Torch-free preprocessing for Mare-MAGE COI + Fish Tree of Life Newick,
with detailed drop examples AND taxonomy/rank retention for downstream evaluation.

OUTPUTS:
  - pruned_tree.newick
  - edge_index.npz, edge_attr.npz
  - nodes.tsv, leaf_map.tsv
  - sequences.jsonl  (includes taxonomy, order, family, genus_raw, species_raw)
  - species_taxonomy.tsv (one row per species with majority-vote order/family/genus + seq_count)
  - report.txt
  - drop_examples.txt
  - iTOL colorstrip datasets (IDs match EXACT tree tip labels; works with trinomials):
      itol_order_colorstrip.txt   (legend shows top-10 orders)
      itol_family_colorstrip.txt  (legend shows top-10 families)

Requires:
  pip install biopython networkx numpy
"""

import argparse
import json
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import networkx as nx
from Bio import Phylo, SeqIO
from Bio.Phylo.BaseTree import Clade


BAD_TOKENS = {"sp", "sp.", "spp", "spp.", "cf", "cf.", "aff", "aff.", "nr", "nr."}


# ---------------------------
# Helpers: drop example recorder
# ---------------------------

class DropRecorder:
    def __init__(self, max_examples: int = 20):
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.max_examples = max_examples

    def add(self, reason: str, example: str):
        self.counts[reason] += 1
        if len(self.examples[reason]) < self.max_examples:
            self.examples[reason].append(example)

    def add_many(self, reason: str, examples: List[str]):
        self.counts[reason] += len(examples)
        space = self.max_examples - len(self.examples[reason])
        if space > 0:
            self.examples[reason].extend(examples[:space])

    def most_common(self):
        return self.counts.most_common()


# ---------------------------
# Taxonomy parsing
# ---------------------------

def parse_tax_ranks(tax: str) -> dict:
    """
    Extract raw ranks from Mare-MAGE taxonomy string.
    We use:
      O_...  for order
      F_...  for family
      G_...  for genus
      s_...  for species
    """
    out = {"order": None, "family": None, "genus_raw": None, "species_raw": None}
    for part in tax.split(";"):
        part = part.strip()
        if part.startswith("O_"):
            out["order"] = part[2:]
        elif part.startswith("F_"):
            out["family"] = part[2:]
        elif part.startswith("G_"):
            out["genus_raw"] = part[2:]
        elif part.startswith("s_"):
            out["species_raw"] = part[2:]
    return out


# ---------------------------
# Species name handling
# ---------------------------

def extract_species_from_taxonomy(tax: str) -> Optional[str]:
    m = re.search(r"(?:^|;)s_([^;]+)", tax)
    return m.group(1).strip() if m else None


def normalize_species_name(raw: str, collapse_subspecies: bool = True) -> Optional[str]:
    """
    Normalizes a raw species string to Genus_species (underscore).
    If collapse_subspecies=True (recommended), uses only first two tokens.
    """
    raw = raw.strip().strip("'").strip('"').replace("_", " ")
    raw = re.sub(r"\s+", " ", raw)
    parts = raw.split()

    if len(parts) < 2:
        return None

    genus = re.sub(r"[^A-Za-z\-]", "", parts[0])
    species = re.sub(r"[^A-Za-z\-]", "", parts[1])

    if not genus or not species:
        return None
    if genus.lower() in BAD_TOKENS or species.lower() in BAD_TOKENS:
        return None

    genus = genus.capitalize()
    species = species.lower()

    # NOTE: collapse_subspecies True means always binomial. If False, keep trinomial.
    if not collapse_subspecies and len(parts) >= 3:
        sub = re.sub(r"[^A-Za-z\-]", "", parts[2]).lower()
        if sub and sub not in BAD_TOKENS:
            return f"{genus}_{species}_{sub}"

    return f"{genus}_{species}"


def normalize_tree_tip(name: str, collapse_subspecies: bool = True) -> Optional[str]:
    """
    Normalize a tree tip to Genus_species (binomial) by default.
    Tree may contain trinomials; we intentionally collapse for matching.
    """
    if not name:
        return None
    name = name.strip().strip("'").strip('"').replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    parts = name.split()
    if len(parts) < 2:
        return None
    raw = f"{parts[0]} {parts[1]}"
    return normalize_species_name(raw, collapse_subspecies=collapse_subspecies)


# ---------------------------
# Reporting utilities
# ---------------------------

def count_species_from_seq_map(seq_to_species: Dict[str, str]) -> int:
    return len(set(seq_to_species.values()))


def log_step(lines: list, step: str, n_seq: int, n_sp: int, extra: Optional[str] = None) -> None:
    msg = f"{step:<38} sequences={n_seq:<8} species={n_sp:<8}"
    if extra:
        msg += f" | {extra}"
    lines.append(msg)


# ---------------------------
# iTOL helpers (Colorstrip + legend)
# ---------------------------

def _hex_color_for_category(cat: str) -> str:
    cat = (cat or "Unknown").strip()
    h = hashlib.md5(cat.encode("utf-8")).hexdigest()
    return f"#{h[:6]}"


def write_itol_colorstrip_with_legend(
    id_to_group: Dict[str, str],
    outpath: Path,
    dataset_label: str,
    top_legend_groups: List[str],
) -> None:
    """
    iTOL DATASET_COLORSTRIP:
      <ID> <COLOR> <optional label>

    Adds a legend containing ONLY `top_legend_groups` (e.g. top 10 orders/families).
    IDs must match tree leaf labels EXACTLY (including spaces/trinomials).
    """
    lines = []
    lines.append("DATASET_COLORSTRIP")
    lines.append("SEPARATOR\tTAB")
    lines.append(f"DATASET_LABEL\t{dataset_label}")
    lines.append("COLOR\t#000000")
    lines.append(f"LEGEND_TITLE\t{dataset_label}")
    lines.append("COLOR_BRANCHES\t0")

    if top_legend_groups:
        # Legend entries (squares)
        legend_colors = [_hex_color_for_category(g) for g in top_legend_groups]
        legend_shapes = ["1"] * len(top_legend_groups)  # square
        # For fields with multiple values, iTOL expects space-separated lists even with TAB separator.
        lines.append("LEGEND_SHAPES\t" + " ".join(legend_shapes))
        lines.append("LEGEND_COLORS\t" + " ".join(legend_colors))
        lines.append("LEGEND_LABELS\t" + " ".join([g.replace(" ", "_") for g in top_legend_groups]))

    lines.append("DATA")

    for node_id, grp in sorted(id_to_group.items(), key=lambda x: x[0]):
        grp = grp if grp and grp.strip() else "Unknown"
        col = _hex_color_for_category(grp)
        # ID \t color \t label
        lines.append(f"{node_id}\t{col}\t{grp}")

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tipname_to_group(tree, species_to_group: Dict[str, str]) -> Dict[str, str]:
    """
    Map *actual tree tip names* -> group, by normalizing tip to binomial for lookup.

    This fixes iTOL errors when the tree contains trinomials like:
      "Aethotaxis mitopteryx mitopteryx"
    while our species keys are binomials:
      "Aethotaxis_mitopteryx"
    """
    tip_to_group: Dict[str, str] = {}
    for tip in tree.get_terminals():
        if not tip.name:
            continue
        norm = normalize_tree_tip(tip.name, collapse_subspecies=True)
        if not norm:
            continue
        tip_to_group[tip.name] = species_to_group.get(norm, "Unknown")
    return tip_to_group


# ---------------------------
# Tree utilities
# ---------------------------

def iter_terminals_with_norm(tree, collapse_subspecies: bool) -> Dict[Clade, str]:
    out = {}
    for tip in tree.get_terminals():
        norm = normalize_tree_tip(tip.name or "", collapse_subspecies=collapse_subspecies)
        if norm:
            out[tip] = norm
    return out


def build_graph_from_tree(tree) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], list]:
    """
    Build an undirected graph from the (pruned) tree.
    Leaf map keys are binomial Genus_species (normalized). Trinomials collapse to binomials.
    """
    G = nx.Graph()
    clade_to_id: Dict[Clade, int] = {}
    next_id = 0

    def get_id(clade: Clade) -> int:
        nonlocal next_id
        if clade not in clade_to_id:
            clade_to_id[clade] = next_id
            next_id += 1
        return clade_to_id[clade]

    def walk(parent: Clade):
        pid = get_id(parent)
        for child in parent.clades:
            cid = get_id(child)
            bl = child.branch_length
            bl = float(bl) if bl is not None else 1.0
            G.add_edge(pid, cid, branch_length=bl)
            walk(child)

    walk(tree.root)

    node_rows = []
    leaf_map: Dict[str, int] = {}
    for clade, nid in clade_to_id.items():
        is_tip = 1 if len(clade.clades) == 0 else 0
        name = ""
        if is_tip:
            name = normalize_tree_tip(clade.name or "", collapse_subspecies=True) or ""
            if name:
                # If multiple tips collapse to the same binomial, last one wins.
                leaf_map[name] = nid
        node_rows.append((nid, is_tip, name))

    edges = list(G.edges(data=True))
    src, dst, bls = [], [], []
    for u, v, data in edges:
        bl = float(data.get("branch_length", 1.0))
        src.extend([u, v])
        dst.extend([v, u])
        bls.extend([bl, bl])

    edge_index = np.vstack([src, dst]).astype(np.int64)
    edge_attr = np.array(bls, dtype=np.float32).reshape(-1, 1)
    return edge_index, edge_attr, leaf_map, node_rows


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--collapse_subspecies", action="store_true")
    ap.add_argument("--min_len", type=int, default=200)
    ap.add_argument("--max_N_frac", type=float, default=0.05)
    ap.add_argument("--require_min_seqs_per_species", type=int, default=2)
    ap.add_argument("--max_examples", type=int, default=20, help="Max examples per drop reason")

    # iTOL legend sizes
    ap.add_argument("--itol_legend_top", type=int, default=10, help="Top-N groups to show in iTOL legend")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    drops = DropRecorder(max_examples=args.max_examples)

    # -----------------------
    # Step 1: Parse taxonomy
    # -----------------------
    raw_rows = 0
    seq_to_species_raw: Dict[str, str] = {}
    seq_to_taxonomy_raw: Dict[str, str] = {}
    seq_to_ranks_raw: Dict[str, dict] = {}

    with open(args.taxonomy, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_rows += 1
            parts = line.split("\t")
            if len(parts) < 2:
                drops.add("taxonomy_bad_row_format", line[:200])
                continue

            seq_id, tax = parts[0].strip(), parts[1].strip()
            raw_sp = extract_species_from_taxonomy(tax)
            if raw_sp is None:
                drops.add("taxonomy_missing_s_field", seq_id)
                continue

            sp = normalize_species_name(raw_sp, collapse_subspecies=args.collapse_subspecies)
            if sp is None:
                drops.add("taxonomy_unusable_species_name", raw_sp)
                continue

            seq_to_species_raw[seq_id] = sp
            seq_to_taxonomy_raw[seq_id] = tax
            seq_to_ranks_raw[seq_id] = parse_tax_ranks(tax)

    log_step(
        report_lines,
        "After taxonomy parse",
        len(seq_to_species_raw),
        count_species_from_seq_map(seq_to_species_raw),
        extra=f"input_rows={raw_rows} drops={sum(drops.counts.values())}",
    )

    # --------------------------------------
    # Step 2: Load FASTA (match seq IDs)
    # --------------------------------------
    fasta_seen = 0
    seqs_raw: Dict[str, str] = {}
    for rec in SeqIO.parse(args.fasta, "fasta"):
        fasta_seen += 1
        sid = rec.id
        if sid not in seq_to_species_raw:
            drops.add("fasta_id_not_in_taxonomy", sid)
            continue
        seq = str(rec.seq).upper()
        seq = re.sub(r"[^ACGTN]", "N", seq)
        seqs_raw[sid] = seq

    missing_in_fasta = sorted(set(seq_to_species_raw.keys()) - set(seqs_raw.keys()))
    drops.add_many("taxonomy_id_missing_in_fasta", missing_in_fasta)

    seq_to_species = {sid: sp for sid, sp in seq_to_species_raw.items() if sid in seqs_raw}
    seq_to_taxonomy = {sid: tx for sid, tx in seq_to_taxonomy_raw.items() if sid in seqs_raw}
    seq_to_ranks = {sid: rk for sid, rk in seq_to_ranks_raw.items() if sid in seqs_raw}

    log_step(
        report_lines,
        "After FASTA ID matching",
        len(seqs_raw),
        count_species_from_seq_map(seq_to_species),
        extra=f"fasta_records={fasta_seen} missing_tax_ids_in_fasta={len(missing_in_fasta)}",
    )

    # --------------------------------------
    # Step 3: Sequence filters
    # --------------------------------------
    seqs_filt: Dict[str, str] = {}
    for sid, seq in seqs_raw.items():
        if len(seq) < args.min_len:
            drops.add("seq_too_short", sid)
            continue
        n_frac = seq.count("N") / max(1, len(seq))
        if n_frac > args.max_N_frac:
            drops.add("seq_too_many_N", sid)
            continue
        seqs_filt[sid] = seq

    keep_ids = set(seqs_filt.keys())
    seq_to_species = {sid: sp for sid, sp in seq_to_species.items() if sid in keep_ids}
    seq_to_taxonomy = {sid: tx for sid, tx in seq_to_taxonomy.items() if sid in keep_ids}
    seq_to_ranks = {sid: rk for sid, rk in seq_to_ranks.items() if sid in keep_ids}

    log_step(
        report_lines,
        "After length/N filters",
        len(seqs_filt),
        count_species_from_seq_map(seq_to_species),
        extra=f"min_len={args.min_len} max_N_frac={args.max_N_frac}",
    )

    # --------------------------------------
    # Step 4: Species min sequences
    # --------------------------------------
    species_to_ids = defaultdict(list)
    for sid, sp in seq_to_species.items():
        species_to_ids[sp].append(sid)

    if args.require_min_seqs_per_species > 1:
        species_keep = {sp for sp, ids in species_to_ids.items() if len(ids) >= args.require_min_seqs_per_species}
        dropped_species = sorted([sp for sp in species_to_ids.keys() if sp not in species_keep])
        drops.add_many("species_dropped_for_low_seq_count", dropped_species)

        kept_seq_ids = {sid for sp in species_keep for sid in species_to_ids[sp]}
        dropped_seq_ids = sorted(set(seqs_filt.keys()) - kept_seq_ids)
        drops.add_many("seq_dropped_due_to_low_seq_species", dropped_seq_ids)

        seqs_filt = {sid: seq for sid, seq in seqs_filt.items() if sid in kept_seq_ids}
        seq_to_species = {sid: sp for sid, sp in seq_to_species.items() if sid in kept_seq_ids}
        seq_to_taxonomy = {sid: tx for sid, tx in seq_to_taxonomy.items() if sid in kept_seq_ids}
        seq_to_ranks = {sid: rk for sid, rk in seq_to_ranks.items() if sid in kept_seq_ids}
    else:
        species_keep = set(species_to_ids.keys())

    log_step(
        report_lines,
        "After min-seqs/species",
        len(seqs_filt),
        len(species_keep),
        extra=f"min_seqs/species={args.require_min_seqs_per_species}",
    )

    # -----------------------
    # Step 5: Load Fish Tree
    # -----------------------
    tree = Phylo.read(args.tree, "newick")
    tree_tips_before = tree.count_terminals()
    tip_norm_map = iter_terminals_with_norm(tree, collapse_subspecies=True)
    tree_species_set = set(tip_norm_map.values())
    log_step(
        report_lines,
        "Fish tree loaded",
        0,
        0,
        extra=f"tree_tips={tree_tips_before} normalized_tip_species={len(tree_species_set)}",
    )

    # --------------------------------------
    # Step 6: Species ∩ tree tips
    # --------------------------------------
    intersect_species = species_keep & tree_species_set
    missing_species = sorted(list(species_keep - tree_species_set))
    drops.add_many("species_missing_in_tree_tips", missing_species)

    log_step(
        report_lines,
        "Species ∩ tree tips",
        0,
        len(intersect_species),
        extra=f"species_keep={len(species_keep)} tree_species={len(tree_species_set)}",
    )

    # Prune
    pruned_attempts = 0
    for tip, norm in list(tip_norm_map.items()):
        if norm not in intersect_species:
            try:
                tree.prune(tip)
                pruned_attempts += 1
            except Exception:
                pass

    tree_tips_after = tree.count_terminals()
    pruned_tree_path = out_dir / "pruned_tree.newick"
    Phylo.write(tree, pruned_tree_path, "newick")

    log_step(
        report_lines,
        "After pruning tree",
        0,
        0,
        extra=f"tips_before={tree_tips_before} tips_after={tree_tips_after} pruned={pruned_attempts}",
    )

    # --------------------------------------
    # Step 7: Build graph
    # --------------------------------------
    edge_index, edge_attr, leaf_map, node_rows = build_graph_from_tree(tree)
    log_step(
        report_lines,
        "After graph build",
        0,
        0,
        extra=f"graph_nodes={len(node_rows)} directed_edges={edge_index.shape[1]} leaf_species={len(leaf_map)}",
    )

    # --------------------------------------
    # Step 8: Filter sequences to tree leaf map
    # --------------------------------------
    leaf_species = set(leaf_map.keys())

    seq_ids_before = set(seqs_filt.keys())
    seq_ids_after = {sid for sid, sp in seq_to_species.items() if sp in leaf_species}

    dropped_seq_ids = sorted(list(seq_ids_before - seq_ids_after))
    drops.add_many("seq_dropped_missing_in_tree_leaf_map", dropped_seq_ids)

    still_missing_species = sorted(list(set(seq_to_species.values()) - leaf_species))
    drops.add_many("species_missing_in_leaf_map_after_prune", still_missing_species)

    seqs_final = {sid: seqs_filt[sid] for sid in seq_ids_after}
    seq_to_species_final = {sid: seq_to_species[sid] for sid in seq_ids_after}
    seq_to_taxonomy_final = {sid: seq_to_taxonomy[sid] for sid in seq_ids_after}
    seq_to_ranks_final = {sid: seq_to_ranks[sid] for sid in seq_ids_after}
    final_species = set(seq_to_species_final.values())

    log_step(
        report_lines,
        "Final (mapped to tree leaves)",
        len(seqs_final),
        len(final_species),
        extra=f"leaf_species={len(leaf_species)}",
    )

    # --------------------------------------
    # Write outputs
    # --------------------------------------
    np.savez(out_dir / "edge_index.npz", edge_index=edge_index)
    np.savez(out_dir / "edge_attr.npz", edge_attr=edge_attr)

    with open(out_dir / "nodes.tsv", "w", encoding="utf-8") as f:
        f.write("node_id\tis_tip\tname\n")
        for nid, is_tip, name in sorted(node_rows, key=lambda x: x[0]):
            f.write(f"{nid}\t{is_tip}\t{name}\n")

    with open(out_dir / "leaf_map.tsv", "w", encoding="utf-8") as f:
        f.write("species\tnode_id\n")
        for sp, nid in sorted(leaf_map.items(), key=lambda x: x[0]):
            f.write(f"{sp}\t{nid}\n")

    with open(out_dir / "sequences.jsonl", "w", encoding="utf-8") as f:
        for sid, sp in seq_to_species_final.items():
            rk = seq_to_ranks_final.get(sid, {})
            f.write(json.dumps({
                "seq_id": sid,
                "species": sp,
                "node_id": int(leaf_map[sp]),
                "sequence": seqs_final[sid],
                "taxonomy": seq_to_taxonomy_final.get(sid),
                "order": rk.get("order"),
                "family": rk.get("family"),
                "genus_raw": rk.get("genus_raw"),
                "species_raw": rk.get("species_raw"),
            }) + "\n")

    # species_taxonomy.tsv majority-vote ranks per species
    species_rank_counts = defaultdict(lambda: {
        "order": Counter(),
        "family": Counter(),
        "genus_raw": Counter(),
    })
    species_seq_counts = Counter()

    for sid, sp in seq_to_species_final.items():
        rk = seq_to_ranks_final.get(sid, {})
        species_seq_counts[sp] += 1
        if rk.get("order"):
            species_rank_counts[sp]["order"][rk["order"]] += 1
        if rk.get("family"):
            species_rank_counts[sp]["family"][rk["family"]] += 1
        if rk.get("genus_raw"):
            species_rank_counts[sp]["genus_raw"][rk["genus_raw"]] += 1

    def majority(counter: Counter) -> Tuple[str, int]:
        if not counter:
            return ("Unknown", 0)
        val, cnt = counter.most_common(1)[0]
        return (val, cnt)

    with open(out_dir / "species_taxonomy.tsv", "w", encoding="utf-8") as f:
        f.write("species\tseq_count\torder\tfamily\tgenus_raw\n")
        for sp in sorted(final_species):
            seq_count = species_seq_counts[sp]
            ord_val, _ = majority(species_rank_counts[sp]["order"])
            fam_val, _ = majority(species_rank_counts[sp]["family"])
            gen_val, _ = majority(species_rank_counts[sp]["genus_raw"])
            f.write(f"{sp}\t{seq_count}\t{ord_val}\t{fam_val}\t{gen_val}\n")

    # --------------------------------------
    # iTOL Color Strip datasets (Order/Family) with top-N legend
    # --------------------------------------
    species_to_order = {}
    species_to_family = {}

    for sp in sorted(final_species):
        if sp not in leaf_species:
            continue
        ord_val, _ = majority(species_rank_counts[sp]["order"])
        fam_val, _ = majority(species_rank_counts[sp]["family"])
        species_to_order[sp] = ord_val or "Unknown"
        species_to_family[sp] = fam_val or "Unknown"

    tip_to_order = build_tipname_to_group(tree, species_to_order)
    tip_to_family = build_tipname_to_group(tree, species_to_family)

    # Determine top-N legend entries by number of tips in the pruned tree
    order_counts = Counter(tip_to_order.values())
    family_counts = Counter(tip_to_family.values())
    top_orders = [g for g, _ in order_counts.most_common(args.itol_legend_top)]
    top_families = [g for g, _ in family_counts.most_common(args.itol_legend_top)]

    write_itol_colorstrip_with_legend(
        tip_to_order,
        out_dir / "itol_order_colorstrip.txt",
        "Order",
        top_orders
    )
    write_itol_colorstrip_with_legend(
        tip_to_family,
        out_dir / "itol_family_colorstrip.txt",
        "Family",
        top_families
    )

    # report.txt
    summary = []
    summary.extend(report_lines)
    summary.append("\nDrop reasons (counts):")
    for k, v in drops.most_common():
        summary.append(f"  {k}: {v}")
    summary.append("\nKey output files:")
    summary.append(f"  pruned_tree.newick:            {pruned_tree_path}")
    summary.append(f"  edge_index.npz:                {out_dir / 'edge_index.npz'}")
    summary.append(f"  edge_attr.npz:                 {out_dir / 'edge_attr.npz'}")
    summary.append(f"  sequences.jsonl:               {out_dir / 'sequences.jsonl'}")
    summary.append(f"  species_taxonomy.tsv:          {out_dir / 'species_taxonomy.tsv'}")
    summary.append(f"  leaf_map.tsv:                  {out_dir / 'leaf_map.tsv'}")
    summary.append(f"  nodes.tsv:                     {out_dir / 'nodes.tsv'}")
    summary.append(f"  itol_order_colorstrip.txt:     {out_dir / 'itol_order_colorstrip.txt'}")
    summary.append(f"  itol_family_colorstrip.txt:    {out_dir / 'itol_family_colorstrip.txt'}")
    summary.append(f"\nTop-{args.itol_legend_top} legend orders: " + ", ".join(top_orders))
    summary.append(f"Top-{args.itol_legend_top} legend families: " + ", ".join(top_families))

    (out_dir / "report.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    # drop_examples.txt
    ex_lines = []
    ex_lines.append(f"Showing up to first {args.max_examples} examples per drop reason.\n")
    for reason, cnt in drops.most_common():
        ex_lines.append(f"{reason} (count={cnt})")
        for ex in drops.examples.get(reason, []):
            ex_lines.append(f"  - {ex}")
        ex_lines.append("")
    (out_dir / "drop_examples.txt").write_text("\n".join(ex_lines), encoding="utf-8")

    print("Done. Wrote outputs to:", out_dir)


if __name__ == "__main__":
    main()
