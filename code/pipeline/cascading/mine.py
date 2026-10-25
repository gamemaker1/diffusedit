"""Cascade pair mining from silver training data — Section 8.5.

Builds training pairs for the consequence prediction head (Section 6.4).
Mining procedure (separate from the evaluation set, which uses gold articles):

  For each article in silver train:
    1. Identify direct edits: sentences whose changed mentions overlap evidence.
    2. Identify changed-but-not-direct sentences (cascade candidates).
    3. For each (direct j, cascade-candidate k):
         - Encode changed_text(j) and changed_text(k) with a sentence encoder.
         - If cosine similarity ≥ threshold → positive pair.
    4. For each direct j, sample ≤ NEG_PER_POS KEEP sentences → negative pairs.

Output JSONL (one pair per line):
    {
        "article_id":  str,
        "j_row":       int,   # row index into features array for sentence j
        "k_row":       int,   # row index into features array for sentence k
        "label":       0|1,   # 1 = positive (cascade), 0 = negative
        "similarity":  float, # cosine similarity (only for mined positives)
        "j_text":      str,   # changed text of j  (for inspection)
        "k_text":      str,   # changed text of k  (for inspection)
    }

Usage:
    python -m cascading.mine \\
        --data datasets/collated/train.jsonl \\
        --features features/train \\
        --out pairs/cascade_pairs.jsonl \\
        --threshold 0.6
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from cascading.mentions import Extractor, get_extractor

NEG_PER_POS = 3
LABELS = ("KEEP", "EDIT", "DROP")
KEEP, EDIT, DROP = 0, 1, 2


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _word_tokens(text: str) -> list[str]:
    """Lowercase word tokens, same split as the sentence aligner in collate.py."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def diffed_text(source: str, target: str) -> str:
    """Return the portion of target that differs from source (inserted/replaced words).

    Uses difflib SequenceMatcher to find matching blocks and returns the
    non-matching target text, which approximates 'what changed'.
    """
    import difflib
    src_toks = _word_tokens(source)
    tgt_toks = _word_tokens(target)
    matcher = difflib.SequenceMatcher(None, src_toks, tgt_toks, autojunk=False)
    changed_words = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            changed_words.extend(tgt_toks[j1:j2])
    return " ".join(changed_words) if changed_words else target


# ---------------------------------------------------------------------------
# Sentence encoder (lightweight, no GPU required for mining)
# ---------------------------------------------------------------------------

class SentenceEncoder:
    """Thin wrapper around sentence-transformers for cosine similarity."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for cascade pair mining.\n"
                    "Install with: pip install sentence-transformers"
                )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings [N, d]."""
        self._load()
        embs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalised vectors (dot product)."""
        return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Miner
# ---------------------------------------------------------------------------

class CascadeMiner:
    """Mine positive and negative pairs for the consequence head."""

    def __init__(
        self,
        extractor: Extractor | None = None,
        encoder: SentenceEncoder | None = None,
        threshold: float = 0.6,
        neg_per_pos: int = NEG_PER_POS,
        seed: int = 42,
    ):
        self.M = extractor or get_extractor()
        self.encoder = encoder or SentenceEncoder()
        self.threshold = threshold
        self.neg_per_pos = neg_per_pos
        import random
        self.rng = random.Random(seed)

    def _is_direct(self, source: str, target: str, evidence_mentions: frozenset[str]) -> bool:
        """True if the sentence's changed mentions overlap with the evidence."""
        changed = self.M.changed_mentions(source, target)
        return bool(changed & evidence_mentions)

    def mine_article(
        self,
        record: dict,
        row_offset: int,        # index of first sentence of this article in features array
    ) -> list[dict]:
        """Mine pairs from a single article record. Returns list of pair dicts."""
        sources = record.get("source_sentences", [])
        labels_raw = record.get("labels", [])
        evidence = record.get("evidence", [])

        # Collect target text per sentence: from labels[i]["target"] if present
        targets = [lbl.get("target", src) for src, lbl in zip(sources, labels_raw)]
        label_ids = [LABELS.index(lbl.get("label", "KEEP")) for lbl in labels_raw]

        # Evidence mentions (union across all snippets)
        evidence_mentions = self.M.from_evidence(evidence)

        # Categorise sentences
        direct_idx: list[int] = []
        cascade_cand_idx: list[int] = []
        keep_idx: list[int] = []

        for i, (src, tgt, lbl) in enumerate(zip(sources, targets, label_ids)):
            if lbl == KEEP:
                keep_idx.append(i)
            elif lbl in (EDIT, DROP):
                if self._is_direct(src, tgt, evidence_mentions):
                    direct_idx.append(i)
                else:
                    cascade_cand_idx.append(i)

        if not direct_idx or not cascade_cand_idx:
            return []

        # Encode changed texts for similarity comparison
        j_texts = [diffed_text(sources[j], targets[j]) for j in direct_idx]
        k_texts = [diffed_text(sources[k], targets[k]) for k in cascade_cand_idx]
        all_texts = j_texts + k_texts
        embs = self.encoder.encode(all_texts)
        j_embs = embs[:len(j_texts)]
        k_embs = embs[len(j_texts):]

        pairs: list[dict] = []

        # Positive pairs
        pos_count = 0
        for ji, j in enumerate(direct_idx):
            for ki, k in enumerate(cascade_cand_idx):
                sim = self.encoder.cosine(j_embs[ji], k_embs[ki])
                if sim >= self.threshold:
                    pairs.append({
                        "article_id": record.get("id", ""),
                        "j_row": row_offset + j,
                        "k_row": row_offset + k,
                        "label": 1,
                        "similarity": round(sim, 4),
                        "j_text": j_texts[ji],
                        "k_text": k_texts[ki],
                    })
                    pos_count += 1

        # Negative pairs: up to neg_per_pos * pos_count KEEP sentences per direct edit
        if pos_count > 0 and keep_idx:
            n_neg_needed = min(len(keep_idx), self.neg_per_pos * pos_count)
            chosen_keep = self.rng.sample(keep_idx, n_neg_needed)
            keep_texts = [sources[k] for k in chosen_keep]
            keep_embs = self.encoder.encode(keep_texts) if keep_texts else np.zeros((0, j_embs.shape[1]))

            for ji, j in enumerate(direct_idx):
                for ki, k in enumerate(chosen_keep):
                    pairs.append({
                        "article_id": record.get("id", ""),
                        "j_row": row_offset + j,
                        "k_row": row_offset + k,
                        "label": 0,
                        "similarity": round(self.encoder.cosine(j_embs[ji], keep_embs[ki]), 4),
                        "j_text": j_texts[ji],
                        "k_text": keep_texts[ki],
                    })

        return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_row_offsets(features_dir: Path) -> dict[str, int]:
    """Map article_id → first row index in the features array."""
    ids = json.loads((features_dir / "ids.json").read_text())
    article_idx = np.load(features_dir / "article.npy")
    offsets: dict[str, int] = {}
    for row, art_i in enumerate(article_idx):
        aid = ids[art_i]
        if aid not in offsets:
            offsets[aid] = row
    return offsets


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="collated JSONL (train split, silver labels)")
    parser.add_argument("--features", required=True,
                        help="feature directory from classifier.extract (for row offsets)")
    parser.add_argument("--out", required=True,
                        help="output JSONL path for mined pairs")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="minimum cosine similarity for a positive pair (default: 0.6)")
    parser.add_argument("--neg-per-pos", type=int, default=NEG_PER_POS,
                        help="negative KEEP sentences per positive (default: 3)")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="sentence encoder model name")
    parser.add_argument("--limit", type=int,
                        help="process only the first N articles (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features_dir = Path(args.features)
    if not features_dir.exists():
        raise SystemExit(f"Features directory not found: {features_dir}")

    print(f"Loading row offsets from {features_dir}...")
    row_offsets = _load_row_offsets(features_dir)
    print(f"  {len(row_offsets)} articles indexed")

    miner = CascadeMiner(
        encoder=SentenceEncoder(args.encoder),
        threshold=args.threshold,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs, total_pos, total_art = 0, 0, 0
    with open(args.data) as data_f, open(out_path, "w") as out_f:
        for line in data_f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            aid = record.get("id", "")
            if aid not in row_offsets:
                continue   # article not in extracted features

            pairs = miner.mine_article(record, row_offset=row_offsets[aid])
            for p in pairs:
                out_f.write(json.dumps(p) + "\n")
            total_pairs += len(pairs)
            total_pos += sum(p["label"] for p in pairs)
            total_art += 1

            if total_art % 1000 == 0:
                print(f"  {total_art} articles  {total_pairs} pairs "
                      f"({total_pos} positive)", flush=True)

            if args.limit and total_art >= args.limit:
                break

    print(f"\nDone. {total_art} articles → {total_pairs} pairs "
          f"({total_pos} positive, {total_pairs-total_pos} negative)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
