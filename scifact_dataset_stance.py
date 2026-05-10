"""Build a 3-class stance dataset for SciFact.

Why this exists
---------------
The original BEIR/scifact-qrels gives only binary relevance (a paper is or is
not cited as evidence for a claim). The original allenai/scifact dataset
preserves the *stance* of each cited paper: SUPPORTS or CONTRADICTS the claim.

This script builds (claim, document, stance_label) triples where:

    label        meaning                                      origin
    --------     -----------------------------------------    -------------
    SUPPORTS     this paper provides evidence FOR the claim   allenai/scifact "evidence"
    REFUTES      this paper provides evidence AGAINST it      allenai/scifact "evidence"
    NEI          this paper is on-topic but takes no stance   sampled (random + TF-IDF hard)

NEI ("not enough info") is constructed exactly like the binary not_relevant
class in scifact_dataset.py — a mix of random and TF-IDF hard negatives — so
the comparison with the binary version is apples-to-apples.

Output:
    data_scifact_stance/scifact_stance_<variant>.csv
        for variant in (title, abstract, title_abstract)
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scifact_config import FIELD_VARIANTS, RANDOM_SEED


# 3-class label scheme (NEI=0 keeps "negative-ish" as the lowest id, just like LABEL2ID)
STANCE_LABEL2ID = {"NEI": 0, "REFUTES": 1, "SUPPORTS": 2}
STANCE_ID2LABEL = {v: k for k, v in STANCE_LABEL2ID.items()}

DATA_DIR = Path("data_scifact_stance")


def document_text(doc: dict, field_variant: str) -> str:
    title = (doc.get("title") or "").strip()
    abstract = (doc.get("text") or doc.get("abstract") or "").strip()
    if field_variant == "title":
        return title
    if field_variant == "abstract":
        return abstract
    if field_variant == "title_abstract":
        return f"{title}. {abstract}".strip()
    raise ValueError(f"Unknown field variant: {field_variant}")


SCIFACT_DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_CACHE = Path(".cache/scifact")


def _download_scifact_archive() -> Path:
    """Download + extract AllenAI's SciFact release tarball. Cached locally."""
    import tarfile
    import urllib.request

    SCIFACT_CACHE.mkdir(parents=True, exist_ok=True)
    tar_path = SCIFACT_CACHE / "data.tar.gz"
    data_dir = SCIFACT_CACHE / "data"

    if data_dir.exists() and (data_dir / "corpus.jsonl").exists():
        return data_dir

    if not tar_path.exists():
        print(f"  downloading {SCIFACT_DATA_URL} ...")
        urllib.request.urlretrieve(SCIFACT_DATA_URL, tar_path)
    print("  extracting tarball ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(SCIFACT_CACHE)

    if not (data_dir / "corpus.jsonl").exists():
        # Some releases nest things differently; find any corpus.jsonl in the cache
        candidates = list(SCIFACT_CACHE.rglob("corpus.jsonl"))
        if candidates:
            return candidates[0].parent
        raise FileNotFoundError("corpus.jsonl not found after extraction")
    return data_dir


def load_scifact_with_stance():
    """Load corpus + stance-annotated claims directly from AllenAI's release.

    Bypasses the HF `datasets` library because allenai/scifact uses a deprecated
    Python loading script. Reads the canonical JSONL release files instead.

    Returns:
        corpus: dict[str, dict] — doc_id -> {title, abstract}
        claims: dict[split_name, list[dict]] — each claim has {id, claim, evidence}
            where evidence = {doc_id: stance_label}  (SUPPORT or CONTRADICT)
    """
    import json

    data_dir = _download_scifact_archive()

    # Corpus (one JSON per line)
    corpus = {}
    with open(data_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            abstract = row.get("abstract") or ""
            if isinstance(abstract, list):
                abstract = " ".join(abstract)
            corpus[doc_id] = {
                "title": row.get("title", "").strip(),
                "abstract": abstract.strip(),
            }
    print(f"  corpus: {len(corpus)} documents")

    # Claims (train + dev). AllenAI labels: SUPPORT / CONTRADICT.
    file_to_split = {
        "claims_train.jsonl": "train",
        "claims_dev.jsonl": "validation",
    }
    claims_data = {}
    for fname, split_name in file_to_split.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                evidence_raw = row.get("evidence") or {}

                # evidence is a dict: doc_id_str -> list of {"sentences": [...], "label": "..."}
                evidence_map = {}
                for did, ev_list in evidence_raw.items():
                    if not ev_list:
                        continue
                    # Use the first rationale's label (they agree per doc in SciFact)
                    label = ev_list[0].get("label", "").upper()
                    if label in ("SUPPORT", "SUPPORTS", "CONTRADICT", "REFUTES"):
                        evidence_map[str(did)] = label

                if evidence_map:
                    rows.append({
                        "id": str(row["id"]),
                        "claim": row["claim"],
                        "evidence": evidence_map,
                    })
        claims_data[split_name] = rows
        print(f"  claims/{split_name}: {len(rows)} with stance evidence")
    return corpus, claims_data


def normalize_stance(label: str) -> str | None:
    """Map AllenAI stance vocabulary onto our 3-class scheme."""
    if label is None:
        return None
    label = label.upper()
    if label in ("SUPPORT", "SUPPORTS"):
        return "SUPPORTS"
    if label in ("CONTRADICT", "REFUTES"):
        return "REFUTES"
    if label in ("NOT_ENOUGH_INFO", "NEI"):
        return "NEI"
    return None


def build_hard_negative_index(corpus: dict, field_variant: str):
    doc_ids = sorted(corpus)
    texts = [document_text(corpus[doc_id], field_variant) for doc_id in doc_ids]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=50_000)
    matrix = vectorizer.fit_transform(texts)
    return doc_ids, vectorizer, matrix


def sample_nei_negatives(claim, evidence_doc_ids, all_doc_ids, vectorizer, doc_matrix,
                         rng, n_random, n_hard):
    """Sample NEI examples for one claim: mix of random + TF-IDF hard negatives."""
    excluded = set(evidence_doc_ids)
    available = [did for did in all_doc_ids if did not in excluded]

    random_ids = rng.sample(available, min(n_random, len(available)))

    qv = vectorizer.transform([claim])
    scores = cosine_similarity(qv, doc_matrix).ravel()
    hard_ids = []
    for idx in np.argsort(scores)[::-1]:
        did = all_doc_ids[int(idx)]
        if did not in excluded and did not in random_ids:
            hard_ids.append(did)
        if len(hard_ids) >= n_hard:
            break
    return random_ids + hard_ids


def make_row(split, variant, query_id, doc_id, claim, doc, label):
    return {
        "split": split,
        "field_variant": variant,
        "query_id": query_id,
        "doc_id": doc_id,
        "claim": claim,
        "title": (doc.get("title") or "").strip(),
        "abstract": (doc.get("abstract") or "").strip(),
        "document_text": document_text(doc, variant),
        "label": label,
        "label_id": STANCE_LABEL2ID[label],
    }


def build_rows_for_variant(corpus, claims_data, variant, n_random, n_hard):
    rng = random.Random(RANDOM_SEED)
    all_doc_ids, vectorizer, doc_matrix = build_hard_negative_index(corpus, variant)

    # AllenAI uses train/validation; we use validation as our test split.
    SPLIT_MAP = {"train": "train", "validation": "test"}

    rows = []
    for src_split, dst_split in SPLIT_MAP.items():
        for cl in claims_data[src_split]:
            cid = cl["id"]
            claim_text = cl["claim"]
            evidence_doc_ids = []

            # Positives (SUPPORTS or REFUTES from evidence map)
            for did, raw_label in cl["evidence"].items():
                norm = normalize_stance(raw_label)
                if norm not in ("SUPPORTS", "REFUTES"):
                    continue
                if did not in corpus:
                    continue
                evidence_doc_ids.append(did)
                rows.append(make_row(dst_split, variant, cid, did, claim_text,
                                     corpus[did], norm))

            # Negatives = NEI samples
            negs = sample_nei_negatives(
                claim_text, evidence_doc_ids, all_doc_ids,
                vectorizer, doc_matrix, rng,
                n_random=n_random, n_hard=n_hard,
            )
            for did in negs:
                rows.append(make_row(dst_split, variant, cid, did, claim_text,
                                     corpus[did], "NEI"))

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random-negatives", type=int, default=2,
                   help="random NEI samples per claim")
    p.add_argument("--hard-negatives", type=int, default=2,
                   help="TF-IDF hard NEI samples per claim")
    args = p.parse_args()

    print("Loading allenai/scifact (corpus + stance-annotated claims) ...")
    corpus, claims_data = load_scifact_with_stance()

    for variant in FIELD_VARIANTS:
        print(f"\nBuilding variant: {variant}")
        rows = build_rows_for_variant(corpus, claims_data, variant,
                                      n_random=args.random_negatives,
                                      n_hard=args.hard_negatives)
        out = DATA_DIR / f"scifact_stance_{variant}.csv"
        write_csv(out, rows)

        # Print class balance per split
        from collections import Counter
        for split in ("train", "test"):
            c = Counter(r["label"] for r in rows if r["split"] == split)
            print(f"  {split}: {sum(c.values())} rows -> {dict(c)}")
        print(f"  wrote {out}")

    print(f"\nDone. CSVs in {DATA_DIR}/")


if __name__ == "__main__":
    main()
