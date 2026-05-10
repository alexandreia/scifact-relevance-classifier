"""scripts/encode_corpus.py — pre-encode the SciFact corpus for retrieval.

Why this exists
---------------
The Space needs to perform retrieve-and-rerank: given a user query, find the
most relevant abstracts in the SciFact corpus, then re-rank them with the
trained classifier.

Encoding ~5000 abstracts with e5-small-v2 takes ~2 minutes on CPU. Doing
that on every Space cold start would be slow and wasteful. Instead we encode
once locally, save the embeddings as a NumPy array, and ship them to the HF
model repo as a static artifact. The Space just downloads them.

Output (in artefacts_scifact/):
    corpus_embeddings.npy    (N, 384) float32 — passage embeddings, L2-normalised
    corpus_meta.csv          id, title, abstract  (parallel rows to embeddings)

Run:
    cd "Assignment 1"
    source venv/bin/activate
    python scripts/encode_corpus.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
ARTEFACTS = ROOT / "artefacts_scifact"
ENCODER_PATH = str(ARTEFACTS / "embedding_model") if (ARTEFACTS / "embedding_model").exists() else "intfloat/e5-small-v2"

OUT_EMB = ARTEFACTS / "corpus_embeddings.npy"
OUT_META = ARTEFACTS / "corpus_meta.csv"


def main():
    print(f"Loading SciFact corpus from BEIR/scifact ...")
    corpus_ds = load_dataset("BeIR/scifact", "corpus")["corpus"]
    print(f"  loaded {len(corpus_ds)} documents")

    # Build texts in the same form the classifier was trained on:
    # "<title>. <abstract>" — matching the title_abstract field variant.
    ids, titles, abstracts, texts = [], [], [], []
    for row in corpus_ds:
        doc_id = str(row["_id"])
        title = (row.get("title") or "").strip()
        abstract = (row.get("text") or "").strip()
        text = f"{title}. {abstract}".strip().rstrip(".")
        ids.append(doc_id)
        titles.append(title)
        abstracts.append(abstract)
        texts.append(text)

    print(f"\nLoading encoder: {ENCODER_PATH}")
    encoder = SentenceTransformer(ENCODER_PATH)

    print(f"\nEncoding {len(texts)} passages with E5 'passage:' prefix ...")
    # batch_size=32 is a sane default; raise if you have more RAM.
    embeddings = encoder.encode(
        [f"passage: {t}" for t in texts],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print(f"\nEmbeddings shape: {embeddings.shape}, dtype={embeddings.dtype}")
    print(f"Estimated size: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    ARTEFACTS.mkdir(exist_ok=True)
    np.save(OUT_EMB, embeddings)
    print(f"Saved embeddings -> {OUT_EMB}")

    with open(OUT_META, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "title", "abstract"])
        for doc_id, title, abstract in zip(ids, titles, abstracts):
            w.writerow([doc_id, title, abstract])
    print(f"Saved metadata   -> {OUT_META} ({len(ids)} rows)")

    print("\nDone. Next: re-run scripts/push_model.py to upload these to HF.")


if __name__ == "__main__":
    main()
