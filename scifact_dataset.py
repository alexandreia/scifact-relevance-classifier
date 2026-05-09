"""Build a custom SciFact claim-document classification dataset.

Pedagogical note:
SciFact is originally a retrieval benchmark: each claim has one or more
relevant scientific abstracts. Our assignment asks for a classifier, so we
convert retrieval judgments into pairwise classification rows:

    (claim, candidate document field) -> relevant / not_relevant

Negatives matter. If negatives are random only, the task can become too easy.
This builder therefore mixes random negatives with lexical hard negatives
selected by TF-IDF similarity to the claim.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scifact_config import (
    DATA_DIR,
    DATASET_CORPUS,
    DATASET_QRELS,
    FIELD_VARIANTS,
    LABEL2ID,
    RANDOM_SEED,
)


def document_text(doc: dict, field_variant: str) -> str:
    """Return exactly the document field used for this experiment condition."""
    title = (doc.get("title") or "").strip()
    abstract = (doc.get("text") or "").strip()
    if field_variant == "title":
        return title
    if field_variant == "abstract":
        return abstract
    if field_variant == "title_abstract":
        return f"{title}. {abstract}".strip()
    raise ValueError(f"Unknown field variant: {field_variant}")


def load_beir_scifact():
    """Load BEIR SciFact corpus, queries, and relevance judgments from HF."""
    corpus_ds = load_dataset(DATASET_CORPUS, "corpus")["corpus"]
    queries_ds = load_dataset(DATASET_CORPUS, "queries")["queries"]
    qrels_ds = load_dataset(DATASET_QRELS)

    corpus = {str(row["_id"]): row for row in corpus_ds}
    queries = {str(row["_id"]): row["text"] for row in queries_ds}
    qrels = {
        split: [(str(row["query-id"]), str(row["corpus-id"])) for row in qrels_ds[split]]
        for split in qrels_ds
    }
    return corpus, queries, qrels


def build_positive_lookup(qrels):
    positives = defaultdict(set)
    for split_rows in qrels.values():
        for query_id, doc_id in split_rows:
            positives[query_id].add(doc_id)
    return positives


def build_hard_negative_index(corpus: dict, field_variant: str):
    doc_ids = sorted(corpus)
    texts = [document_text(corpus[doc_id], field_variant) for doc_id in doc_ids]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=50_000)
    matrix = vectorizer.fit_transform(texts)
    return doc_ids, vectorizer, matrix


def choose_negatives(
    query: str,
    positive_doc_ids: set[str],
    all_doc_ids: list[str],
    vectorizer,
    doc_matrix,
    rng: random.Random,
    random_negatives: int,
    hard_negatives: int,
) -> list[str]:
    """Choose a stable mix of random and lexical hard negatives."""
    positive_doc_ids = {str(doc_id) for doc_id in positive_doc_ids}
    available = [doc_id for doc_id in all_doc_ids if doc_id not in positive_doc_ids]

    random_ids = rng.sample(available, min(random_negatives, len(available)))

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_matrix).ravel()
    hard_ids = []
    for idx in np.argsort(scores)[::-1]:
        doc_id = all_doc_ids[int(idx)]
        if doc_id not in positive_doc_ids and doc_id not in random_ids:
            hard_ids.append(doc_id)
        if len(hard_ids) >= hard_negatives:
            break

    return random_ids + hard_ids


def build_rows_for_variant(
    corpus: dict,
    queries: dict,
    qrels: dict,
    field_variant: str,
    random_negatives: int,
    hard_negatives: int,
):
    rng = random.Random(RANDOM_SEED)
    all_positives = build_positive_lookup(qrels)
    all_doc_ids, vectorizer, doc_matrix = build_hard_negative_index(corpus, field_variant)

    rows = []
    for split, split_qrels in qrels.items():
        query_to_positive_docs = defaultdict(set)
        for query_id, doc_id in split_qrels:
            query_to_positive_docs[query_id].add(doc_id)

        for query_id, positive_doc_ids in sorted(query_to_positive_docs.items()):
            claim = queries[query_id]
            for doc_id in sorted(positive_doc_ids):
                doc = corpus[doc_id]
                rows.append(make_row(split, field_variant, query_id, doc_id, claim, doc, "relevant"))

            negatives = choose_negatives(
                claim,
                all_positives[query_id],
                all_doc_ids,
                vectorizer,
                doc_matrix,
                rng,
                random_negatives=random_negatives,
                hard_negatives=hard_negatives,
            )
            for doc_id in negatives:
                doc = corpus[doc_id]
                rows.append(make_row(split, field_variant, query_id, doc_id, claim, doc, "not_relevant"))

    return rows


def make_row(split, field_variant, query_id, doc_id, claim, doc, label):
    return {
        "split": split,
        "field_variant": field_variant,
        "query_id": query_id,
        "doc_id": doc_id,
        "claim": claim,
        "title": (doc.get("title") or "").strip(),
        "abstract": (doc.get("text") or "").strip(),
        "document_text": document_text(doc, field_variant),
        "label": label,
        "label_id": LABEL2ID[label],
    }


def write_csv(path, rows):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-negatives", type=int, default=2)
    parser.add_argument("--hard-negatives", type=int, default=2)
    args = parser.parse_args()

    corpus, queries, qrels = load_beir_scifact()
    print(f"Loaded {len(corpus)} documents, {len(queries)} claims.")

    all_rows = []
    for variant in FIELD_VARIANTS:
        rows = build_rows_for_variant(
            corpus,
            queries,
            qrels,
            variant,
            random_negatives=args.random_negatives,
            hard_negatives=args.hard_negatives,
        )
        write_csv(DATA_DIR / f"scifact_pairs_{variant}.csv", rows)
        all_rows.extend(rows)
        print(f"{variant}: wrote {len(rows)} rows")

    write_csv(DATA_DIR / "scifact_pairs_all_fields.csv", all_rows)
    print(f"All variants: wrote {len(all_rows)} rows to {DATA_DIR}")


if __name__ == "__main__":
    main()
