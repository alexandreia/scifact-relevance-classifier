"""Embedding feature builders for claim-document relevance classification."""

from __future__ import annotations

import numpy as np


def e5_queries(texts: list[str]) -> list[str]:
    return [f"query: {text}" for text in texts]


def e5_passages(texts: list[str]) -> list[str]:
    return [f"passage: {text}" for text in texts]


def pair_features(model, claims: list[str], documents: list[str], show_progress_bar=False):
    """Build standard sentence-pair features from two embedding vectors.

    q and d alone give the classifier raw semantic position. abs(q-d) exposes
    distance dimensions. q*d exposes alignment dimensions. cosine gives a
    single retrieval-style similarity signal.
    """
    q = model.encode(
        e5_queries(claims),
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    d = model.encode(
        e5_passages(documents),
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    cosine = np.sum(q * d, axis=1, keepdims=True)
    return np.hstack([q, d, np.abs(q - d), q * d, cosine])
