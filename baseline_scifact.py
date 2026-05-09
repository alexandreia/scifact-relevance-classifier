"""baseline_scifact.py — the simplest embedding + classifier pipeline you can run.

Why this exists
---------------
Our "real" pipeline (train_scifact_classifier.py) feeds the classifier 1537
features per pair:
    [q, d, |q-d|, q*d, cosine(q,d)]

That is a lot of feature engineering. Does it actually help, or could a naive
baseline get similar numbers?

This baseline is intentionally minimal:
    1. Embed the claim with E5's "query:" prefix.
    2. Embed the document with E5's "passage:" prefix.
    3. Concatenate the two vectors. No |q-d|, no q*d, no cosine.
    4. Train logistic regression on top of those 768 features.
    5. Evaluate on the same train/test split and the same three field variants
       so the comparison with the elaborate pipeline is apples-to-apples.

Two possible outcomes after running this:
    - The elaborate pipeline beats the baseline by a real margin
      -> the |q-d| / q*d interaction features earn their keep, and the report
         can quantify how much.
    - The elaborate pipeline barely beats the baseline
      -> most of the signal is already in the raw embeddings; the extra
         feature engineering is decoration, and the report should say so.

Either result is a real finding. Running an unanchored single number is not.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Reuse the embedding model already saved by the elaborate pipeline so we are
# not re-downloading 100MB every time we run.
EMBEDDING_MODEL = "intfloat/e5-small-v2"
LOCAL_MODEL = Path("artefacts_scifact/embedding_model")
DATA_DIR = Path("data_scifact")
OUT_PATH = Path("artefacts_scifact/baseline_results.json")

VARIANTS = ("title", "abstract", "title_abstract")
LABEL2ID = {"not_relevant": 0, "relevant": 1}


def read_rows(variant: str, split: str):
    """Load one split (train or test) for one field variant from disk."""
    path = DATA_DIR / f"scifact_pairs_{variant}.csv"
    with open(path, encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["split"] == split]


def encode_pairs(model, claims, documents):
    """The core baseline: separate E5-asymmetric encoding, then concatenate.

    Pedagogical note: E5 was trained asymmetrically. Queries are tagged with
    "query:" and documents with "passage:". Using the wrong prefix quietly
    degrades quality, especially on retrieval-style tasks. So even our simplest
    baseline respects E5's expected input format.

    Output: shape (n, 768) — 384 dims for the claim, 384 dims for the document.
    """
    q = model.encode(
        [f"query: {c}" for c in claims],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    d = model.encode(
        [f"passage: {doc}" for doc in documents],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.hstack([q, d])


def evaluate_variant(model, variant: str):
    train_rows = read_rows(variant, "train")
    test_rows = read_rows(variant, "test")
    print(f"\n=== {variant}: {len(train_rows)} train, {len(test_rows)} test ===")

    X_train = encode_pairs(
        model,
        [r["claim"] for r in train_rows],
        [r["document_text"] for r in train_rows],
    )
    X_test = encode_pairs(
        model,
        [r["claim"] for r in test_rows],
        [r["document_text"] for r in test_rows],
    )
    y_train = np.array([LABEL2ID[r["label"]] for r in train_rows])
    y_test = np.array([LABEL2ID[r["label"]] for r in test_rows])

    # StandardScaler matters here: LR with regularisation likes zero-mean, unit-
    # variance inputs. class_weight="balanced" mirrors the elaborate pipeline so
    # the comparison is fair (the test set is 64% not_relevant).
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return {
        "variant": variant,
        "feature_dim": int(X_train.shape[1]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "report": classification_report(
            y_test,
            pred,
            target_names=["not_relevant", "relevant"],
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else EMBEDDING_MODEL
    print(f"Loading embedding model: {model_path}")
    model = SentenceTransformer(model_path)

    results = [evaluate_variant(model, v) for v in VARIANTS]

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n--- Baseline summary (concat embedding + logistic regression) ---")
    print(f"{'variant':>20}  {'dim':>4}  {'acc':>5}  {'macro-F1':>8}")
    for r in results:
        print(f"{r['variant']:>20}  {r['feature_dim']:>4}  {r['accuracy']:.3f}     {r['macro_f1']:.3f}")
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
