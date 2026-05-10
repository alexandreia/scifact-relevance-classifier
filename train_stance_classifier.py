"""Train SciFact 3-class stance classifiers on top of embedding features.

Same 5 classifier families as the binary relevance pipeline; only the labels
and feature dimensions are unchanged. Output:
  artefacts_stance/
    classifier_stance.pkl
    metadata_stance.json
    experiment_results_stance.json
"""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scifact_config import EMBEDDING_MODEL, FIELD_VARIANTS, RANDOM_SEED
from scifact_features import pair_features
from scifact_dataset_stance import STANCE_LABEL2ID, STANCE_ID2LABEL


DATA_DIR = Path("data_scifact_stance")
OUT_DIR = Path("artefacts_stance")
LABELS_ORDER = ["NEI", "REFUTES", "SUPPORTS"]


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classifier_factories():
    """Same 5 families as the binary pipeline, all multi-class capable."""
    return {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced",
                # multinomial loss is the default for multi-class in sklearn 1.5+;
                # multi_class= and solver= no longer need to be specified.
                random_state=RANDOM_SEED,
            ),
        ),
        "linear_svm": lambda: make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256,), activation="relu", solver="adam",
                max_iter=200, early_stopping=True, random_state=RANDOM_SEED,
            ),
        ),
    }


def evaluate(name, clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    label_ids = [STANCE_LABEL2ID[lbl] for lbl in LABELS_ORDER]
    return {
        "classifier": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "report": classification_report(
            y_test, pred,
            labels=label_ids, target_names=LABELS_ORDER,
            output_dict=True, zero_division=0,
        ),
    }, clf


def train_variant(model, variant: str):
    rows = read_rows(DATA_DIR / f"scifact_stance_{variant}.csv")
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    print(f"\nVariant: {variant}")
    print(f"  train: {len(train_rows)} rows | test: {len(test_rows)} rows")
    print(f"  train classes: {dict(Counter(r['label'] for r in train_rows))}")
    print(f"  test  classes: {dict(Counter(r['label'] for r in test_rows))}")

    X_train = pair_features(model,
                            [r["claim"] for r in train_rows],
                            [r["document_text"] for r in train_rows],
                            show_progress_bar=True)
    X_test = pair_features(model,
                           [r["claim"] for r in test_rows],
                           [r["document_text"] for r in test_rows],
                           show_progress_bar=True)
    y_train = np.asarray([int(r["label_id"]) for r in train_rows])
    y_test = np.asarray([int(r["label_id"]) for r in test_rows])

    results = []
    trained = {}
    for name, factory in classifier_factories().items():
        result, clf = evaluate(name, factory(), X_train, y_train, X_test, y_test)
        results.append(result)
        trained[name] = clf
        print(f"  {name:24s} acc={result['accuracy']:.3f} macro-F1={result['macro_f1']:.3f}")

    best = max(results, key=lambda r: (r["macro_f1"], r["accuracy"]))
    return {
        "variant": variant,
        "feature_dim": int(X_train.shape[1]),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "results": results,
        "best_classifier": best["classifier"],
        "best_macro_f1": best["macro_f1"],
        "best_accuracy": best["accuracy"],
    }, trained[best["classifier"]]


def save_final(model, clf, summary):
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "classifier_stance.pkl", "wb") as f:
        pickle.dump(clf, f)

    metadata = {
        "project": "Scientific evidence stance classification (3-class)",
        "embedding_model": EMBEDDING_MODEL,
        "field_variant": summary["variant"],
        "classifier": summary["best_classifier"],
        "feature_dim": summary["feature_dim"],
        "label2id": STANCE_LABEL2ID,
        "id2label": {str(k): v for k, v in STANCE_ID2LABEL.items()},
    }
    with open(OUT_DIR / "metadata_stance.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    # Reuse local copy if present (saved by train_scifact_classifier.py)
    local = Path("artefacts_scifact/embedding_model")
    model = SentenceTransformer(str(local) if local.exists() else EMBEDDING_MODEL)

    summaries = []
    best_models = {}
    for variant in FIELD_VARIANTS:
        s, c = train_variant(model, variant)
        summaries.append(s)
        best_models[variant] = c

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "experiment_results_stance.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    best_summary = max(summaries, key=lambda r: (r["best_macro_f1"], r["best_accuracy"]))
    save_final(model, best_models[best_summary["variant"]], best_summary)

    print("\nBest stance model:")
    print(f"  {best_summary['variant']} + {best_summary['best_classifier']} "
          f"macro-F1={best_summary['best_macro_f1']:.3f}")
    print(f"Saved artefacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
