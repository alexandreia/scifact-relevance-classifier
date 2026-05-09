"""Train SciFact relevance classifiers on top of embedding features.

This is the main assignment artifact:
  input  = scientific claim + candidate paper field
  model  = frozen embedding model + trained sklearn classifier
  output = relevant / not_relevant

Run:
  python scifact_dataset.py
  python train_scifact_classifier.py
"""

from __future__ import annotations

import csv
import json
import pickle
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

from scifact_config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    FIELD_VARIANTS,
    ID2LABEL,
    LABEL2ID,
    OUT_DIR,
    RANDOM_SEED,
)
from scifact_features import pair_features


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classifier_factories():
    """Return one factory per classifier family we want to compare.

    Pedagogical note — three model families are represented here:
      - Linear models     : logistic_regression, linear_svm
      - Tree ensembles    : random_forest, hist_gradient_boosting
      - Small neural net  : mlp

    Comparing across families tells us about the *shape* of the decision
    boundary in the embedding space:
      - If linear models win, the embeddings are already (close to) linearly
        separable — the encoder is doing all the work.
      - If trees or the MLP win by a meaningful margin, useful non-linear
        feature interactions exist that the linear models cannot exploit.

    Every factory is a 0-arg lambda so each (variant, classifier) gets a
    fresh, unfit estimator. Re-using a single fitted instance across variants
    would silently leak training data between conditions.

    Scaling: linear models and MLP are sensitive to feature magnitudes, so
    they sit inside a StandardScaler pipeline. Tree-based methods split on
    feature *order*, not magnitude, so they don't need scaling.
    """
    return {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=RANDOM_SEED),
        ),
        "linear_svm": lambda: make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        # MLPClassifier does not accept class_weight directly. With 64/36 class
        # imbalance the bias is mild; we rely on MLP regularisation + early
        # stopping. If results show a strong majority-class bias for this model,
        # we can switch to passing sample_weight via fit_params.
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256,),
                activation="relu",
                solver="adam",
                max_iter=200,
                early_stopping=True,
                random_state=RANDOM_SEED,
            ),
        ),
    }


def evaluate_classifier(name, clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    labels = [ID2LABEL[i] for i in sorted(ID2LABEL)]
    return {
        "classifier": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "report": classification_report(
            y_test,
            pred,
            labels=sorted(ID2LABEL),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }, clf


def train_variant(model, variant: str):
    rows = read_rows(DATA_DIR / f"scifact_pairs_{variant}.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]

    print(f"\nVariant: {variant}")
    print(f"  train rows: {len(train_rows)} | test rows: {len(test_rows)}")

    X_train = pair_features(
        model,
        [row["claim"] for row in train_rows],
        [row["document_text"] for row in train_rows],
        show_progress_bar=True,
    )
    X_test = pair_features(
        model,
        [row["claim"] for row in test_rows],
        [row["document_text"] for row in test_rows],
        show_progress_bar=True,
    )
    y_train = np.asarray([int(row["label_id"]) for row in train_rows])
    y_test = np.asarray([int(row["label_id"]) for row in test_rows])

    results = []
    trained = {}
    for clf_name, factory in classifier_factories().items():
        result, clf = evaluate_classifier(clf_name, factory(), X_train, y_train, X_test, y_test)
        results.append(result)
        trained[clf_name] = clf
        print(
            f"  {clf_name}: accuracy={result['accuracy']:.3f} "
            f"macro-F1={result['macro_f1']:.3f}"
        )

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


def save_final_model(model, clf, best_summary):
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    model.save(str(OUT_DIR / "embedding_model"))

    metadata = {
        "project": "Scientific evidence relevance classification",
        "embedding_model": EMBEDDING_MODEL,
        "field_variant": best_summary["variant"],
        "classifier": best_summary["best_classifier"],
        "feature_dim": best_summary["feature_dim"],
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
    }
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    summaries = []
    best_models = {}
    for variant in FIELD_VARIANTS:
        summary, clf = train_variant(model, variant)
        summaries.append(summary)
        best_models[variant] = clf

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    best_summary = max(summaries, key=lambda r: (r["best_macro_f1"], r["best_accuracy"]))
    save_final_model(model, best_models[best_summary["variant"]], best_summary)

    print("\nBest model:")
    print(
        f"  {best_summary['variant']} + {best_summary['best_classifier']} "
        f"macro-F1={best_summary['best_macro_f1']:.3f}"
    )
    print(f"Saved artefacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
